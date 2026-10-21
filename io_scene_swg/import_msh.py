# MIT License
#
# Copyright (c) 2022 Nick Rafalski
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import base64, os, bpy, time, datetime, math
import numpy as np
from mathutils import Matrix, Vector, Color, Quaternion, Euler

from bpy_extras.io_utils import unpack_list
from bpy_extras.wm_utils.progress_report import ProgressReport
from bpy_extras.object_utils import AddObjectHelper, object_data_add
from bpy_extras import node_shader_utils
from bpy import utils

from . import vertex_buffer_format
from . import swg_types
from . import support

_configured_shaders = {}   # material name -> shader path it was last configured from

def _degenerate_faces(corners):
	"""Faces whose three corners are not all different."""
	a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]
	return (a == b) | (b == c) | (a == c)

def _repeated_faces(corners):
	"""Faces that repeat an earlier face's vertex set, in any order."""
	first_face, _ = support.unique_rows(np.sort(corners, axis=1))
	repeated = np.ones(len(corners), dtype=bool)
	repeated[first_face] = False
	return repeated

def weld_coincident_vertices(positions, corners):
	"""Merge vertices with identical coordinates (the first occurrence survives), remap the corners,
	and drop faces that collapsed or repeat an earlier face's vertex set.
	Returns (positions, corners, keep) where keep marks the faces that were kept."""
	first, inverse = support.unique_rows(positions)
	new_positions = positions[first]
	new_corners = inverse[corners]
	keep = ~(_degenerate_faces(new_corners) | _repeated_faces(new_corners))
	return new_positions, new_corners[keep], keep

def _flat(a):
	"""Contiguous 1-D array for foreach_set."""
	return np.ascontiguousarray(a).reshape(-1)

class _MeshArrays:
	"""Mesh data for all SPS as flat arrays, filled one SPS at a time.

	positions   (V, 3) float32   Blender axis order
	corners     (F, 3) int64     vertex of each face corner, (p3, p2, p1) order
	mat_index   (F,)   int32     material slot per face
	loop_attrs  name -> (3F, k)  per-corner 'normal', 'color0'/'color1' (if any SPS has them), 'uv0'..'uvN'
	"""

	def __init__(self, spss):
		num_verts = sum(len(sps.vertex_data) for sps in spss)
		num_faces = sum(len(sps.indices) for sps in spss)
		num_loops = num_faces * 3
		self.max_uv_sets = max((sps.getNumUVSets() for sps in spss), default=0)
		self.positions = np.zeros((num_verts, 3), np.float32)
		self.corners = np.zeros((num_faces, 3), np.int64)
		self.mat_index = np.zeros(num_faces, np.int32)
		self.loop_attrs = {"normal": np.zeros((num_loops, 3), np.float32)}
		if any(sps.hasColor0() for sps in spss):
			self.loop_attrs["color0"] = np.ones((num_loops, 4), np.float64)   # corners of an SPS without this colour stay white
		if any(sps.hasColor1() for sps in spss):
			self.loop_attrs["color1"] = np.ones((num_loops, 4), np.float64)
		for depth in range(self.max_uv_sets):
			self.loop_attrs[f"uv{depth}"] = np.zeros((num_loops, 2), np.float64)   # corners of an SPS without this depth stay (0, 0)
		self._next_vert = 0
		self._next_face = 0

	@property
	def num_faces(self):
		return len(self.corners)

	@property
	def num_loops(self):
		return len(self.corners) * 3

	def add_sps(self, sps, material_index, flip_uv_vertical):
		"""Append one SPS's vertices, faces (offset by the vertices already added) and per-corner values."""
		vertex_data = sps.vertex_data
		fields = vertex_data.dtype.names
		n = len(vertex_data)
		t = len(sps.indices)
		verts = slice(self._next_vert, self._next_vert + n)
		faces = slice(self._next_face, self._next_face + t)
		loops = slice(self._next_face * 3, (self._next_face + t) * 3)
		corners = sps.indices[:, ::-1]      # (p3, p2, p1) per triangle
		corner_vert = corners.reshape(-1)   # local vertex index of every corner

		if 'pos' in fields:
			self.positions[verts] = support.convert_vector3_array(vertex_data['pos'])
		self.corners[faces] = corners + self._next_vert
		self.mat_index[faces] = material_index
		if 'normal' in fields:
			self.loop_attrs["normal"][loops] = support.convert_vector3_array(vertex_data['normal'])[corner_vert]
		for name in ("color0", "color1"):
			if name in fields:
				self.loop_attrs[name][loops] = vertex_buffer_format.color_bytes_to_rgba(vertex_data[name])[corner_vert]
		for depth in range(sps.getNumUVSets()):
			src = vertex_data[f'uv{depth}'].reshape(n, -1)[corner_vert]
			uv = self.loop_attrs[f'uv{depth}'][loops]
			width = min(2, src.shape[1])   # DOT3 sets are 4 wide; keep the uv part
			uv[:, :width] = src[:, :width]
			if flip_uv_vertical:
				uv[:, 1] = 1.0 - uv[:, 1]
		self._next_vert += n
		self._next_face += t

	def weld_duplicate_vertices(self):
		"""Weld identical positions and drop collapsed or repeated faces. Returns (verts removed, faces removed)."""
		verts_before = len(self.positions)
		self.positions, self.corners, keep = weld_coincident_vertices(self.positions, self.corners)
		if not keep.all():
			self.mat_index = self.mat_index[keep]
			loop_keep = np.repeat(keep, 3)
			self.loop_attrs = {name: a[loop_keep] for name, a in self.loop_attrs.items()}
		return verts_before - len(self.positions), int((~keep).sum())

def _fill_blender_mesh(mesh, arrays):
	"""Write the arrays into an empty Blender mesh."""
	num_faces = arrays.num_faces
	num_loops = arrays.num_loops
	mesh.vertices.add(len(arrays.positions))
	mesh.loops.add(num_loops)
	mesh.polygons.add(num_faces)
	mesh.vertices.foreach_set("co", _flat(arrays.positions))
	if num_faces > 0:
		mesh.polygons.foreach_set("loop_total", np.full(num_faces, 3, dtype=np.int32))
		mesh.polygons.foreach_set("loop_start", np.arange(0, num_loops, 3, dtype=np.int32))
		mesh.polygons.foreach_set("vertices", _flat(arrays.corners).astype(np.int32))
		mesh.update(calc_edges=True, calc_edges_loose=False)
		mesh.polygons.foreach_set("material_index", np.ascontiguousarray(arrays.mat_index))

	mesh.use_auto_smooth = True
	mesh.normals_split_custom_set(np.ascontiguousarray(arrays.loop_attrs["normal"]))

	for depth in range(arrays.max_uv_sets):
		flat_uv = arrays.loop_attrs[f'uv{depth}']
		uv_layer = mesh.uv_layers.new(name=f'uvmap-{depth}')
		if len(uv_layer.data) == num_loops:
			uv_layer.data.foreach_set("uv", _flat(flat_uv))
		else:
			print(f"*** UVMap: {uv_layer.name} has {len(uv_layer.data)} loops but {num_loops} were expected")
			for i in range(min(len(uv_layer.data), num_loops)):
				uv_layer.data[i].uv = flat_uv[i]

	for name in ("color0", "color1"):
		if name not in arrays.loop_attrs:
			continue
		flat = arrays.loop_attrs[name]
		mesh.vertex_colors.new(name=name)
		color_layer = mesh.vertex_colors[name]
		if len(color_layer.data) == len(flat):
			color_layer.data.foreach_set("color", _flat(flat))
		else:
			print(f"Warning! {name} only has {len(flat)} but need {len(color_layer.data)}")

def images_in_tree(node_tree):
	for node in node_tree.nodes:
		if hasattr(node, "image"):
			yield node.image
		if hasattr(node, "node_tree"):
			yield from images_in_tree(node.node_tree)

def all_images():
	for mat in bpy.data.materials:
		if mat.use_nodes:
			return set(filter(None, images_in_tree(mat.node_tree)))

def rotate_object(obj, rot_mat):
	# decompose world_matrix's components, and from them assemble 4x4 matrices
	orig_loc, orig_rot, orig_scale = obj.matrix_world.decompose()
	#
	orig_loc_mat   = Matrix.Translation(orig_loc)
	orig_rot_mat   = orig_rot.to_matrix().to_4x4()
	orig_scale_mat = (Matrix.Scale(orig_scale[0],4,(1,0,0)) @ 
					  Matrix.Scale(orig_scale[1],4,(0,1,0)) @ 
					  Matrix.Scale(orig_scale[2],4,(0,0,1)))
	#
	# assemble the new matrix
	obj.matrix_world = orig_loc_mat @ rot_mat @ orig_rot_mat @ orig_scale_mat

def import_msh(context,
			   filepath,
			   parent=None,
			   flip_uv_vertical=False,
			   remove_duplicate_verts=True,
			   just_the_mesh = False,
	):  

	print(f'Importing msh: {filepath} Flip UV: {flip_uv_vertical}')

	swg_root = context.preferences.addons[__package__].preferences.swg_root
	msh = swg_types.SWGMesh(filepath, swg_root)
	if not msh.load():
		return {'CANCELLED'}
	
		
	name=os.path.basename(filepath).rsplit( ".", 1 )[ 0 ]
	mesh = bpy.data.meshes.new(name=f'{name}-mesh')
	obj = bpy.data.objects.new(name, mesh)

	if parent == None:
		parent = bpy.context.scene.collection
	
	parent.objects.link(obj)

	# gather every SPS into flat arrays, then build the mesh in one go
	arrays = _MeshArrays(msh.spss)
	for index, sps in enumerate(msh.spss):
		num_uv_sets = sps.getNumUVSets()
		mat_name = sps.stripped_shader_name()
		material = None

		for mat in bpy.data.materials:
			if mat.name == mat_name:
				material = mat

		if material == None:
			material = bpy.data.materials.new(sps.stripped_shader_name())

		material["DOT3"] = sps.hasDOT3()
		material["UVSets"] = num_uv_sets
		material["Color0"] = sps.hasColor0()
		material["Color1"] = sps.hasColor1()

		if sps.real_shader and not (material.use_nodes and _configured_shaders.get(material.name) == sps.full_shader_path):
			tex_to_png = context.preferences.addons[__package__].preferences.convert_tex_to_png
			support.configure_material_from_swg_shader(material, sps.real_shader, swg_root, tex_to_png)
			_configured_shaders[material.name] = sps.full_shader_path

		mesh.materials.append(material)
		arrays.add_sps(sps, index, flip_uv_vertical)

	if remove_duplicate_verts and len(arrays.positions) > 0:
		verts_removed, faces_removed = arrays.weld_duplicate_vertices()
		print(f"Removed: {verts_removed} verts, {faces_removed} faces")

	_fill_blender_mesh(mesh, arrays)

	mesh.update() 
	mesh.validate()

	if not just_the_mesh:
		for data in msh.hardpoints:
			support.create_hardpoint_obj(data[12], data[0:12], parent = obj)

	return obj
	 
def load_new(context,
			 filepath,
			 *,	 
			 global_matrix=None,
			 flip_uv_vertical=False,
			 remove_duplicate_verts=True,
			 ):  

	obj = import_msh(context, filepath, None, flip_uv_vertical, remove_duplicate_verts)
	return {'FINISHED'}