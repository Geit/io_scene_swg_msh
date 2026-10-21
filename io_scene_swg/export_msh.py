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

import os
import bpy
import base64
import bmesh
import time, datetime, functools, math
import numpy as np
from . import vector3D
from . import swg_types
from . import vertex_buffer_format
from . import extents
from . import support

from mathutils import Matrix, Vector, Color
from bpy_extras import io_utils, node_shader_utils

from bpy_extras.wm_utils.progress_report import (
	ProgressReport,
	ProgressReportSubstep,
)

def mesh_triangulate(me):
	bm = bmesh.new()
	bm.from_mesh(me)
	bmesh.ops.triangulate(bm, faces=bm.faces)
	bm.to_mesh(me)
	bm.free()

def save(context, filepath, *, flip_uv_vertical=False):
	objects = context.selected_objects

	if len(objects) == 0:
		print(f"Nothing selected. Aborting!")
		return {'CANCELLED'}
	else:
		print(f"Objects to export: {len(objects)}")

	for ob in objects:
		print(f"Exporting: {ob.name}")
		if ob.type != 'MESH':
			print(f"Skipping {ob.name} with type: {ob.type}")
			continue
		else:
			dirname = os.path.dirname(filepath)
			fullpath = os.path.join(dirname, ob.name+".msh")
			extract_dir=context.preferences.addons[__package__].preferences.swg_root
			result = export_one(fullpath, extract_dir, ob, flip_uv_vertical)
			if not 'FINISHED' in result:
				return {'CANCELLED'}
	return {'FINISHED'}

def _foreach_array(collection, attr, count, width, dtype=np.float64):
	"""Read a per-item property of a bpy collection into a numpy array of shape (count, width) or (count,)."""
	arr = np.empty(count * width, dtype=dtype)
	if count:
		collection.foreach_get(attr, arr)
	return arr.reshape(count, width) if width > 1 else arr

def _material_wants_dot3(material):
	return material is not None and "DOT3" in material and bool(material["DOT3"])

def _is_all_triangles(me):
	"""True when every polygon has three corners."""
	loop_totals = _foreach_array(me.polygons, "loop_total", len(me.polygons), 1, np.int32)
	return not (loop_totals != 3).any()

def _material_indices_in_first_use_order(poly_mat):
	"""Distinct material indices, in order of first use."""
	if len(poly_mat) == 0:
		return []
	_, first_face = np.unique(poly_mat, return_index=True)
	return poly_mat[np.sort(first_face)].tolist()

def _corner_loops(poly_start, face_ids):
	"""Loop index of each corner of the given triangles, three per face."""
	first = poly_start[face_ids]
	return np.column_stack((first, first + 1, first + 2)).reshape(-1)

def _dedup_corners(loops, loop_vert, loop_normals, uv0):
	"""Merge corners with the same vertex, normal and uv0 (rounded to 4 places).
	Returns (vert_loop, corner_vert): the defining loop of each unique vertex, and each corner's unique vertex as (faces, 3)."""
	key = np.empty((len(loops), 6), dtype=np.float64)
	key[:, 0] = loop_vert[loops]
	key[:, 1:4] = np.round(loop_normals[loops], 4)
	key[:, 4:6] = np.round(uv0[loops], 4)
	first, inverse = support.unique_rows(key)
	return loops[first], inverse.reshape(-1, 3)

def export_one(fullpath, extract_dir, obj, flip_uv_vertical):
	newMsh = swg_types.SWGMesh(fullpath, extract_dir)
	start = time.time()
	print(f'Exporting msh: {fullpath} Flip UV: {flip_uv_vertical}')

	me = obj.to_mesh()
	if not _is_all_triangles(me):
		mesh_triangulate(me)
	me.calc_normals_split()

	for layer in me.vertex_colors:
		print(f"Color layer: {layer.name}")

	num_loops = len(me.loops)
	num_polys = len(me.polygons)
	num_me_verts = len(me.vertices)

	# loop normals, read before calc_tangents and flip_normals
	loop_normals = _foreach_array(me.loops, "normal", num_loops, 3)

	if any(_material_wants_dot3(slot.material) for slot in obj.material_slots):
		for name in [uvlayer.name for uvlayer in me.uv_layers]:
			me.calc_tangents(uvmap=name)

	#If negative scaling, we have to invert the normals...
	if obj.matrix_world.determinant() < 0.0:
		me.flip_normals()

	coords = _foreach_array(me.vertices, "co", num_me_verts, 3)
	loop_vert = _foreach_array(me.loops, "vertex_index", num_loops, 1, np.int32)
	poly_mat = _foreach_array(me.polygons, "material_index", num_polys, 1, np.int32)
	poly_start = _foreach_array(me.polygons, "loop_start", num_polys, 1, np.int32)
	uv_arrays = [_foreach_array(layer.data, "uv", num_loops, 2) for layer in me.uv_layers]
	color_arrays = {layer.name: _foreach_array(layer.data, "color", num_loops, 4) for layer in me.vertex_colors}
	tangents = None
	bitangent_signs = None

	if len(uv_arrays) == 0:
		print(f"Error! {obj.name} has no UV map. A UV map is required to export a mesh.")
		return {'CANCELLED'}

	mat_order = _material_indices_in_first_use_order(poly_mat)
	for index in mat_order:
		print(f"Faces_by_material[{index}]: {int((poly_mat == index).sum())}")

	this_mat_index=0
	total_tris=0
	total_verts=0
	for mat_index in mat_order:
		face_ids = np.nonzero(poly_mat == mat_index)[0]
		try:
			material = obj.material_slots[mat_index].material
		except :
			print(f"Asked for material index: {mat_index} but we only have {len(obj.material_slots)}. Won't do anything with {len(face_ids)} triangles I guess")
			continue

		sps_no = this_mat_index
		this_mat_index += 1

		uvSets = 1
		if "UVSets" in material:
			uvSets = material["UVSets"]

		doDOT3 = _material_wants_dot3(material)
		flags = vertex_buffer_format.setPosition(0, True)
		flags = vertex_buffer_format.setNormal(flags, True)
		flags = vertex_buffer_format.setNumberOfTextureCoordinateSets(flags, uvSets)

		doColor0 = "Color0" in material and (material["Color0"] == 1)
		flags = vertex_buffer_format.setColor0(flags, doColor0)
		doColor1 = "Color1" in material and (material["Color1"] == 1)
		flags = vertex_buffer_format.setColor1(flags, doColor1)

		for i in range(0, uvSets):
			flags = vertex_buffer_format.setTextureCoordinateSetDimension(flags, i, 2)

		if doDOT3:
			uv_dim = vertex_buffer_format.getNumberOfTextureCoordinateSets(flags) + 1
			flags = vertex_buffer_format.setNumberOfTextureCoordinateSets(flags, uv_dim)
			flags = vertex_buffer_format.setTextureCoordinateSetDimension(flags, uv_dim - 1, 4)

		loops = _corner_loops(poly_start, face_ids)
		vert_loop, corner_vert = _dedup_corners(loops, loop_vert, loop_normals, uv_arrays[0])

		# one record per unique vertex, in file layout
		num_unique = len(vert_loop)
		vertex_data = np.zeros(num_unique, dtype=vertex_buffer_format.vertex_dtype(flags))
		vertex_data['pos'] = support.convert_vector3_array(coords[loop_vert[vert_loop]])
		vertex_data['normal'] = support.convert_vector3_array(loop_normals[vert_loop])
		if doColor0:
			vertex_data['color0'] = vertex_buffer_format.rgba_to_color_bytes(color_arrays["color0"][vert_loop])
		if doColor1:
			vertex_data['color1'] = vertex_buffer_format.rgba_to_color_bytes(color_arrays["color1"][vert_loop])
		for i in range(0, uvSets):
			if i >= len(uv_arrays):
				print(f"Warning! {obj.name} material {material.name} asks for {uvSets} UV sets but the mesh only has {len(uv_arrays)}. Missing sets are written as zeros.")
				break
			uv = uv_arrays[i][vert_loop].copy()
			if flip_uv_vertical:
				uv[:, 1] = 1.0 - uv[:, 1]
			vertex_data[f'uv{i}'] = uv
		if doDOT3:
			if tangents is None:
				tangents = _foreach_array(me.loops, "tangent", num_loops, 3)
				bitangent_signs = _foreach_array(me.loops, "bitangent_sign", num_loops, 1)
			# tangent (x, y, z, bitangent sign) in the extra 4-wide UV set
			tangent_xyz = support.convert_vector3_array(tangents[vert_loop])
			vertex_data[f'uv{uv_dim - 1}'] = np.column_stack((tangent_xyz, bitangent_signs[vert_loop]))

		# corners in reverse order (p3, p2, p1)
		indices = np.ascontiguousarray(corner_vert[:, ::-1], dtype=np.int32)
		thisSPS = swg_types.SPS(sps_no, f'shader/{material.name}.sht', flags, vertex_data, indices)
		total_verts += num_unique
		total_tris += len(face_ids)

		print(f"SPS {thisSPS.no}: Unique Verts: {num_unique} UV Channels: {vertex_buffer_format.getNumberOfTextureCoordinateSets(flags)} Has flags {flags}")
		newMsh.spss.append(thisSPS)
		this_mat_index += 1

	newMsh.extents = _box_extents(coords)

	for ob in bpy.data.objects:
		if ob.parent == obj:
			if ob.type != 'MESH' and ob.type == 'EMPTY' and ob.empty_display_type == "ARROWS":
				newMsh.hardpoints.append(support.hardpoint_from_obj(ob))

	newMsh.write(fullpath)
	now = time.time()
	print(f"total_tris: {total_tris} total_verts: {total_verts} Successfully wrote: {fullpath} Duration: " + str(datetime.timedelta(seconds=(now-start))))

	return {'FINISHED'}

def _box_extents(coords):
	"""Axis-aligned extents, in SWG axis order, of an (N, 3) array of Blender vertex positions."""
	if len(coords) == 0:
		return extents.BoxExtents([None, None, None], [None, None, None])
	swg = support.convert_vector3_array(coords)
	return extents.BoxExtents(swg.min(axis=0).tolist(), swg.max(axis=0).tolist())

def get_extents(obj):
	me = obj.to_mesh()
	return _box_extents(_foreach_array(me.vertices, "co", len(me.vertices), 3))

def avg_vert_position_in_blender(obj):
	me = obj.to_mesh()
	n = len(me.vertices)
	if n == 0:
		return None
	co = _foreach_array(me.vertices, "co", n, 3, np.float32)
	# sequential float32 sum (cumsum, not sum) to round exactly like the old Vector accumulation
	total = np.cumsum(co, axis=0, dtype=np.float32)[-1]
	return Vector((total / np.float32(n)).tolist())
