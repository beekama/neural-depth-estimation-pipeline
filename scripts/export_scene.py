import bpy
import numpy as np

def export_objects(objects, filename):
	for obj in objects:
			obj.select_set(True)
	bpy.ops.export_scene.obj(
		filepath=filename,
		use_selection=True,
		use_mesh_modifiers=True,
		use_materials=True
	)
	bpy.ops.object.select_all(action="DESELECT")

def export_camera(cam2world, filename):
	np.savetxt(filename, cam2world)