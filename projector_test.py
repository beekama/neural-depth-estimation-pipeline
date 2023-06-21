### based on the quickstart example ###
import blenderproc as bproc
import numpy as np

bproc.init()

# Create a simple object:
obj = bproc.object.create_primitive("CUBE")
obj.set_location([0,-0,0])
#print(obj.get_location())
#print(obj.get_rotation_euler())

# Create a point light next to it
#light = bproc.types.Light()
#light.set_location([2, -2, 0])
#light.set_energy(300)
#print(light.get_rotation())

# Set the camera to be in front of the object
cam_pose = bproc.math.build_transformation_mat([0, -5, 0], [np.pi / 2, 0, 0])
bproc.camera.add_camera_pose(cam_pose)

# Create projector next to camera
#proj = bproc.loader.load_obj("resources/projector.obj")
proj = bproc.loader.load_obj("resources/strip.obj")
proj[0].set_location([1, -2, 0])
proj[0].set_rotation_euler([0,0,0])
#proj[0].set_scale([2,2,4])
proj[0].set_rotation_mat([[np.cos(45), -np.sin(45), 0],[np.sin(45),np.cos(45), 0],[0, 0, 1]])
bproc.lighting.light_surface(proj, 10)

# Render the scene
data = bproc.renderer.render()

# Write the rendering into an hdf5 file
bproc.writer.write_hdf5("output/", data)