### based on the quickstart example ###
import blenderproc as bproc
import numpy as np
from PIL import Image


bproc.init()

# Create a simple object:
obj = bproc.object.create_primitive("MONKEY")
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
bproc.camera.set_stereo_parameters(interocular_distance=0.05, convergence_mode="PARALLEL")

# Create projector next to camera
proj = bproc.types.Light()
proj.set_type('SPOT')
proj.set_energy(3000)
fov = bproc.camera.get_fov()
pattern_img = np.asarray(Image.open("resources/stips.png"))

#roj.set_location([1, -2, 0])
#proj.set_rotation_euler([0,0,0])
#roj.set_rotation_mat([[np.cos(45), -np.sin(45), 0],[np.sin(45),np.cos(45), 0],[0, 0, 1]])

proj.setup_as_projector(pattern_img)
# Render the scene
data = bproc.renderer.render()

# Write the rendering into an hdf5 file
bproc.writer.write_hdf5("output/", data)