#!/bin/python3

import subprocess
from tqdm import tqdm
import os.path, glob

SCENE_DEFAULT = "saved_scene*"
CAM_DEFAULT = "saved_camera*"	

def loop(output_dir, poses, runs, load_scene, load_camera, save_scene, save_camera, save_camera_intrinsics, patterntype):
	
	load_folder = os.path.isdir(load_scene) and os.path.isdir(load_camera)
	if os.path.isdir(load_scene) != os.path.isdir(load_camera):
		raise ValueError("'save_scene-file' and 'save_camera-file' must be both file or both folder")
	if load_folder:
		files_scene = glob.glob(os.path.join(load_scene, SCENE_DEFAULT)) 
		files_scene.sort()
		files_camera = glob.glob(os.path.join(load_camera, CAM_DEFAULT)) 
		files_camera.sort()

	for i in tqdm(range(runs)):
		command = f"blenderproc run randomroom.py -o {output_dir} --full --num_poses {poses} "
		
		# Check if optional arguments are provided and add them to the command
		if load_folder:
			command += f'--load_scene {files_scene[i]} '
			command += f'--load_camera {files_camera[i]} '
		else:	
			command += f'--load_scene {load_scene} ' if load_scene else ''
			command += f'--load_camera {load_camera} ' if load_camera else ''
		if save_scene:
			command += f'--save_scene {save_scene} '
		if save_camera:
			command += f'--save_camera {save_camera} '
		if save_camera_intrinsics:
			command += f'--save_camera_intrinsics {save_camera_intrinsics} '
		if patterntype:
			command += f'--pattern_type {patterntype} '
		
		subprocess.run(command.split(), stdout=subprocess.PIPE)

	print("Done generating images")