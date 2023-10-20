#!/bin/python3

import subprocess
from tqdm import tqdm

def loop(output_dir, poses, runs, load_scene, load_camera, save_scene, save_camera, save_camera_intrinsics):
	

	for i in tqdm(range(runs)):
		command = f"blenderproc run randomroom.py -o {output_dir} --full --num_poses {poses} "
		
		# Check if optional arguments are provided and add them to the command
		if load_scene:
			command += f'--load_scene {load_scene} '
		if load_camera:
			command += f'--load_camera {load_camera} '
		if save_scene:
			command += f'--save_scene {save_scene} '
		if save_camera:
			command += f'--save_camera {save_camera} '
		if save_camera_intrinsics:
			command += f'--save_camera_intrinsics {save_camera_intrinsics} '
		
		subprocess.run(command.split(), stdout=subprocess.PIPE)

	print("Done generating images")