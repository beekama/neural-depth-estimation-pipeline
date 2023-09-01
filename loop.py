#!/bin/python3

import subprocess
from tqdm import tqdm

def loop(output_dir, poses, runs, load_scene, load_camera, save_scene, save_camera, save_camera_intrinsics):
	# normalos
	for i in tqdm(range(runs)):
		command = f"blenderproc run randomroom.py -o {output_dir} --full --num_poses {poses} --load_scene {load_scene}, --load_camera {load_camera} --save_scene {save_scene} --save_camera {save_camera} --save_camera_intrinsics {save_camera_intrinsics}"
		process = subprocess.run(command.split(), stdout=subprocess.PIPE)

	print("Done generating images")