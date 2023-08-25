#!/bin/python3

import subprocess
from tqdm import tqdm

def loop(output_dir, poses, runs):
	# normalos
	for i in tqdm(range(runs)):
		command = f"blenderproc run randomroom.py -o {output_dir} --full --num_poses {poses} --num_pattern 256"
		process = subprocess.run(command.split(), stdout=subprocess.PIPE)
	for i in tqdm(range(runs)):
		command = f"blenderproc run randomroom.py -o {output_dir} --full --num_poses {poses} --num_pattern 1000"
		process = subprocess.run(command.split(), stdout=subprocess.PIPE)
	for i in tqdm(range(runs)):
		command = f"blenderproc run randomroom.py -o {output_dir} --full --num_poses {poses} --num_pattern 2560"
		process = subprocess.run(command.split(), stdout=subprocess.PIPE)
	for i in tqdm(range(runs)):
		command = f"blenderproc run randomroom.py -o {output_dir} --full --num_poses {poses} --num_pattern 25600"
		process = subprocess.run(command.split(), stdout=subprocess.PIPE)
	for i in tqdm(range(runs)):
		command = f"blenderproc run randomroom.py -o {output_dir} --full --num_poses {poses} --num_pattern 40000"
		process = subprocess.run(command.split(), stdout=subprocess.PIPE)
	print("Done generating images")
