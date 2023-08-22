#!/bin/python3

import subprocess
from tqdm import tqdm

def loop(output_dir, poses, runs):
	# normalos
	for i in tqdm(range(runs)):
		command = f"blenderproc run randomroom.py -o {output_dir} --full --num_poses {poses}"
		process = subprocess.run(command.split(), stdout=subprocess.PIPE)

	print("Done generating images")
