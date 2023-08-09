#!/bin/python3

import subprocess
from tqdm import tqdm

def loop(output_dir):
	# normalos
	for i in tqdm(range(250)):
		command = f"blenderproc run randomroom.py -o {output_dir} --full"
		process = subprocess.run(command.split(), stdout=subprocess.PIPE)

	print("Done generating images")