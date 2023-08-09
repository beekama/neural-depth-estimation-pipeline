#!/bin/python3

import subprocess
from tqdm import tqdm

def loop():
	# normalos
	for i in tqdm(range(250)):
		command = f"blenderproc run randomroom.py -o test --full"
		process = subprocess.run(command.split(), stdout=subprocess.PIPE)

	# # patterns
	# for i in tqdm(range(10)):
	# 	command = f"blenderproc run randomroom.py -o test/PATTERN -proj -pat points"
	# 	process = subprocess.run(command.split(), stdout=subprocess.PIPE)

	# # patterns infrared
	# for i in tqdm(range(10)):
	# 	command = f"blenderproc run randomroom.py -o test/INFRARED -proj -pat points --infrared"
	# 	process = subprocess.run(command.split(), stdout=subprocess.PIPE)

	print("Done generating images")