#!/bin/python3

import subprocess
from tqdm import tqdm

# normalos
for i in tqdm(range(10)):
	command = f"blenderproc run randomroom.py -o output_normalos_{i}"
	process = subprocess.run(command.split(), stdout=subprocess.PIPE)

# patterns
for i in tqdm(range(10)):
	command = f"blenderproc run randomroom.py -o output_patterns_{i} -proj -pat points"
	process = subprocess.run(command.split(), stdout=subprocess.PIPE)

# patterns infrared
for i in tqdm(range(10)):
	command = f"blenderproc run randomroom.py -o output_infrared_{i} -proj -pat points --infrared"
	process = subprocess.run(command.split(), stdout=subprocess.PIPE)

print("DONEEE")
