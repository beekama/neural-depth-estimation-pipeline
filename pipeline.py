#!/bin/python3

import argparse
import loop

# import helper-scripts
import os
import sys

sys.path.append('scripts')
sys.path.append('neuronalDepthEst')
from extract_images import extract_images
from depthestimation import depthestimation
from loop import loop

RAW_FOLDER = "pipeline_1"
POSES = 5
EPOCHES = 10


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pipeline from random-room-generation to neuronal monocular depth-estimation',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--loop', action=argparse.BooleanOptionalAction, help='enable/disable image-generation', default=False)
    parser.add_argument('--output_dir', '-o', help='name of outputfolder', default=RAW_FOLDER)    
    parser.add_argument('--num_poses', '-poses', help='Number of poses within one frame', default=POSES)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("neuronalDepthEst" + args.output_dir, exist_ok=True)
    
    if (args.loop):
        loop(args.output_dir, args.num_poses)
    
    extract_images(args.output_dir + '/NORMALOS/', 'neuronalDepthEst/' + args.output_dir + '/NORMALOS')
    extract_images(args.output_dir + '/PATTERN/', 'neuronalDepthEst/' + args.output_dir + '/PATTERN')
    extract_images(args.output_dir + '/INFRARED', 'neuronalDepthEst/' + args.output_dir + '/INFRARED')

    depthestimation("neuronalDepthEst/" + args.output_dir + "/NORMALOS", True, EPOCHES)
    depthestimation("neuronalDepthEst/" + args.output_dir + "/PATTERN", True, EPOCHES)
    depthestimation("neuronalDepthEst/" + args.output_dir + "/INFRARED", True, EPOCHES)

