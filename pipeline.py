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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pipeline from random-room-generation to neuronal monocular depth-estimation',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--loop', action=argparse.BooleanOptionalAction, help='enable/disable image-generation', default=False)
    parser.add_argument('--output_dir', '-o', help='name of outputfolder', default=RAW_FOLDER)
    args = parser.parse_args()

    os.makedirs(RAW_FOLDER, exist_ok=True)
    os.makedirs("neuronalDepthEst" + RAW_FOLDER, exist_ok=True)
    
    if (args.loop):
        loop(RAW_FOLDER)
    
    extract_images(args.output_dir + '/NORMALOS/', 'neuronalDepthEst/' + args.output_dir + '/NORMALOS')
    extract_images(args.output_dir + '/PATTERN/', 'neuronalDepthEst/' + args.output_dir + '/PATTERN')
    extract_images(args.output_dir + '/INFRARED', 'neuronalDepthEst/' + args.output_dir + '/INFRARED')

    depthestimation("neuronalDepthEst/" + args.output_dir + "/NORMALOS", True)
    depthestimation("neuronalDepthEst/" + args.output_dir + "/PATTERN", True)
    depthestimation("neuronalDepthEst/" + args.output_dir + "/INFRARED", True)

