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

RAW_FOLDER = "pipeline_points"
IMAGES = 250
POSES = 5
EPOCHES = 10


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pipeline from random-room-generation to neuronal monocular depth-estimation',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--loop', action=argparse.BooleanOptionalAction, help='enable/disable image-generation', default=False)
    parser.add_argument('--output_dir', '-o', help='name of outputfolder', default=RAW_FOLDER)    
    parser.add_argument('--num_poses', '-poses', help='Number of poses within one frame', default=POSES)
    parser.add_argument('--num_images', '-images', help='Number of created frames', default=IMAGES)

    args = parser.parse_args()
    
    if (args.loop):
        loop(args.output_dir, args.num_poses, args.num_images) 

    #extract_images(args.output_dir + '/NORMALOS/', 'neuronalDepthEst/' + args.output_dir + '/NORMALOS')
    #extract_images(args.output_dir + '/PATTERN/256/', 'neuronalDepthEst/' + args.output_dir + '/PATTERN/256')
    #extract_images(args.output_dir + '/PATTERN/1000/', 'neuronalDepthEst/' + args.output_dir + '/PATTERN/1000')
    extract_images(args.output_dir + '/PATTERN/2560/', 'neuronalDepthEst/' + args.output_dir + '/PATTERN/2560')
    #extract_images(args.output_dir + '/PATTERN/25600/', 'neuronalDepthEst/' + args.output_dir + '/PATTERN/25600')
    #extract_images(args.output_dir + '/PATTERN/40000/', 'neuronalDepthEst/' + args.output_dir + '/PATTERN/40000')
    #extract_images(args.output_dir + '/INFRARED/256/', 'neuronalDepthEst/' + args.output_dir + '/INFRARED/256')
    #extract_images(args.output_dir + '/INFRARED/1000/', 'neuronalDepthEst/' + args.output_dir + '/INFRARED/1000')
    extract_images(args.output_dir + '/INFRARED/2560/', 'neuronalDepthEst/' + args.output_dir + '/INFRARED/2560')
    #extract_images(args.output_dir + '/INFRARED/25600/', 'neuronalDepthEst/' + args.output_dir + '/INFRARED/25600')
    #extract_images(args.output_dir + '/INFRARED/40000/', 'neuronalDepthEst/' + args.output_dir + '/INFRARED/40000')

    #depthestimation("neuronalDepthEst/" + args.output_dir + "/PATTERN/256/", True, EPOCHES)
    #depthestimation("neuronalDepthEst/" + args.output_dir + "/PATTERN/1000/", True, EPOCHES)
    depthestimation("neuronalDepthEst/" + args.output_dir + "/PATTERN/2560/", True, EPOCHES)
    #depthestimation("neuronalDepthEst/" + args.output_dir + "/PATTERN/25600/", , EPOCHES)
    #depthestimation("neuronalDepthEst/" + args.output_dir + "/PATTERN/40000/", False, EPOCHES)
    #depthestimation("neuronalDepthEst/" + args.output_dir + "/INFRARED/256/", True, EPOCHES)
    #depthestimation("neuronalDepthEst/" + args.output_dir + "/INFRARED/1000/", True, EPOCHES)
    depthestimation("neuronalDepthEst/" + args.output_dir + "/INFRARED/2560/", True, EPOCHES)
    #depthestimation("neuronalDepthEst/" + args.output_dir + "/INFRARED/25600/", True, EPOCHES)
    #depthestimation("neuronalDepthEst/" + args.output_dir + "/INFRARED/40000/", True, EPOCHES)
