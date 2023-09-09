#!/bin/python3

import argparse
import loop

from neuronalDepthEst import UNet
from neuronalDepthEst import UNetResNet
from neuronalDepthEst import UNetRplus

# import helper-scripts
import os
import sys

sys.path.append('scripts')
sys.path.append('neuronalDepthEst')
from extract_images import extract_images
from depthestimation import depthestimation
from loop import loop

RAW_FOLDER = "pipeline_test"

IMAGES = 5
LOAD_SCENE = False
SAVE_SCENE = False
SCENE_FILE = ""
CAMERA_FILE = ""
POSES = 5

EPOCHES = 10
TRAINING = True
MODEL = "Unet" # "Unetresnet" "Unetplus"



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pipeline from random-room-generation to neuronal monocular depth-estimation',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    model_choices = {
    'Unet': UNet,
    'Unetresnet': UNetResNet,
    'Unetplus': UNetRplus,
    }
    
    parser.add_argument('--loop', action=argparse.BooleanOptionalAction, help='enable/disable image-generation', default=False)
    parser.add_argument('--output_dir', '-o', help='name of outputfolder', default=RAW_FOLDER)    
    parser.add_argument('--num_poses', '-poses', help='Number of poses within one frame', default=POSES)
    parser.add_argument('--num_images', '-images', help='Number of created frames', default=IMAGES)
    parser.add_argument('--training', action=argparse.BooleanOptionalAction, help='(un)set trainings loop for depthestimation', default=TRAINING)
    parser.add_argument('--load_scene', help='File to load scene from', default="")
    parser.add_argument('--load_camera', help='File to load camerapose from', default="")
    parser.add_argument('--save_scene', help='File to save scene to', default="")
    parser.add_argument('--save_camera', help='File to save camerapose to', default="")
    parser.add_argument('--save_camera_intrinsics', help='File to save camera-intrinsics from/to', default="")
    parser.add_argument('--model', choices=model_choices.keys(), help="select model type", default=UNet)

    args = parser.parse_args()
    
    if (args.loop):
        loop(args.output_dir, args.num_poses, args.num_images, args.load_scene, args.load_camera, args.save_scene, args.save_camera, args.save_camera_intrinsics)
    
    extract_images(args.output_dir + '/NORMALOS/', 'neuronalDepthEst/' + args.output_dir + '/NORMALOS')
    extract_images(args.output_dir + '/PATTERN/', 'neuronalDepthEst/' + args.output_dir + '/PATTERN')
    extract_images(args.output_dir + '/INFRARED', 'neuronalDepthEst/' + args.output_dir + '/INFRARED')

    depthestimation("neuronalDepthEst/" + args.output_dir + "/NORMALOS", args.training, EPOCHES, model_choices[args.model])
    depthestimation("neuronalDepthEst/" + args.output_dir + "/PATTERN", args.training, EPOCHES, model_choices[args.model])
    depthestimation("neuronalDepthEst/" + args.output_dir + "/INFRARED", args.training, EPOCHES, model_choices[args.model])

