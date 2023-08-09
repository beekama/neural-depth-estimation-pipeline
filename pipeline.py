#!/bin/python3

import argparse
import loop

# import helper-scripts
import sys
sys.path.append('scripts')
sys.path.append('neuronalDepthEst')
from extract_images import extract_images
from depthestimation import depthestimation
from loop import loop


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pipeline from random-room-generation to neuronal monocular depth-estimation',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--loop', action=argparse.BooleanOptionalAction, help='enable/disable image-generation', default=False)

    #parser.add_argument('--folder_in', '-f', help='location of hdf5-folder', nargs='+')
    #parser.add_argument('--output_dir', '-o', help='location of output-folder for images and depth', required=True)
    args = parser.parse_args()
    
    if (args.loop):
        loop()
    
    #extract_images('test/NORMALOS/', 'neuronalDepthEst/test/NORMALOS')
    #extract_images('test/PATTERN/', 'neuronalDepthEst/test/PATTERN')
    #extract_images('test/INFRARED', 'neuronalDepthEst/test/INFRARED')

    depthestimation("neuronalDepthEst/test/NORMALOS", True)
    depthestimation("neuronalDepthEst/test/PATTERN", True)
    depthestimation("neuronalDepthEst/test/INFRARED", True)

