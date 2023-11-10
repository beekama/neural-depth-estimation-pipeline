# Bachelorarbeit
Automatic dataset generation for neural depth estimation in active setups

# Requirements
- python version 3.9.18
- [BlenderProc](https://github.com/DLR-RM/BlenderProc)
- torch version matching cuda version eg. for ACID: ```conda install pytorch=1.11.0 torchvision=0.12 torchaudio=0.11 cudatoolkit=11.5 -c pytorch -c conda-forge```
- ```python3 -m pip install -r req.txt```
- Download [Resources](https://nextcloud.beekama.de/index.php/s/RZq2xxSGmWeKQHF) and unpack folder to ./blenderproc/
# Usage
## Dataset generation and evaluation pipeline
    usage: pipeline.py [-h] [--loop | --no-loop] [--output_dir OUTPUT_DIR] [--num_poses NUM_POSES] [--num_images NUM_IMAGES] [--training | --no-training] [--load_scene LOAD_SCENE] [--load_camera LOAD_CAMERA]
                       [--save_scene SAVE_SCENE] [--save_camera SAVE_CAMERA] [--save_camera_intrinsics SAVE_CAMERA_INTRINSICS] [--model {Unet,Unetresnet,all}] [--combined COMBINED]
                       [--pattern {rainbow,continuous,points,stripes}]
    ----
    pipeline from random-room-generation to neuronal monocular depth-estimation
    ----
    optional arguments:                                                                                                                                                                                             
      -h, --help            show this help message and exit
      --loop, --no-loop     enable/disable image-generation (default: False)
      --output_dir OUTPUT_DIR, -o OUTPUT_DIR
                            name of outputfolder (default: meow_stacked)
      --num_poses NUM_POSES, -poses NUM_POSES
                            Number of poses within one frame (default: 5)
      --num_images NUM_IMAGES, -images NUM_IMAGES
                            Number of created frames (default: 250)
      --training, --no-training
                            (un)set trainings loop for depthestimation (default: True)
      --load_scene LOAD_SCENE
                            File to load scene from (default: )
      --load_camera LOAD_CAMERA
                            File to load camerapose from (default: )
      --save_scene SAVE_SCENE
                            File/folder to save scene to (default: )
      --save_camera SAVE_CAMERA
                            File/folder to save camerapose to (default: )
      --save_camera_intrinsics SAVE_CAMERA_INTRINSICS
                            File to save camera-intrinsics from/to (default: )
      --model {Unet,Unetresnet,all}
                            select model type (default: all)
      --combined COMBINED   Stack Normals and Infrareds (default: True)
      --pattern {rainbow,continuous,points,stripes}
                            select pattern type (default: points)
## Visualize data
The default value for the maximum depth is set to 20. Since a normalization in the range [0,1] *(scientific standard)* was performed, this value must now be adjusted. </br>
```blenderproc vis --depth_max=1 hdf5 <path/to/file.hdf5>```
## Extract data
Extract hdf5 files and split into images.png and depth.png</br>
```python scripts/extract_images.py -i <.../NORMALOS/*> -o <.../test&train>```
## Split into training-set and test-set
TODO - By now this step has to be done manually. </br>
## Neuronal Depth Estimator
### Train data
Only train and save the model: </br>
```python neuronalDepthEst/depthestimation.py -f <folder> --train``` </br>
Train and test the model and plot/save results: </br>
```python neuronalDepthEst/depthestimation.py -f <folder> --full```
### Test data
Load saved model and test/plot: </br>
```python neuronalDepthEst/depthestimation.py -f <folder> --test```
# Todo:
## minor stuff
\[X\] Change current depth-scale from 0-20 to 0-1 </br>
\[ \] Different light-sources </br>

## add projector with patterns
[X] point- and stripe-pattern
## move camera on spline

## optional
\[ \] physic simulation for spawning objects </br>
