# Bachelorarbeit
Automatic dataset generation for neural depth estimation in active setups

# Requirements
- python version 3.9.18
- [BlenderProc](https://github.com/DLR-RM/BlenderProc)
- torch version matching cuda version eg. for ACID: ```conda install pytorch=1.11.0 torchvision=0.12 torchaudio=0.11 cudatoolkit=11.5 -c pytorch -c conda-forge```
- ```python3 -m pip install -r req.txt```
- Download [Resources](https://nextcloud.beekama.de/index.php/s/RZq2xxSGmWeKQHF) and unpack folder to ./blenderproc/
# Usage
## Generate random room
### No pattern
```blenderproc run randomroom.py```
### With pattern
Infrared mode disables all light-sources except of the projector. </br>
```blenderproc run randomroom.py -proj -pat <points/stripes> -npat <NUM_OF_STRIPES/POINTS> <--infrared>```
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
