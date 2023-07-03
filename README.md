# Bachelorarbeit
Automatic dataset generation for neural depth estimation in active setups

# Requirements
- [BlenderProc](https://github.com/DLR-RM/BlenderProc)
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
The default value for the maximum depth is set to 20. Since a normalization in the range [0,1] *(scientific standard)* was performed, this value must now be adjusted. </br></br>
```blenderproc vis --depth_max=1 hdf5 <path/to/file.hdf5>```

# Todo:
## minor stuff
\[X\] Change current depth-scale from 0-20 to 0-1 </br>
\[ \] Different light-sources </br>

## add projector with patterns
[X] point- and stripe-pattern
## move camera on spline

## optional
\[ \] physic simulation for spawning objects </br>
