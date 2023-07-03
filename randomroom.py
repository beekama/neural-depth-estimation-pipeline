#!/usr/bin/python3

### IMPORTS ###
import blenderproc as bproc
import argparse
import numpy as np
from PIL import Image
from random import randrange

# import helper-scripts
import sys
sys.path.append('scripts')
import readhdf5, create_striped_img

### DEFAULTS ###
MATERIALS_PATH = 'resources/materials'
MESH_PATH = 'resources/test/'

NUM_OF_OBJECTS = 3
NUM_OF_PERSPECTIVES = 5
NUM_OF_LIGHTSOURCES = 1



def testDataGenerator(args):
    
    # enable cycles renderer and sets some speedup options for rendering
    bproc.init()

    #RESOURCES = ['indoor plant_02.blend', "12221_Cat_v1_l3.obj"]
    RESOURCES = ['plant_bunt.blend', 'sofa_bunt.blend']
    
    
    # load materias and objects
    materials = bproc.loader.load_ccmaterials(args.material_path, ["Bricks", "Wood", "Carpet", "Tile", "Marble"])
    interior_objects = []
    for i in range (args.num_meshes):
        interior_objects.extend(bproc.loader.load_blend(args.mesh_path +  RESOURCES[randrange(len(RESOURCES))]))
    
    # construct random room
    objects = bproc.constructor.construct_random_room(used_floor_area=25, interior_objects=interior_objects,materials=materials, amount_of_extrusions=5)
    
    # light sources
    if not args.infrared:
        bproc.lighting.light_surface([obj for obj in objects if obj.get_name() == "Ceiling"], emission_strength=4.0, emission_color=[1,1,1,1])


    # init bvh tree containing all mesh objects
    bvh_tree = bproc.object.create_bvh_tree_multi_objects(objects)
    floor = [obj for obj in objects if obj.get_name() == "Floor"][0]
    poses = 0
    # Get point-of-interest, the camera should look towards and determine starting point
    poi = bproc.object.compute_poi(objects)
    start_location = bproc.sampler.upper_region(floor, min_height=1.5, max_height=1.8)
    spline_vector = poi-start_location
    step_size = spline_vector/5
    current_loc = start_location

    while poses < 5:
        # Sample random camera location above objects
        rand = np.random.uniform([-0.1, -0.3, 0], [0.1, 0.3, 0.05])
        location = current_loc + rand
        current_loc += step_size

        # Compute rotation based on vector going from location towards poi
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi-location, inplane_rot=np.random.uniform(-0.7854, 0.7854))
        # Add homog cam pose based on location an rotation
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        
        # check that obstacles are at least 1 meter away from camera and make sure the view is interesting enough
        #if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min":1.2}, bvh_tree) and \
        #        bproc.camera.scene_coverage_score(cam2world_matrix) > 0.4 and \
        #        floor.position_is_above_object(location):
        #    # persist camera pose
        bproc.camera.add_camera_pose(cam2world_matrix)
        poses += 1
            
    # define new light source as projector
    if args.projection:
        if args.proj_pattern == "stripes":
            pattern_img = create_striped_img.create_img(args.num_pattern)
        else:
            pattern_img = bproc.utility.generate_random_pattern_img(1280, 720, args.num_pattern)
        proj = bproc.types.Light()
        proj.set_type('SPOT')
        proj.set_energy(3000)
        proj.setup_as_projector(pattern_img)
    
    # activate depth rendering
    bproc.renderer.enable_depth_output(activate_antialiasing=False, output_dir='output', file_prefix='depth_', convert_to_distance=False)
    #bproc.renderer.set_light_bounces(max_bounces=200, diffuse_bounces=200, glossy_bounces=200, transmission_bounces=200, transparent_max_bounces=200)
    #m bproc.material.add_alpha_channel_to_textures(blurry_edges=True)
    #bproc.renderer.toggle_stereo(True)
    
    # render whole pipeline
    data = bproc.renderer.render()
    # Apply stereo matching to each pair of images
    data["stereo-depth"], data["disparity"] = bproc.postprocessing.stereo_global_matching(data["colors"], disparity_filter=False)

    # normalize depth
    depth = data['depth']
    data['depth'] = readhdf5.normalize(depth, 0, 1)
    # write data to .hdf5 container
    bproc.writer.write_hdf5("output/", data)


### MAIN-METHOD ###
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='random test data generation tool',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--helpme", "-help", action="help", help="Show the helper")
    
    parser.add_argument('--material_path', '-mat', help='Path for material resources', default=MATERIALS_PATH)
    parser.add_argument('--mesh_path', '-mesh', help='Path to .OBJ or .BLEND resource', default=MESH_PATH)

    parser.add_argument('--num_meshes', '-objects', help='Number of objects within one image', default=NUM_OF_OBJECTS)
    parser.add_argument('--num_perspectives', '-cam', help='Number of different perspectives', default=NUM_OF_PERSPECTIVES)
    parser.add_argument('--num_lightsources', '-light', help='Number of lightsources used in image', default=NUM_OF_LIGHTSOURCES)

    # projection
    parser.add_argument('--projection', '-proj', action='store_true', help='Enable projection mode')
    parser.add_argument('--proj_pattern', '-pat', choices=['points', 'stripes'], default='stripes', help='Define projection pattern')
    parser.add_argument('--num_pattern', '-npat', type=int, help='Number of points or stripes')
    parser.add_argument('--infrared', action='store_true', help='Turn off all additional lightsources')

    args = parser.parse_args()

    # check projection dependencies
    #if not args.projection and (args.proj_pattern or args.infrared):
    #    parser.error("Invalid argument: 'proj_pattern', 'infrared' only allowed if 'proj' is set")

    # Handle the default value for '--num_pattern' based on the value of '--proj_pattern'
    if args.num_pattern is None:
        if args.proj_pattern == 'stripes':
            args.num_pattern = 5
        elif args.proj_pattern == 'points':
            args.num_pattern = 2560


    testDataGenerator(args)
