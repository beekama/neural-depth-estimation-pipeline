#!/usr/bin/python3

### IMPORTS ###
import blenderproc as bproc
import argparse
import numpy as np
from PIL import Image
import cv2 as cv
from random import randrange
import bpy
import os
import random

# import helper-scripts
import sys
sys.path.append('scripts')
import readhdf5, export_scene, pattern_rainbow, pattern_continuous_colors, patter_stripes_repeated_grayscale

### DEFAULTS ###
MATERIALS_PATH = 'resources/materials'
MESH_PATH = 'resources/test/' # todo consitens
OUTPUT_DIR = 'output'

NUM_OF_POSES = 5
NUM_OF_OBJECTS = 3
NUM_OF_PERSPECTIVES = 5
NUM_OF_LIGHTSOURCES = 1
 
SAVE_SCENE = False
LOAD_SCENE = False
SAVE_INTRINSICS = False

PATTERN = "points"


light_sources = []


def save_pattern(pattern, filename):
    arr_reshaped = pattern.reshape(pattern.shape[0], -1)
    np.savetxt(filename, arr_reshaped)


def load_pattern(filename):
    pat = np.loadtxt(filename)
    return pat.reshape( 720, 1280, 4)


def normalizeAndSave(bproc, data, outputfolder):
    # normalize depth
    depth = data['depth']
    data['depth'] = readhdf5.normalize(depth, 0, 1)
    # write data to .hdf5 container
    bproc.writer.write_hdf5(outputfolder, data, append_to_existing_output=True)


def normalos(bproc, objects):
    bproc.lighting.light_surface([obj for obj in objects if (obj.get_name() == "Ceiling" or obj.get_name() == "Wall_Plane")], emission_strength=4.0, emission_color=[1,1,1,1])
    # disable projector
    [ls.set_energy(0) for ls in light_sources if ls.get_name() == "projector"]
    # render whole pipelinem
    bproc.renderer.toggle_stereo(True)
    data = bproc.renderer.render()
    # Apply stereo matching to each pair of images
    data["stereo-depth"], data["disparity"] = bproc.postprocessing.stereo_global_matching(data["colors"], disparity_filter=False)
    normalizeAndSave(bproc, data, args.output + "/NORMALOS")
   

def pattern(bproc, objects, bvh_tree, pattern_type):
    bproc.lighting.light_surface([obj for obj in objects if (obj.get_name() == "Ceiling" or obj.get_name() == "Wall_Plane")], emission_strength=4.0, emission_color=[1,1,1,1])

    if pattern_type == "points":
        try:
            pattern_img = load_pattern("PAN.txt")
        except FileNotFoundError:
            pattern_img = bproc.utility.generate_random_pattern_img(1280, 720, args.num_pattern)
    elif pattern_type == "rainbow":
        try:
            pattern_img = cv.cvtColor(cv.imread("PATTERN_RAINBOW.png"),cv.COLOR_RGB2RGBA)
        except cv.error:
            pattern_rainbow.create_pattern_rainbow("PATTERN_RAINBOW.png")
            pattern_img = cv.cvtColor(cv.imread("PATTERN_RAINBOW.png"),cv.COLOR_RGB2RGBA)
    elif pattern_type == "stripes":
        try:
            pattern_img = cv.cvtColor(cv.imread("PATTERN_STRIPES.png"),cv.COLOR_RGB2RGBA)
        except cv.error:
            patter_stripes_repeated_grayscale.create_pattern_stripes("PATTERN_STRIPES.png")
            pattern_img = cv.cvtColor(cv.imread("PATTERN_STRIPES.png"),cv.COLOR_RGB2RGBA)
    elif pattern_type == "continuous":
        try:
            pattern_img = cv.cvtColor(cv.imread("PATTERN_CONTINUOUS.png"),cv.COLOR_RGB2RGBA)
        except cv.error:
            pattern_continuous_colors.create_pattern_continuous("PATTERN_CONTINUOUS.png")
            pattern_img = cv.cvtColor(cv.imread("PATTERN_CONTINUOUS.png"),cv.COLOR_RGB2RGBA)
    else:
        raise ValueError("Patterntype unknown!")
            
    proj = bproc.types.Light()
    proj.set_type('SPOT')
    proj.set_energy(3000)
    proj.setup_as_projector(pattern_img)
    proj.set_name("projector")
    light_sources.append(proj)

    # translate camera-position to create distance to projector-position
    cam2world = bproc.camera.get_camera_pose()
    rot_matrix = cam2world[0:3, 0:3]
    upward = np.linalg.norm(rot_matrix[:,1]) 
    cam2world[0:3,3]+= 0.1 * upward
    # replace camera-position
    bproc.utility.reset_keyframes()

    # check if new camera position is still interesting enough
    if bproc.camera.perform_obstacle_in_view_check(cam2world, {"min":0.8}, bvh_tree) and \
                bproc.camera.scene_coverage_score(cam2world) > 0.4:
        
        bproc.camera.add_camera_pose(cam2world)
        # render whole pipeline
        bproc.renderer.toggle_stereo(True)
        data = bproc.renderer.render()
        # Apply stereo matching to each pair of images
        data["stereo-depth"], data["disparity"] = bproc.postprocessing.stereo_global_matching(data["colors"], disparity_filter=False)
        normalizeAndSave(bproc, data, args.output + "/PATTERN")
    else:
        print("Relocated camera position too close to obstacle or too little szene coverage!", file=sys.stderr)
        sys.exit(1)

def infrared(bproc, objects):
    bproc.lighting.light_surface([obj for obj in objects if (obj.get_name() == "Ceiling" or obj.get_name() == "Wall_Plane")], emission_strength=0.0, emission_color=[1,1,1,1])
    # render whole pipeline
    bproc.renderer.toggle_stereo(True)
    data = bproc.renderer.render()
    # Apply stereo matching to each pair of images
    data["stereo-depth"], data["disparity"] = bproc.postprocessing.stereo_global_matching(data["colors"], disparity_filter=False)
    normalizeAndSave(bproc, data, args.output + "/INFRARED")


def testDataGenerator(args):
    
    # enable cycles renderer and sets some speedup options for rendering
    bproc.init()
    bproc.renderer.enable_depth_output(activate_antialiasing=False, file_prefix='depth_', convert_to_distance=False)

    if LOAD_SCENE:
        objects = bproc.loader.load_obj(args.load_scene)
        
        # choose materials for wall, ceiling, floor randomly
        for obj in objects:
            print(obj.get_name())
            if obj.get_name() == "Wall_Plane":
                obj.replace_materials(random.choice(materials))
            elif "Floor_Plane" in obj.get_name(): 
                obj.replace_materials(random.choice(materials))
            elif "Ceiling_Plane" in obj.get_name():
                obj.replace_materials(random.choice(materials))

        cam2world_matrix = np.loadtxt(args.load_camera)
        bproc.camera.add_camera_pose(cam2world_matrix)
        bvh_tree = bproc.object.create_bvh_tree_multi_objects(objects)
    else:
        #RESOURCES = ['sofa_bunt.obj', 'cupboard.obj', 'kallax.obj', 'kommode.obj', 'klappstuhl.obj']
        RESOURCES = ['sofa_bunt.obj', 'cupboard.obj', 'klappstuhl.obj']
        
        
        # load materias and objects
        materials = bproc.loader.load_ccmaterials(args.material_path, ["Bricks", "Wood", "Carpet", "Tile", "Marble"])
        interior_objects = []
        for i in range (args.num_meshes):
            #interior_objects.extend(bproc.loader.load_blend(args.mesh_path +  RESOURCES[randrange(len(RESOURCES))]))
            interior_objects.extend(bproc.loader.load_obj(args.mesh_path +  RESOURCES[randrange(len(RESOURCES))]))
        
        # construct random room
        objects = bproc.constructor.construct_random_room(used_floor_area=45, interior_objects=interior_objects,materials=materials, amount_of_extrusions=5)
        
        # light sources
        #if not args.infrared:
        #    bproc.lighting.light_surface([obj for obj in objects if obj.get_name() == "Ceiling"], emission_strength=4.0, emission_color=[1,1,1,1])


        # init bvh tree containing all mesh objects
        bvh_tree = bproc.object.create_bvh_tree_multi_objects(objects)
        floor = [obj for obj in objects if obj.get_name() == "Floor"][0]
        poses = 0
        tries = 0

        # Get point-of-interest, the camera should look towards and determine starting point
        poi = bproc.object.compute_poi(objects)
        start_location = bproc.sampler.upper_region(floor, min_height=1.5, max_height=1.8)
        spline_vector = poi-start_location
        step_size = spline_vector/NUM_OF_POSES
        current_loc = start_location

        while tries < 10000 and poses < NUM_OF_POSES:
            # Sample random camera location above objects
            rand = np.random.uniform([-0.1, -0.3, 0], [0.1, 0.3, 0.05])
            location = current_loc + rand
            current_loc += step_size

            # Compute rotation based on vector going from location towards poi
            rotation_matrix = bproc.camera.rotation_from_forward_vec(poi-location, inplane_rot=np.random.uniform(-0.7854, 0.7854))
            # Add homog cam pose based on location an rotation
            cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
            
            # check that obstacles are at least 1 meter away from camera and make sure the view is interesting enough
            if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min":0.8}, bvh_tree) and \
                    bproc.camera.scene_coverage_score(cam2world_matrix) > 0.4 and \
                    floor.position_is_above_object(location):
                # persist camera pose
                bproc.camera.add_camera_pose(cam2world_matrix)
                poses += 1
            tries += 1
        
    if SAVE_SCENE:
        cam = bproc.camera.get_camera_pose()
        if SAVE_FOLDER:
            output_path_scene = os.path.join(args.save_scene, "saved_scene")
            output_path_camera = os.path.join(args.save_scene, "saved_camera")
            # check if files with the base filename already exist
            existing_files = [f for f in os.listdir(args.save_scene) if f.startswith("saved_scene")]
            # If there are existing files, determine the highest number from the filename - more efficient than .exists-loop
            existing_numbers = [int(f.split("_")[-1].split(".")[0]) for f in existing_files if f.count("_") == 2]

            if existing_numbers:
                num = max(existing_numbers) + 1
            else:
                num = 1

            output_path_scene = f"{output_path_scene}_{num:03d}.obj"
            export_scene.export_objects(bpy.data.objects, output_path_scene)

            # check if the numbering also fits save_camera-files
            if not (os.path.exists(os.path.join(args.save_camera, f"saved_camera_{(num-1):03d}.txt") and num>0) or
                os.path.exists(os.path.join(args.save_camera), f"saved_camera_{num:03d}.txt")):
                raise ValueError("save_camera - wrong file handling. more or less camera files than scene files!")
            
            output_path_camera = f"{output_path_camera}_{num:03d}.txt"
            export_scene.export_camera(cam, output_path_camera)
        else:
            # save scene/camera in specific file
            export_scene.export_objects(bpy.data.objects, args.save_scene)
            export_scene.export_camera(cam, args.save_camera)
    if SAVE_INTRINSICS:
        cam = bproc.camera.get_intrinsics_as_K_matrix()
        export_scene.export_camera(cam, args.save_camera_intrinsics)

    ### !IMPORTANT! if "pattern" is executed it should be executed first as it changes the camera position ###
    ### correct order: pattern -> infrared -> normalos
    #if args.full or args.projection:
    pattern(bproc, objects, bvh_tree, args.pattern_type)
    #if args.full or args.infrared:
    infrared(bproc, objects)
    #normalos(bproc, objects)


### MAIN-METHOD ###
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='random test data generation tool',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--helpme", "-help", action="help", help="Show the helper")
    
    parser.add_argument('--material_path', '-mat', help='Path for material resources', default=MATERIALS_PATH)
    parser.add_argument('--mesh_path', '-mesh', help='Path to .OBJ or .BLEND resource', default=MESH_PATH)

    parser.add_argument('--num_poses', '-poses', help='Number of poses within one frame', default=NUM_OF_POSES)
    parser.add_argument('--num_meshes', '-objects', help='Number of objects within one frame', default=NUM_OF_OBJECTS)
    parser.add_argument('--num_perspectives', '-cam', help='Number of different perspectives', default=NUM_OF_PERSPECTIVES)
    parser.add_argument('--num_lightsources', '-light', help='Number of lightsources used in image', default=NUM_OF_LIGHTSOURCES)

    parser.add_argument('--output', '-o', help='Path of output subfolder (main: NORMALOS, PATTERN, INFRARED)', default=OUTPUT_DIR) #todo remove tailing /

    # projection
    parser.add_argument('--projection', '-proj', action='store_true', help='Enable projection mode')
    parser.add_argument('--proj_pattern', '-pat', choices=['points', 'stripes'], default='points', help='Define projection pattern')
    parser.add_argument('--num_pattern', '-npat', type=int, help='Number of points or stripes')
    parser.add_argument('--infrared', action='store_true', help='Turn off all additional lightsources')
    parser.add_argument('--pattern_type', choices={'points', 'rainbow', 'stripes', 'continuous'}, help="select pattern type", default=PATTERN)

    parser.add_argument('--full', '-f', action='store_true', help='normal, pattern and infrared images')

    # inport export
    parser.add_argument('--load_scene', help='File to load scene from', default="")
    parser.add_argument('--load_camera', help='File to load camerapose from', default="")
    parser.add_argument('--save_scene', help='File/folder to save scene to', default="saved_scene.obj")
    parser.add_argument('--save_camera', help='File/folder to save camerapose to', default="saved_camera.txt")
    parser.add_argument('--save_camera_intrinsics', help='File to save camera-intrinsics from/to', default="saved_camera_intrinsics.txt")

    args = parser.parse_args()
    
    if int(args.num_poses) < 1 or int(args.num_poses) > 10:
        parser.error("Invalid argument: 'num_poses' must be between 1 and 10!")

    # check load or save scene
    LOAD_SCENE = args.load_scene != "" or args.load_camera != ""
    SAVE_SCENE = args.save_scene != "" or args.save_camera != ""
    SAVE_FOLDER = os.path.isdir(args.save_camera) and os.path.isdir(args.save_scene)
    SAVE_INTRINSICS = args.save_camera_intrinsics != ""
    
    if LOAD_SCENE and (args.load_scene == "" or args.load_camera == ""):
        parser.error("Invalid argument: both 'load_scene-file' and 'load_camera-file' must be set")
    if SAVE_SCENE and (args.save_scene == "" or args.save_camera == ""):
        parser.error("Invalid argument: both 'save_scene-file' and 'save_camera-file' must be set")
    if ((os.path.isdir(args.save_scene) and not os.path.isdir(args.save_camera)) or 
        (os.path.isdir(args.save_camera) and not os.path.isdir(args.save_scene))):
        parser.error("Invalid argument: 'save_scene-file' and 'save_camera-file' must be both file or both folder")


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
