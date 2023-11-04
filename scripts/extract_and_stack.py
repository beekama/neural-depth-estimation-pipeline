import h5py
import argparse
import os
from PIL import Image
import numpy as np

# take NORMAL and INFRARED and combine it into output
def extract_images(folder_in_A, foulder_in_B, folder_out):
    # deterministische bearbeitung bei wildcards
    os.environ["LC_COLLATE"]="en_US.UTF-8" 
    
    os.makedirs(folder_out + "/train", exist_ok=True)
    os.makedirs(folder_out + "/valid", exist_ok=True)
    os.makedirs(folder_out + "/test", exist_ok=True)
    os.makedirs(folder_out + "/depth_maps", exist_ok=True)
    os.makedirs(folder_out + "/disparity", exist_ok=True)
    os.makedirs(folder_out + "/disparity_depth", exist_ok=True)

    # count hdf5-files
    files = os.listdir(folder_in_A)
    num_of_files = len([file for file in files])
    threshold_train = int(num_of_files*0.7)
    threshold_valid = int(num_of_files*0.9)

    counter = 0
    for filename in files:
        file_A = os.path.join(folder_in_A, filename)
        file_B = os.path.join(foulder_in_B, filename)
        if os.path.exists(file_B):
            # extract image A
            hfile_A = h5py.File(file_A, "r+")
            img_array_A= hfile_A["colors"][0]
            img_data_A = Image.fromarray(img_array_A)
            if img_data_A.mode != 'RGB':
                img_data_A = img_data_A.convert('RGB')

            # extract image B
            hfile_B = h5py.File(file_B, "r+")
            img_array_B= hfile_B["colors"][0]
            img_data_B = Image.fromarray(img_array_B)
            if img_data_B.mode != 'RGB':
                img_data_B = img_data_B.convert('RGB')

            if img_data_A.size != img_data_B.size:
                raise ValueError("Both images must have same size")
            
            # split images into rbg channels
            r_a, g_a, b_a = img_data_A.split()
            r_b, g_b, b_b = img_data_B.split()

            # combine images
            combined_array = np.dstack((img_array_A, img_array_B))
            combined_array_reshaped = combined_array.reshape(combined_array.shape[0], -1)

            # save images
            if (counter < threshold_train):
                np.savetxt(folder_out + "/train/" + f"{counter:03d}.png", combined_array_reshaped)
                #combined_image.save(folder_out + "/train/" + f"{counter:03d}.png")
            elif (counter < threshold_valid):
                #combined_image.save(folder_out + "/valid/" + f"{counter:03d}.png")
                np.savetxt(folder_out + "/valid/" + f"{counter:03d}.png", combined_array_reshaped)
            else:
                #combined_image.save(folder_out + "/test/" + f"{counter:03d}.png")
                np.savetxt(folder_out + "/test/" + f"{counter:03d}.png", combined_array_reshaped)

            # save disparity
            if hfile_A["colors"].size > 1:
                print("shape disparity" + str(hfile_A["colors"][1].shape))
                combined_image = Image.fromarray(hfile_A["colors"][1])
                if combined_image.mode != 'RGB':
                    combined_image = combined_image.convert('RGB')
                combined_image.save(folder_out + "/disparity/" + f"{counter:03d}.png")

            # save disparity-depth
            if hfile_A["depth"].size > 1:
                print((hfile_A["depth"][1].shape))
                normalized = hfile_A['depth'][1] * 256
                depth_data = Image.fromarray(normalized)
                if depth_data.mode != 'L':
                    depth_data = depth_data.convert('L')
                depth_data.save(folder_out + "/disparity_depth/" + f"{counter:03d}.png")

            # save depth
            normalized = hfile_A['depth'][0] * 256
            depth_data = Image.fromarray(normalized)
            print(depth_data.mode)
            if depth_data.mode != 'L':
                depth_data = depth_data.convert('L')
            depth_data.save(folder_out + "/depth_maps/" + f"{counter:03d}.png")
            counter+=1
        else:
            raise ValueError(f"File {filename} does not exist in folder {foulder_in_B} but in folder {folder_in_A}")
        
# take STEREO images and combine it into output
def extract_images(folder_in, folder_out):
    # deterministische bearbeitung bei wildcards
    os.environ["LC_COLLATE"]="en_US.UTF-8" 
    
    os.makedirs(folder_out + "/train", exist_ok=True)
    os.makedirs(folder_out + "/valid", exist_ok=True)
    os.makedirs(folder_out + "/test", exist_ok=True)
    os.makedirs(folder_out + "/depth_maps", exist_ok=True)
    #os.makedirs(folder_out + "/disparity", exist_ok=True)
    #os.makedirs(folder_out + "/disparity_depth", exist_ok=True)

    # count hdf5-files
    files = os.listdir(folder_in)
    num_of_files = len([file for file in files])
    threshold_train = int(num_of_files*0.7)
    threshold_valid = int(num_of_files*0.9)

    counter = 0
    for filename in files:
        file = os.path.join(folder_in, filename)
        hfile = h5py.File(file, "r+")

        # extract LEFT image 
        img_array_left= hfile["colors"][0]
        img_data_left = Image.fromarray(img_array_left)
        if img_data_left.mode != 'RGB':
            img_data_left = img_data_left.convert('RGB')
  
        # extract RIGTH image
        img_array_right= hfile["colors"][1]
        img_data_right = Image.fromarray(img_array_right)
        if img_data_right.mode != 'RGB':
            img_data_right = img_data_right.convert('RGB')

        if img_data_left.size != img_data_right.size:
            raise ValueError("Both images must have same size")
            
        # split images into rbg channels
        r_a, g_a, b_a = img_data_left.split()
        r_b, g_b, b_b = img_data_right.split()

        # combine images
        combined_array = np.dstack((img_array_left, img_array_right))
        combined_array_reshaped = combined_array.reshape(combined_array.shape[0], -1)

        # save images
        if (counter < threshold_train):
            np.savetxt(folder_out + "/train/" + f"{counter:03d}.png", combined_array_reshaped)
            #combined_image.save(folder_out + "/train/" + f"{counter:03d}.png")
        elif (counter < threshold_valid):
            #combined_image.save(folder_out + "/valid/" + f"{counter:03d}.png")
            np.savetxt(folder_out + "/valid/" + f"{counter:03d}.png", combined_array_reshaped)
        else:
            #combined_image.save(folder_out + "/test/" + f"{counter:03d}.png")
            np.savetxt(folder_out + "/test/" + f"{counter:03d}.png", combined_array_reshaped)


        # save depth
        normalized = hfile['depth'][0] * 256
        depth_data = Image.fromarray(normalized)
        print(depth_data.mode)
        if depth_data.mode != 'L':
            depth_data = depth_data.convert('L')
        depth_data.save(folder_out + "/depth_maps/" + f"{counter:03d}.png")
        counter+=1
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='extract images and depth from hdf5-files',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--folder_in1', '-f1', help='location of hdf5-folder')
    parser.add_argument('--folder_in2', '-f2', help='location of hdf5-folder')
    parser.add_argument('--stereo', action=argparse.BooleanOptionalAction, help='enable/disable image-generation', default=False)
    parser.add_argument('--output_dir', '-o', help='location of output-folder for images and depth', required=True)
    args = parser.parse_args()
    if args.stereo:
        if args.folder_in1 and args.folder_in2:
            raise ValueError("For stereo mode only one input folder is allowed!")
        extract_images(args.folder_in1, args.output_dir)
    else:
        if args.folder_in1 == None or args.folder_in2 == None:
            raise ValueError("two input folders required for non-stereo stacking!")
        extract_images(args.folder_in1, args.folder_in2, args.output_dir)

    # todo extract for stereos