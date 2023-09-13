import h5py
import argparse
import os
from PIL import Image

def extract_images(folder_in, folder_out):
    # deterministische bearbeitung bei wildcards
    os.environ["LC_COLLATE"]="en_US.UTF-8" 
    
    os.makedirs(folder_out + "/train", exist_ok=True)
    os.makedirs(folder_out + "/valid", exist_ok=True)
    os.makedirs(folder_out + "/test", exist_ok=True)
    os.makedirs(folder_out + "/depth_maps", exist_ok=True)
    os.makedirs(folder_out + "/disparity", exist_ok=True)
    os.makedirs(folder_out + "/disparity_depth", exist_ok=True)

    # count hdf5-files
    num_of_files = len([file for file in os.listdir(folder_in)])
    threshold_train = int(num_of_files*0.7)
    threshold_valid = int(num_of_files*0.9)

    counter = 0
    for filename in os.listdir(folder_in):
        file = os.path.join(folder_in, filename)
        # extract image and depth
        hfile = h5py.File(file, "r+")
        img_data = Image.fromarray(hfile["colors"][0])
        if img_data.mode != 'RGB':
            img_data = img_data.convert('RGB')
        if (counter < threshold_train):
            img_data.save(folder_out + "/train/" + f"{counter:03d}.png")
        elif (counter < threshold_valid):
            img_data.save(folder_out + "/valid/" + f"{counter:03d}.png")
        else:
            img_data.save(folder_out + "/test/" + f"{counter:03d}.png")
        print("shape color" + str(hfile["colors"][0].shape))

        if hfile["colors"].size > 1:
            print("shape disparity" + str(hfile["colors"][1].shape))
            img_data = Image.fromarray(hfile["colors"][1])
            if img_data.mode != 'RGB':
                img_data = img_data.convert('RGB')
            img_data.save(folder_out + "/disparity/" + f"{counter:03d}.png")

        if hfile["depth"].size > 1:
            print((hfile["depth"][1].shape))
            normalized = hfile['depth'][1] * 256
            depth_data = Image.fromarray(normalized)
            if depth_data.mode != 'L':
                depth_data = depth_data.convert('L')
            depth_data.save(folder_out + "/disparity_depth/" + f"{counter:03d}.png")

        print((hfile["depth"].shape))
        normalized = hfile['depth'][0] * 256
        depth_data = Image.fromarray(normalized)
        print(depth_data.mode)
        #depth_data.convert('RGB')
        if depth_data.mode != 'L':
            depth_data = depth_data.convert('L')
        depth_data.save(folder_out + "/depth_maps/" + f"{counter:03d}.png")
        counter+=1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='extract images and depth from hdf5-files',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--folder_in', '-f', help='location of hdf5-folder', required=True)
    parser.add_argument('--output_dir', '-o', help='location of output-folder for images and depth', required=True)
    args = parser.parse_args()
    extract_images(args.folder_in, args.output_dir)
