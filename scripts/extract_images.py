import h5py
import argparse
import os
from PIL import Image
import glob



'''
def count_files_in_folder(paths):
    count = 0
    for path in paths:
        for root, dirs, files in os.walk(path):
            count += len(files)
    return count # todo not used
'''
def extract_images(folder_in, folder_out):
    # deterministische bearbeitung bei wildcards
    os.environ["LC_COLLATE"]="en_US.UTF-8" 

    # check if output dir exists and is empty
    # os.path.exists(folder_out):
    #     if len(os.listdir(folder_out)) != 0:
    #         raise Exception("output directory must be empty")
    os.makedirs(folder_out + "/train", exist_ok=True)
    os.makedirs(folder_out + "/test", exist_ok=True)
    os.makedirs(folder_out + "/depth_maps", exist_ok=True)

    # count hdf5-files
    num_of_files = len([file for file in os.listdir(folder_in)])
    files_treshhold = int(num_of_files*0.8)
    print(num_of_files)
    
    files_treshhold = int(num_of_files*0.8)
    counter = 0
    # read folders file for file
    #for filename in list(glob.iglob(folder_in, recursive=True)):
    for filename in os.listdir(folder_in):
        file = os.path.join(folder_in, filename)
        # extract image and depth
        hfile = h5py.File(file, "r+")
        img_data = Image.fromarray(hfile['colors'][()])
        if img_data.mode != 'RGB':
            img_data = img_data.convert('RGB')
        if (counter < files_treshhold):
            img_data.save(folder_out + "/train/" + f"{counter:03d}.png")
        else:
            img_data.save(folder_out + "/test/" + f"{counter:03d}.png")

        normalized = hfile['depth'][()] * 256
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
