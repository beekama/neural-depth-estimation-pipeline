import h5py
import argparse
import os
from PIL import Image

def extract_images(args):
    counter = 0
    # read folders file for file
    for folder in args.folder_in:
        for filename in os.listdir(folder):
            file = os.path.join(folder, filename)
            # extract image and depth
            hfile = h5py.File(file, "r+")
            img_data = Image.fromarray(hfile['colors'][()])
            if img_data.mode != 'RGB':
                img_data = img_data.convert('RGB')
            img_data.save(args.output_dir + "/images/" + f"{counter:03d}.png")

            normalized = hfile['depth'][()] * 256
            depth_data = Image.fromarray(normalized)
            print(depth_data.mode)
            #depth_data.convert('RGB')
            if depth_data.mode != 'L':
                depth_data = depth_data.convert('L')
            depth_data.save(args.output_dir + "/depths/" + f"{counter:03d}.png")
            counter+=1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='extract images and depth from hdf5-files',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--folder_in', '-f', help='location of hdf5-folder', nargs='+')
    parser.add_argument('--output_dir', '-o', help='location of output-folder for images and depth', required=True)
    args = parser.parse_args()

    # check if output dir exists and is empty
    if os.path.exists(args.output_dir):
        if len(os.listdir(args.output_dir)) != 0:
            parser.error("output directory must be empty")
    os.makedirs(args.output_dir + "/images")
    os.makedirs(args.output_dir + "/depths")

    extract_images(args)
