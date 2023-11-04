import os
import argparse
import cv2 as cv
import numpy as np
import torch
import torchvision.transforms as transforms
from torchmetrics.image import StructuralSimilarityIndexMeasure as  SSIM
from torchmetrics.regression import MeanSquaredError


FOLDER = ""
PATTERN = r"(\d{3})_depth"

def calculate_mse(folder, pattern, save_maps):

    number_total = 0
    ssim_total = 0
    mse_total = 0
    mse_min = ""
    mse_min_val = -1
    mse_max = ""
    mse_max_val = -1
    sorted_files = {}
    result_file = "results_error.txt"
    
    # group depth images
    for filename in os.listdir(folder):
        if "depth" in filename:
            number = filename[0:3]
            if number not in sorted_files:
                sorted_files[number] = []
            sorted_files[number].append(filename)

    # calculate mse
    for number, files in sorted_files.items():
        if len(files) != 2:
            raise ValueError(f"wrong number of files with num: {number}")
        
        # convert to tensor
        transform = transforms.Compose([transforms.ToTensor()])
        img1 = cv.imread(os.path.join(folder, files[0]), cv.IMREAD_GRAYSCALE)
        img2 = cv.imread(os.path.join(folder, files[1]), cv.IMREAD_GRAYSCALE)

        img1_tensor = transform(img1).unsqueeze(0)
        img2_tensor = transform(img2).unsqueeze(0)
    
        mse = MeanSquaredError()(img1_tensor, img2_tensor)
        ssim = SSIM()(img1_tensor, img2_tensor)

        if save_maps:
            ## MSE ##
            abs_diff = cv.absdiff(img1, img2)
            squared_error = abs_diff.astype(np.float32) ** 2
            # normalize and apply colormap
            norm_error = cv.normalize(squared_error, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
            errormap = cv.applyColorMap(norm_error, cv.COLORMAP_JET)
            cv.imwrite(os.path.join(folder,f"{int(number):03d}_zerrormap_mse.png"), errormap)

        print(f"me:ansquared: {mse}")
        if mse < mse_min_val or mse_min_val == -1:
            mse_min = files[0]
            mse_min_val = mse
        if mse > mse_max_val or mse_max_val == -1:
            mse_max = files[1]
            mse_max_val = mse
            
        mse_total += mse
        ssim_total += ssim
        if int(number) > number_total:
            number_total = int(number)
    mse_total /= number_total
    ssim_total /= number_total

    # save in file
    with open(os.path.join(folder, result_file), 'a') as file:
        msg = f"MSE: {mse_total:.3f}\nMSE_min: {mse_min}: {mse_min_val:.3f}\nMSE_max: {mse_max}: {mse_max_val:.3f}\nSSIM: {ssim_total:.3f}\n"
        file.write(msg)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='create errorvalues for depth estimation results',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('folder', help="path to folder with depth results", default=FOLDER)
    parser.add_argument('--save_errormap', action=argparse.BooleanOptionalAction, help='save errormaps of folder', default=True)
    #parser.add_argument('pattern', help="pattern of depth files: e.g. r'\d{3})_depth'", default=PATTERN)
    
    args = parser.parse_args()

    calculate_mse(args.folder, PATTERN, args.save_errormap)
