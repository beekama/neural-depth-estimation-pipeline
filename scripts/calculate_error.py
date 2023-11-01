import os
import argparse
import cv2 as cv
import numpy as np

FOLDER = ""
PATTERN = r"(\d{3})_depth"

def calculate_mse(folder, pattern, save_maps):

    number_total = 0
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
        print(f"filenumber: {number}")
        for file in files:
            print(f"- {file}")
        abs_diff = cv.absdiff(cv.imread(os.path.join(folder, files[0]), cv.IMREAD_GRAYSCALE), cv.imread(os.path.join(folder, files[1]), cv.IMREAD_GRAYSCALE))
        squared_error = abs_diff.astype(np.float32) ** 2
        mse_pp = np.mean(squared_error)

        if save_maps:
            norm_error = cv.normalize(squared_error, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
            errormap = cv.applyColorMap(norm_error, cv.COLORMAP_JET)
            cv.imwrite(f"{number:03}_errormap.png", errormap)

        print(f"meansquared: {mse_pp}")
        if mse_pp < mse_min_val or mse_min_val == -1:
            mse_min = file
            mse_min_val = mse_pp
        if mse_pp > mse_max_val or mse_max_val == -1:
            mse_max = file
            mse_max_val = mse_pp
            
        mse_total += mse_pp
        if int(number) > number_total:
            number_total = int(number)
    mse_total /= number_total

    # save in file
    with open(os.path.join(folder, result_file), 'a') as file:
        msg = f"MSE: {mse_total:.3f}\nMSE_min: {mse_min}: {mse_min_val:.3f}\nMSE_max: {mse_max}: {mse_max_val:.3f}\n"
        file.write(msg)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='create errorvalues for depth estimation results',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('folder', help="path to folder with depth results", default=FOLDER)
    parser.add_argument('--save_errormap', action=argparse.BooleanOptionalAction, help='save errormaps of folder', default=False)
    #parser.add_argument('pattern', help="pattern of depth files: e.g. r'\d{3})_depth'", default=PATTERN)
    
    args = parser.parse_args()

    calculate_mse(args.folder, PATTERN, args.save_errormap)
