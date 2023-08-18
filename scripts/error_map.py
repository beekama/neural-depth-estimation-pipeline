import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import argparse

def create_error_map(img_1, img_2):
    imgL = cv.imread(img_1, cv.IMREAD_GRAYSCALE)
    imgR = cv.imread(img_2, cv.IMREAD_GRAYSCALE)

    # calculate mean squared difference
    abs_diff = cv.absdiff(imgL, imgR)
    squared_error = abs_diff.astype(np.float32) ** 2
    mse_pp = np.mean(squared_error)
    # normalize and apply colormap
    norm_error = cv.normalize(squared_error, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    errormap = cv.applyColorMap(norm_error, cv.COLORMAP_JET)

    cv.imwrite("errormap.png", errormap)
    plt.imshow(errormap)
    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='create errormap for depth estimation results',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--images', '-i', metavar=('est-file', 'gt-file'), help='both images used for error detection', required=True)
    
    args = parser.parse_args()
    img_1, img_2 = args.images

    create_error_map(img_1, img_2)

