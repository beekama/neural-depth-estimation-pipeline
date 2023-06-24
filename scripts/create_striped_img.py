import numpy as np

import matplotlib.pyplot as plt


def create_img(num_stripes, width=1280, height=720):
    array = np.zeros((height, width, 4), dtype=np.uint8)
    num_stripes = num_stripes*2
    stripe_width = width // num_stripes

    for i in range(num_stripes):
        start = i * stripe_width
        end = start + stripe_width

        if i % 2 == 0:
            # White stripe
            array[:, start:end] = [255,255,255,255]
        else:
            pass

    #plt.imshow(array, cmap='gray')
    #plt.show()
    return array

if __name__ == "__main__":
    create_img(5)