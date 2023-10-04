import numpy as np
from PIL import Image, ImageDraw

WIDTH = 270 ## todo check even
HEIGHT = 270
NUM_PEAKS = 3

peak_intensity = 255
base_length = 90 #WIDTH // NUM_PEAKS
distance_between_peaks = 30 #base_length // NUM_PEAKS

image = Image.new("RGB", (WIDTH, HEIGHT), (255,255,255))
draw = ImageDraw.Draw(image)


def create_peak_array(base_length, shift):
    step = 255/ (base_length / 2)
    
    total_length = NUM_PEAKS * base_length
    peak_array = np.zeros(total_length, dtype = int)

    for i in range(NUM_PEAKS):
        start_index = i * base_length + shift
        end_index = start_index + base_length

        if end_index > total_length:
            excess_length = end_index - total_length

            for j in range(start_index, min(total_length, end_index - excess_length)):
                peak_array[j] += int(min((j - start_index), (base_length - (j - start_index))) * step)

            for j in range(0, excess_length):
                wrapped_j = j + (base_length - excess_length) 
                peak_array[j] = int(min(wrapped_j, (base_length - (wrapped_j))) * step)
        else: 
            for j in range(start_index, end_index):
                peak_array[j] += int(min((j - start_index), (base_length - (j - start_index))) * step)

    return peak_array

red = create_peak_array(base_length, 0)
green = create_peak_array(base_length, distance_between_peaks)
blue = create_peak_array(base_length, 2*distance_between_peaks)

for i in range(len(red)):
    draw.line([(i, 0), (i, HEIGHT)], fill=(red[i], green[i], blue[i]))


image.save("continuouscolor_image.png")