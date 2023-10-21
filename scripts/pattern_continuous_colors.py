import numpy as np
from PIL import Image, ImageDraw
import argparse

WIDTH = 1278 
HEIGHT = 720
NUM_PEAKS = 3
OUTPUT = "../PATTERN_CONTINUOUS.png"

# width must be a multiple of 2xNUM_PEAKS, otherwise the pattern will be larger than the viewport!
def create_pattern_continuous(output=OUTPUT, width=WIDTH, height=HEIGHT, peaks=NUM_PEAKS):
    
    # resize width to fit intensity peaks
    doublePeak = 2*peaks
    if width % doublePeak != 0:
        width = ((width // doublePeak) * doublePeak) + doublePeak

    peak_intensity = 255
    base_length = width//peaks 
    distance_between_peaks = base_length//peaks 

    image = Image.new("RGB", (width, height), (255,255,255))
    draw = ImageDraw.Draw(image)


    def create_peak_array(base_length, shift):
        step = peak_intensity // (base_length / 2)
        
        total_length = peaks * base_length
        peak_array = np.zeros(int(total_length), dtype = int)

        for i in range(peaks):
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
        draw.line([(i, 0), (i, height)], fill=(red[i], green[i], blue[i]))


    image.save(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pattern generator for continuous color code pattern',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--width', help='width of pattern', default=WIDTH)
    parser.add_argument('--height', help='height of pattern', default=HEIGHT)
    parser.add_argument('--output', help='output path for pattern', default=OUTPUT)

    args = parser.parse_args()
    create_pattern_continuous(args.width, args.height, NUM_PEAKS, args.output)