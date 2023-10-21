from PIL import Image, ImageDraw
from itertools import permutations
import argparse

WIDTH = 1278
HEIGHT = 720
OUTPUT = "../PATTERN_STRIPES.png"

# width must be a multiple of 18, otherwise the pattern will be larger than the viewport!
def create_pattern_stripes(output_path=OUTPUT, width=WIDTH, height=HEIGHT):
    GRAY = (150, 150, 150)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    # resize width to fit stripes (18 stripes)
    if width % 18 != 0:
        width = ((width // 18) * 18) + 18

    image = Image.new("RGB", (width, height), (255,255,255))
    draw = ImageDraw.Draw(image)

    modi = [GRAY, WHITE, BLACK]

    combinations =  list(permutations(modi))
    strip_width = WIDTH // (len(combinations)*len(modi))

    linecount = 0
    for combi in combinations:
        for color in range(len(combi)):
            for col in range(strip_width):
                draw.line([(linecount, 0), (linecount, height)], fill=combi[color])
                linecount+=1
    image.save(output_path)
    #image.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pattern generator for striped repeated grayscale pattern',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--width', help='width of pattern', default=WIDTH)
    parser.add_argument('--height', help='height of pattern', default=HEIGHT)
    parser.add_argument('--output', help='output path for pattern', default=OUTPUT)

    args = parser.parse_args()
    create_pattern_stripes(args.width, args.height, args.output)