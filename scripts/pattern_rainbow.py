from PIL import Image, ImageDraw
import argparse

WIDTH = 1280
HEIGHT = 720
OUTPUT = "../PATTERN_RAINBOW.png"

# width must be a even, otherwise the pattern will be larger than the viewport!
def create_pattern_rainbow(output=OUTPUT, width=WIDTH, height=HEIGHT):

    # resize width be even
    if width % 2 != 0:
        width += 1 

    image = Image.new("RGB", (width, height), (255,255,255))
    draw = ImageDraw.Draw(image)

    color1 = (255, 0, 0)
    color2 = (0, 255, 0)
    color3 = (0, 0, 255)

    for col in range(width):
        pct = (col *2*100 / (width-1))/100 
        if col < width/2:
            grad = (
                int((1-pct) * color1[0] + pct * color2[0]),
                int((1-pct) * color1[1] + pct * color2[1]),
                0
            )
        else:
            pct -=1
            grad = (
                0,
                int((1-pct) * color2[1] + pct * color3[1]),
                int((1-pct) * color2[2] + pct * color3[2])
            )
        draw.line([(col, 0), (col, height)], fill=grad)

    image.save(output)
    #image.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pattern generator for rainbow pattern',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--width', help='width of pattern', default=WIDTH)
    parser.add_argument('--height', help='height of pattern', default=HEIGHT)
    parser.add_argument('--output', help='output path for pattern', default=OUTPUT)

    args = parser.parse_args()
    create_pattern_rainbow(args.width, args.height, args.output)