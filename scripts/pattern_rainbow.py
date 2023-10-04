import numpy as np
from PIL import Image, ImageDraw

# check that even
WIDTH = 100
HEIGHT = 100

image = Image.new("RGB", (WIDTH, HEIGHT), (255,255,255))
draw = ImageDraw.Draw(image)

color1 = (255, 0, 0)
color2 = (0, 255, 0)
color3 = (0, 0, 255)

for col in range(WIDTH):
    pct = (col * 2*100 / (WIDTH-1))/100 
    if col < WIDTH/2:
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
    print("column: " + str(col) + ", gradient: " + str(grad) + ", pct: " + str(pct))
    draw.line([(col, 0), (col, HEIGHT)], fill=grad)

image.save("gradient_image.png")
#image.show()
