import time
from PIL import Image
from PIL import ImageDraw

from Adafruit_LED_Backpack import BicolorMatrix8x8

def straight_face(color):
    print "Straight face - Yellow"
    display = BicolorMatrix8x8.BicolorMatrix8x8()
    display.begin()

    display.clear()

    x_coordinates_straight = [0,0,0,0,1,1,2,2,2,2,3,3,4,4,4,4,5,4,4,5,6,6,7,7,7,7]
    y_cooordinates_straight = [2,3,4,5,1,6,0,2,5,7,0,7,0,2,5,7,0,3,4,7,1,6,2,3,4,5]

    for x,y in zip(x_coordinates_straight,y_cooordinates_straight):
        display.set_pixel(x, y, color)
        display.write_display()
    time.sleep(1)

def smiling_face(color):
    display = BicolorMatrix8x8.BicolorMatrix8x8()
    display.begin()

    print "Smiley Face - Green"

    x_coordinates = [0,0,0,0,1,1,2,2,2,2,3,3,4,4,4,4,5,5,5,5,6,6,7,7,7,7]
    y_cooordinates = [2,3,4,5,1,6,0,2,5,7,0,7,0,2,5,7,0,3,4,7,1,6,2,3,4,5]

    for x,y in zip(x_coordinates,y_cooordinates):
        display.set_pixel(x, y,color)
        display.write_display()
    time.sleep(2)

def colour_gauge(smile_count,seconds_elapsed):
    print "Colour Gauge"
    display = BicolorMatrix8x8.BicolorMatrix8x8()
    display.begin()

    display.clear()

    image = Image.new('RGB', (8, 8))
    draw = ImageDraw.Draw(image)

    display.clear()
    if 0 < smile_count <= 10 or smile_count > 0:
        x_coordinates_gauge = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7, 7]
        y_cooordinates_gauge =[2, 3, 4, 5, 1, 6, 0, 7, 0, 7, 0, 7, 0, 7, 1, 6, 2, 3, 4, 5]

        for x, y in zip(x_coordinates_gauge, y_cooordinates_gauge):
            draw.point((x,y),fill=(255,0,0))

    # time.sleep(1)

    if 10 < smile_count <= 20 or smile_count > 10:
        draw.line((6, 2, 6, 5), fill=(255, 0, 0))

    # time.sleep(1)

    if 20 < smile_count <= 30 or smile_count > 20:
        draw.line((5, 1, 5, 6), fill=(255, 255, 0))

    # time.sleep(1)

    if 30 < smile_count <= 40 or smile_count > 30:
        draw.line((4, 1, 4, 6), fill=(255, 255, 0))

    # time.sleep(1)

    if 40 < smile_count <= 50 or smile_count > 40:
        draw.line((3, 1, 3, 6), fill=(255, 255, 0))

    # time.sleep(1)

    if 50 < smile_count <= 60 or smile_count > 50:
        draw.line((2, 1, 2, 6), fill=(0, 255, 0))

    # time.sleep(1)

    if 60 < smile_count <= 70 or smile_count > 60:
        draw.line((1, 2, 1, 5), fill=(0, 255, 0))

    # time.sleep(5)

    display.set_image(image)
    display.write_display()

    display.clear()
    display.set_image(display.create_blank_image())
    display.write_display()

    if seconds_elapsed%10 == 0:
        scrollable = display.horizontal_scroll(image)
        display.animate(scrollable)
