import time
from PIL import Image
from PIL import ImageDraw

import sys
sys.path.insert(0, '/home/pi/OpenVINO/inference_engine_vpu_arm/inference_engine/samples/build/armv7l/Release/HappySpace/Adafruit_Python_LED_Backpack/Adafruit_LED_Backpack')
import BicolorMatrix8x8

def straight_face(color):
    """
    Function used to display a 8X8 Matrix Straight Face

    :param color: Change color using [BicolorMatrix8x8.RED, BicolorMatrix8x8.GREEN, BicolorMatrix8x8.YELLOW]
    """
    print("Straight face - Yellow")
    display = BicolorMatrix8x8.BicolorMatrix8x8()
    display.begin()

    display.clear()

    x_coordinates_straight = [0,0,0,0,1,1,2,2,2,2,3,3,4,4,4,4,5,4,4,5,6,6,7,7,7,7]
    y_cooordinates_straight = [2,3,4,5,1,6,0,2,5,7,0,7,0,2,5,7,0,3,4,7,1,6,2,3,4,5]

    for x,y in zip(x_coordinates_straight,y_cooordinates_straight):
        display.set_pixel(x, y, color)
        display.write_display()
    # time.sleep(0.1)

def smiling_face(color):
    """
    Function used to display a 8x8 Matrix Smiling Face

    :param color: Change color using [BicolorMatrix8x8.RED, BicolorMatrix8x8.GREEN, BicolorMatrix8x8.YELLOW]
    """
    display = BicolorMatrix8x8.BicolorMatrix8x8()
    display.begin()

    print("Smiley Face - Green")

    x_coordinates = [0,0,0,0,1,1,2,2,2,2,3,3,4,4,4,4,5,5,5,5,6,6,7,7,7,7]
    y_cooordinates = [2,3,4,5,1,6,0,2,5,7,0,7,0,2,5,7,0,3,4,7,1,6,2,3,4,5]

    for x,y in zip(x_coordinates,y_cooordinates):
        display.set_pixel(x, y,color)
        display.write_display()
    # time.sleep(0.3)

def colour_gauge_update(smile_count):
    """
    Function used to display a color gauge with the bottom two rows as red, middle three as yellow and top three as green
    based on the smile count

    :param smile_count: Receives value of the Total Smile Count from the main script

    """
    print("Colour Gauge")
    display = BicolorMatrix8x8.BicolorMatrix8x8()
    display.begin()
    display.clear()

    #Change color using [BicolorMatrix8x8.RED, BicolorMatrix8x8.GREEN, BicolorMatrix8x8.YELLOW]
    if 0 < smile_count <= 6 or smile_count > 0:
        display.set_pixel(7, 7, BicolorMatrix8x8.RED)

    if 7 < smile_count <= 12 or smile_count > 7:
        display.set_pixel(7, 6, BicolorMatrix8x8.RED)

    if 13 < smile_count <= 18 or smile_count > 13:
        display.set_pixel(7, 5, BicolorMatrix8x8.YELLOW)

    if 19 < smile_count <= 24 or smile_count > 19:
        display.set_pixel(7, 4, BicolorMatrix8x8.YELLOW)

    if 25 < smile_count <= 30 or smile_count > 25:
        display.set_pixel(7, 3, BicolorMatrix8x8.YELLOW)

    if 31 < smile_count <= 36 or smile_count > 31:
        display.set_pixel(7, 2, BicolorMatrix8x8.GREEN)

    if 37 < smile_count <= 42 or smile_count > 37:
        display.set_pixel(7, 1, BicolorMatrix8x8.GREEN)

    if smile_count > 42:
        display.set_pixel(7, 0, BicolorMatrix8x8.GREEN)

    display.write_display()


def colour_gauge(smile_count,seconds_elapsed):
    """
    Function used to display a color gauge based on smile ranges, bottom three display green, middle three yellow and top
    two red based on the smile count. Also includes, a horizontal scroll animation, if needed.

    :param smile_count: Receives value of the Total Smile Count from the main script
    """
    print("Colour Gauge")
    display = BicolorMatrix8x8.BicolorMatrix8x8()
    display.begin()

    display.clear()
    display.set_image(display.create_blank_image())
    display.write_display()

    image = Image.new('RGB', (8, 8))
    draw = ImageDraw.Draw(image)

    display.clear()
    if 0 < smile_count <= 10 or smile_count > 0:
        x_coordinates_gauge = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7, 7]
        y_cooordinates_gauge =[2, 3, 4, 5, 1, 6, 0, 7, 0, 7, 0, 7, 0, 7, 1, 6, 2, 3, 4, 5]

        for x, y in zip(x_coordinates_gauge, y_cooordinates_gauge):
            draw.point((x,y),fill=(0,255,0))

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


    # Scroll functionality
    # if seconds_elapsed%10 == 0:
    #     scrollable = display.horizontal_scroll(image)
    #     display.animate(scrollable)
