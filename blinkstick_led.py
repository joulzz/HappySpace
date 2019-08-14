from blinkstick import blinkstick


def main():

    """ Description

    Script used to isolate the Blinkstick module from the main functionality. Function used to find all blinkstick leds
    and morph into random colors based on the duration value for morphing in milliseconds.

    :except: KeyboardInterrupt, SystemExit: Catches abrupt departure from script to switch off all leds

    Blinkstick Github Repo: https://github.com/arvydas/blinkstick-python/blob/master/blinkstick/blinkstick.py
    """
    try:
        try:
            print("Running Blinkstick")
            for led in blinkstick.find_all():
                led.set_mode(3)
        except AttributeError:
            return None
        while True:
            brightness = 20
            r, g, b = led._hex_to_rgb("#25FFFF")
            r = (brightness / 100.0 * r)
            g = (brightness / 100.0 * g)
            b = (brightness / 100.0 * b)
            led.set_color(red=r, green=g, blue=b)
    except (KeyboardInterrupt, SystemExit):
        print("Switching off Blinkstick")
        for led in blinkstick.find_all():
            led.turn_off()

if __name__ == "__main__":
    main()