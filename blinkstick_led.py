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
            for led in blinkstick.find_all():
                led.set_mode(3)
        except AttributeError:
            return None
        while True:
            led.morph(name="random",duration=10000)
    except (KeyboardInterrupt, SystemExit):
        for led in blinkstick.find_all():
            led.turn_off()

if __name__ == "__main__":
    main()