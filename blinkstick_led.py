from blinkstick import blinkstick


def main():
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