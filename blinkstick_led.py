from blinkstick import blinkstick


def main():
    try:
        led = blinkstick.find_first()
        led.set_mode(3)
        while True:
            led.morph(name="random",duration=10000)
    except (KeyboardInterrupt, SystemExit):
        for led in blinkstick.find_all():
            led.turn_off()

if __name__ == "__main__":
    main()