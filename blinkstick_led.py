from blinkstick import blinkstick


def led_blink():
    try:
        led = blinkstick.find_first()
        led.set_mode(3)
        # led.blink(name=color,delay=10)
        while True:
            led.morph(name="random",duration=10000)
    except (KeyboardInterrupt, SystemExit):
        led.turn_off()
