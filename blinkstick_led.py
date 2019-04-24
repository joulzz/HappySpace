from blinkstick import blinkstick

def led_blink(color):
    led = blinkstick.find_first()
    led.set_mode(3)
    # led.blink(name=color,delay=10)
    led.morph(name="random",duration=10000)
