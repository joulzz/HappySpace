from picamera.array import PiRGBArray
from picamera import PiCamera
from time import sleep
import cv2


class PiCamVideoStream:
    def __init__(self, resolution=(640, 480), framerate=32):
        self.camera = PiCamera()
        self.camera.resolution = resolution
        self.camera.framerate = framerate
        self.rawCapture = PiRGBArray(self.camera, size=resolution)
        self.stream = self.camera.capture_continuous(self.rawCapture, format="bgr", use_video_port=True)
        self.stopped = False
        sleep(2)

        for f in self.stream:
            self.frame = f.array
            self.rawCapture.truncate(0)
            break

    def start(self):
        self.update()
        return self

    def update(self):
        while True:
            for f in self.stream:
                self.frame = f.array
                self.rawCapture.truncate(0)

                if self.stopped:
                    self.stream.close()
                    self.rawCapture.close()
                    self.camera.close()
                    return

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

