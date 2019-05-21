from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2


class PiCamVideoStream:
    def __init__(self, resolution=(640, 480), framerate=32):
        self.camera = PiCamera()
        self.camera.resolution = resolution
        self.camera.framerate = framerate
        self.rawCapture = PiRGBArray(self.camera, size=resolution)
        self.stream = self.camera.capture_continuous(self.rawCapture, format="bgr", use_video_port=True)
        self.frame = None
        self.update

    def update(self):
        for f in self.stream:
            self.frame = f.array
            self.rawCapture.truncate(0)

    def read(self):
        return self.frame

