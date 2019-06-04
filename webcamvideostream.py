import cv2


class WebCamVideoStream:
    def __init__(self, src=0, resolution=(640, 480),framerate=32):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.stream.set(cv2.CAP_PROP_FPS, framerate)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        self.update
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
