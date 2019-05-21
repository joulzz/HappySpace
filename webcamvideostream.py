import cv2


class WebCamVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.update

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame