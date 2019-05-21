from picamvideostream import PiCamVideoStream
from webcamvideostream import WebCamVideoStream

class VideoStream:
    def __init__(self, src=0, usePiCamera=False, resolution=(640, 480),framerate=32):
        if usePiCamera:
            self.stream = PiCamVideoStream(resolution=resolution,framerate=framerate)
        else:
            self.stream = WebCamVideoStream(src=src)

    def read(self):
        return self.stream.read()

