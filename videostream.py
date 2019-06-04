from webcamvideostream import WebCamVideoStream


class VideoStream:
    def __init__(self, src=0, usePiCamera=False, resolution=(640, 480),framerate=32):
        if usePiCamera:
            from picamvideostream import PiCamVideoStream

            self.stream = PiCamVideoStream(resolution=resolution,framerate=framerate)
        else:
            self.stream = WebCamVideoStream(src=src,resolution=resolution,framerate=framerate)

    def start(self):
        return self.stream.start()

    def read(self):
        return self.stream.read()

    def update(self):
        return self.stream.update()

    def stop(self):
        return self.stream.stop()

