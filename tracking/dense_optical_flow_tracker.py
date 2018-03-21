import cv2

class Tracker:
    def __init__(self):
        pass

    def initialize_tracker(self, frame):
        self.previous = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def run_tracker(self, frame):
        self.current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.flow = cv2.calcOpticalFlowFarneback(self.previous, self.current, None, 0.5, 3, 15, 3, 10, 1.2, 0)
        self.previous = self.current


