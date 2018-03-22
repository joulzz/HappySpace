import dlib
import numpy as np

class Tracker:
    def __init__(self):
        self.tracker = dlib.correlation_tracker()
        print "Correlation tracker created"

    def initialize_tracker(self, frame, bbox):
        self.tracker.start_track(frame, dlib.rectangle(bbox[0][0] - 10, bbox[0][1] - 20, bbox[1][0] + 10, bbox[1][1] + 20))

    def tracker_run(self, frame):
        quality = self.tracker.update(frame)
        bbox = self.tracker.get_position()
        # if quality < 7:
        #     self.tracked_bbox = []
        # else:
        self.tracked_bbox = [(int(bbox.left()), int(bbox.top())), (int(bbox.left() + bbox.width()), int(bbox.top() + bbox.height()))]