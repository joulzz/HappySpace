import cv2
import numpy as np

class Tracker:
    def __init__(self):
        self.tracker = cv2.MultiTracker_create()
        print "MultiTracker created"
    def initialize_tracker(self, frame, bboxes):
        for bbox in bboxes:
            bbox = (bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1])
            ret = self.tracker.add(cv2.TrackerKCF_create(), frame, bbox)

    def run_tracker(self, frame):
        self.tracked_boxes = []
        retval, bboxes = self.tracker.update(frame)
        for bbox in bboxes:
            self.tracked_boxes.append([(int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[1]), int(bbox[1] + bbox[2]))])