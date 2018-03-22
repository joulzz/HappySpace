import cv2
import numpy as np

class Tracker:
    def __init__(self):
        self.tracked_bbox = []
        self.track_window = None
        self.track_box = None
        self.hist = None
        self.hsv = None
        self.mask = None

    def get_frame(self, frame):
        self.hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        self.mask = cv2.inRange(self.hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

    def initialize_tracker(self, frame, bbox):
        self.get_frame(frame)
        x0, y0, x1, y1 = bbox[0][0], bbox[1][1], bbox[1][0], bbox[0][1]
        self.track_window = (x0, y0, x1 - x0, y1 - y0)
        hsv_roi = self.hsv[y0:y1, x0:x1]
        mask_roi = self.mask[y0:y1, x0:x1]
        hist = cv2.calcHist([hsv_roi], [0], mask_roi, [16], [0, 180])
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        self.hist = hist.reshape(-1)

    def tracker_run(self, frame):
        self.tracked_bbox = []
        self.get_frame(frame)
        prob = cv2.calcBackProject([self.hsv], [0], self.hist, [0, 180], 1)
        prob &= self.mask
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        self.track_box, self.track_window = cv2.CamShift(prob, self.track_window, term_crit)
        rotated_rect = cv2.boxPoints(self.track_box)
        x_points = []
        y_points = []
        for rect in rotated_rect:
            x_points.append(rect[0])
            y_points.append(rect[1])

        if len(x_points) != 0 and len(y_points) != 0:
            x0 = min(x_points)
            y0 = max(y_points)
            x1 = max(x_points)
            y1 = min(y_points)
            self.tracked_bbox = ((x0, y0), (x1, y1))

            if x0 == 0 and x1 == 0 and y0 == 0 and y1 == 0:
                self.tracked_bbox = []
        else:
            self.tracked_bbox = []
