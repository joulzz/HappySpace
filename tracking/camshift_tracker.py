import cv2
import numpy as np

class Tracker:
    def __init__(self):
        self.tracked_bboxes = []
        self.hist_array = []


    def get_frame(self, frame):
        self.hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        self.mask = cv2.inRange(self.hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

    def initialize_tracker(self, frame, bboxes):
        self.bboxes = bboxes
        self.get_frame(frame)
        for bbox in bboxes:
            print "New person tracking now"
            x0, y0, x1, y1 = bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][0]
            self.track_window = (x0, y0, x1 - x0, y1 - y0)
            hsv_roi = self.hsv[y0:y1, x0:x1]
            mask_roi = self.mask[y0:y1, x0:x1]
            hist = cv2.calcHist([hsv_roi], [0], mask_roi, [16], [0, 180])
            cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
            self.hist_array.append(hist.reshape(-1))

    def tracker_run(self, frame):
        self.get_frame(frame)
        self.tracked_bboxes = []
        for hist in self.hist_array:
            prob = cv2.calcBackProject([self.hsv], [0], hist, [0, 180], 1)
            prob &= self.mask
            term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
            self.track_box, self.track_window = cv2.CamShift(prob, self.track_window, term_crit)
            self.tracked_bboxes.append(self.track_box)

