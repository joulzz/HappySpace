import cv2
import numpy as np

class Tracker:
    def __init__(self):
        self.feature_params = dict(maxCorners=100,
                              qualityLevel=0.3,
                              minDistance=7,
                              blockSize=7)

        # Parameters for lucas kanade optical flow
        self.lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


    def initialize_tracker(self, frame):
        self.gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.features1 = cv2.goodFeaturesToTrack(self.gray_frame, mask = None, **self.feature_params)


    def run_tracker(self, frame):
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        features_new, st, err = cv2.calcOpticalFlowPyrLK(self.gray_frame, current_gray, self.features1, None, **self.lk_params)
        self.good_new_features = features_new[st==1]
        self.good_old_features = self.features1[st==1]

        self.features1 = features_new
        self.gray_frame = current_gray


