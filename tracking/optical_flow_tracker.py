import cv2
import numpy as np

class Tracker:
    def __init__(self):
        self.feature_params = dict(maxCorners=500,
                              qualityLevel=0.3,
                              minDistance=7,
                              blockSize=7,)

        # Parameters for lucas kanade optical flow
        self.lk_params = dict(winSize=(30, 30),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


    def initialize_tracker(self, frame):
        self.gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # self.orb_feature_detector(self.gray_frame)
        self.shi_tomasu_feature_detector(self.gray_frame)

    def harris_corner(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        features = cv2.cornerHarris(frame, 2, 3, 0.04)
        print features

    def shi_tomasu_feature_detector(self, frame):
        self.features1 = cv2.goodFeaturesToTrack(frame, mask = None, useHarrisDetector=True, **self.feature_params)

    def orb_feature_detector(self, frame):
        orb = cv2.ORB_create()
        # find the keypoints with ORB
        kp = orb.detect(frame, None)
        kp, des = orb.compute(frame, kp)
        self.features1 = np.array([[[p.pt[0], p.pt[1]]] for p in kp]).astype(np.float32)


    def run_tracker(self, frame):
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        features_new, st, err = cv2.calcOpticalFlowPyrLK(self.gray_frame, current_gray, self.features1, None, **self.lk_params)
        self.good_new_features = features_new[st==1]
        self.good_old_features = self.features1[st==1]
        self.harris_corner(frame)
        self.shi_tomasu_feature_detector(current_gray)
        # self.orb_feature_detector(current_gray)
        print self.features1.shape
        self.gray_frame = current_gray


