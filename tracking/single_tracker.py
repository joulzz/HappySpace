import cv2

class KCFFilter:
    def __init__(self):
        self.tracker = cv2.KCFTracker_create()
        