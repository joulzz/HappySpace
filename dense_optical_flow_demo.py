from person_detector.ssd_person_detection import PersonDetector
from tracking.dense_optical_flow_tracker import Tracker
import cv2
import numpy as np
from face_detector.face_detector import FaceDetection
from smile_counter.people_counter import PeopleCounter
from smile_counter.smile_counter import SmileCounter
from sentiment_net.sentiment_net import SmileDetector

def person_counter():
    persondetect = PersonDetector("Models/ssd/ssd.pb", "Models/ssd/ssd.pbtxt", "ssd")
    people_counter = PeopleCounter()
    face_detector = FaceDetection("Models/haarcascade_frontalface_default.xml")
    smile_detector = SmileDetector()
    smile_counter = SmileCounter()
    tracker = Tracker()
    cap = cv2.VideoCapture(1)

    tracking_set = True
    # k = 0
    while True:
        _, frame = cap.read()
        draw_frame = np.copy(frame)
        persondetect.detect_person(frame)
        detected_bboxes = persondetect.person_bounding_boxes
        if len(people_counter.total_detected_bboxes) == 0:
            tracker.initialize_tracker(frame)
        for bbox in detected_bboxes:
            people_counter.add_to_total(bbox)
        tracker.run_tracker(frame)
        print tracker.flow.shape

        # displacement_matrix =
        cv2.imshow('frame', draw_frame)
        cv2.waitKey(5)

if __name__ == "__main__":
    person_counter()