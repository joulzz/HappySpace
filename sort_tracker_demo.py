from person_detector.ssd_person_detection import PersonDetector
import cv2
import numpy as np
from face_detector.face_detector import FaceDetection
from smile_counter.people_counter import PeopleCounter
from smile_counter.smile_counter import SmileCounter
from sentiment_net.sentiment_net import SmileDetector
from tracking.sort_tracker import Sort

def person_counter():
    persondetect = PersonDetector("Models/ssd/ssd.pb", "Models/ssd/ssd.pbtxt", "ssd")
    people_counter = PeopleCounter()
    face_detector = FaceDetection("Models/haarcascade_frontalface_default.xml")
    smile_detector = SmileDetector()
    smile_counter = SmileCounter()

    cap = cv2.VideoCapture("/home/suraj/Downloads/videoplayback (2).webm")
    tracker = Sort()
    tracking_set = True
    k = 0
    while True:
        _, frame = cap.read()
        # if k < 500:
        #     k += 1
        #
        #     continue
        draw_frame = np.copy(frame)
        persondetect.detect_person(frame)
        detected_bboxes = persondetect.person_bounding_boxes
        det_array = []
        for bbox in detected_bboxes:
            det_array.append([float(bbox[0][0]), float(bbox[0][1]), float(bbox[1][0]), float(bbox[1][1]), 0.5])
            cv2.rectangle(draw_frame, bbox[0], bbox[1], (255, 0, 0), 2)

        numpy_det_array = np.array(det_array)
        print det_array
        tracked_array = tracker.update(numpy_det_array)
        print tracked_array
        cv2.imshow('frame', draw_frame)
        cv2.waitKey(10)


if __name__ == "__main__":
    person_counter()