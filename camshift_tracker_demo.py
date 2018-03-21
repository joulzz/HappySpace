from person_detector.ssd_person_detection import PersonDetector
from tracking.camshift_tracker import Tracker
import cv2
import numpy as np
from face_detector.face_detector import FaceDetection
from smile_counter.people_counter import PeopleTracker
from smile_counter.smile_counter import SmileCounter
from sentiment_net.sentiment_net import SmileDetector

def person_counter():
    persondetect = PersonDetector("Models/ssd/ssd.pb", "Models/ssd/ssd.pbtxt", "ssd")
    people_tracker = PeopleTracker()
    face_detector = FaceDetection("Models/haarcascade_frontalface_default.xml")
    smile_detector = SmileDetector()
    smile_counter = SmileCounter()
    cap = cv2.VideoCapture("/home/suraj/Videos/Webcam/2018-03-17-142827.webm")
    tracker = Tracker()
    tracking_set = True
    previous_frame = []
    current_frame = None
    total_id_list = []
    current_frame_person_ids = []
    removed_indices = []

    while True:
        _, frame = cap.read()

        # Set previous frame at the start
        if len(previous_frame) == 0:
            previous_frame = np.copy(frame)
        current_frame = frame
        # print current_frame.shape, previous_frame.shape
        # Set frame for drawing purposes
        draw_frame = np.copy(frame)

        # Initialize person detection
        persondetect.detect_person(current_frame)
        people_tracker.current_frame_bboxes = persondetect.person_bounding_boxes

        # for draw_bbox in people_tracker.previous_frame_bboxes:
        #     cv2.rectangle(draw_frame, draw_bbox[0], draw_bbox[1], (0, 0, 255), 2)


        # print people_tracker.previous_frame_bboxes
        # print people_tracker.current_frame_bboxes
        for previous_bbox in people_tracker.previous_frame_bboxes:
            for current_bbox in people_tracker.current_frame_bboxes:
                tracker.initialize_tracker(previous_frame, previous_bbox)
                tracker.tracker_run(current_frame)
                tracked_bbox = tracker.tracked_bbox
                if len(tracked_bbox) == 0:
                    people_tracker.remove_from_previous(previous_bbox)
                center = (tracked_bbox[0][0] + tracked_bbox[1][0])/2, (tracked_bbox[0][1] + tracked_bbox[1][1])/2

                if center[0] > current_bbox[0][0] and center[0] < current_bbox[1][0] and center[1] > current_bbox[1][1] and center[1] < current_bbox[0][1]:
                    people_tracker.replace_previous_frame(previous_bbox, current_bbox)
                    people_tracker.current_frame_bboxes.remove(current_bbox)


        for i, previous_bbox in enumerate(people_tracker.previous_frame_bboxes):
            if i not in people_tracker.replaced_indices:
                people_tracker.previous_frame_bboxes.remove(previous_bbox)

        for bbox in people_tracker.current_frame_bboxes:
            # print "Added new person"
            people_tracker.add_to_previous(bbox)
            if len(current_frame_person_ids) != 0:
                total_id_list.append(max(current_frame_person_ids) + 1)
            else:
                total_id_list.append(1)

        print current_frame_person_ids
        cv2.imshow('frame', draw_frame)
        cv2.waitKey(5)


if __name__ == "__main__":
    person_counter()