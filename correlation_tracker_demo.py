from person_detector.ssd_person_detection import PersonDetector
from tracking.correlation_tracker import Tracker
import cv2
import numpy as np
from face_detector.face_detector import FaceDetection
from smile_counter.people_counter import PeopleTracker, PeopleCounter, People
from smile_counter.smile_counter import SmileCounter
from sentiment_net.sentiment_net import SmileDetector
import pandas as pd

def main():
    persondetect = PersonDetector("Models/ssd/ssd.pb", "Models/ssd/ssd.pbtxt", "ssd")
    people_tracker = PeopleTracker()
    person_counter = PeopleCounter()
    face_detector = FaceDetection("Models/haarcascade_frontalface_default.xml")
    smile_detector = SmileDetector()
    smile_counter = SmileCounter()
    cap = cv2.VideoCapture(0)
    tracker = Tracker()
    previous_frame = []
    current_frame = None
    frame_count = 0
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
        face_detector.run_facedetector(current_frame)
        # persondetect.detect_person(current_frame)
        people_tracker.current_frame_bboxes = face_detector.faces

        # for faces in people_tracker.current_frame_bboxes:
        #     cv2.rectangle(draw_frame, faces[0], faces[1], (0, 0, 255), 2)
        # cv2.imshow('frame', draw_frame)
        # cv2.waitKey()
        print people_tracker.current_frame_bboxes
        state = []
        bboxes = []
        # Draw current frame persons
        for person in person_counter.people:
            state.append(person.current)
            bboxes.append(person.bbox)
            if person.current:
                cv2.rectangle(draw_frame, person.bbox[0], person.bbox[1], (255, 0, 0), 2)
                cv2.putText(draw_frame, "Person ID: {0}".format(person.id), person.bbox[0], cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
                cv2.putText(draw_frame, "Smiles Count: {0}".format(person.count), (person.bbox[0][0], person.bbox[1][1]), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

        print state
        print bboxes

        for person in person_counter.people:
            if person.current:
                person.current = False
                tracker.initialize_tracker(previous_frame, person.bbox)
                tracker.tracker_run(current_frame)
                tracked_bbox = tracker.tracked_bbox

                cv2.rectangle(draw_frame, tracked_bbox[0], tracked_bbox[1], (0, 255, 0), 2)
                center = (tracked_bbox[0][0] + tracked_bbox[1][0]) / 2, (tracked_bbox[0][1] + tracked_bbox[1][1]) / 2

                for current_bbox in people_tracker.current_frame_bboxes:
                    if center[0] >= current_bbox[0][0] and center[0] <= current_bbox[1][0] and center[1] >= current_bbox[0][1] and center[1] <= current_bbox[1][1]:
                        person.bbox = current_bbox
                        person.current = True
                        people_tracker.current_frame_bboxes.remove(current_bbox)

        max_idx = len(person_counter.people)
        for bbox in people_tracker.current_frame_bboxes:
            new_person = People()
            new_person.bbox = bbox
            new_person.count = 0
            new_person.current = True
            new_person.id = max_idx
            person_counter.add(new_person)

        if frame_count % 10 == 0:
            for people in person_counter.people:
                if people.current:
                    bbox = people.bbox
                    smile_detector.preprocess_image(current_frame[bbox[0][1]: bbox[1][1], bbox[0][0]: bbox[1][0]])
                    if smile_detector.predict():
                        people.count += 1


        if frame_count % 1000 == 0:
            df = pd.DataFrame()
            ids = []
            smile_count = []
            last_bbox = []

            for people in person_counter.people:
                ids.append(people.id)
                smile_count.append(people.count)
                last_bbox.append(people.bbox)
            df["ID"] = ids
            df["Smiles_Detected"] = smile_count
            df["Last_Location"] = last_bbox

            df.to_csv("output.csv", index=False)
        frame_count += 1

        # print people_tracker.previous_frame_bboxes
        # print people_tracker.current_frame_bboxes
        # for previous_bbox in people_tracker.previous_frame_bboxes:
        #     for current_bbox in people_tracker.current_frame_bboxes:
        #         tracker.initialize_tracker(previous_frame, previous_bbox)
        #         tracker.tracker_run(current_frame)
        #         tracked_bbox = tracker.tracked_bbox
        #         cv2.rectangle(draw_frame, tracked_bbox[0], tracked_bbox[1], (0, 0, 255), 2)
        #         if len(tracked_bbox) == 0:
        #             people_tracker.remove_from_previous(previous_bbox)
        #         center = (tracked_bbox[0][0] + tracked_bbox[1][0])/2, (tracked_bbox[0][1] + tracked_bbox[1][1])/2
        #
        #         if center[0] > current_bbox[0][0] and center[0] < current_bbox[1][0] and center[1] > current_bbox[1][1] and center[1] < current_bbox[0][1]:
        #             people_tracker.replace_previous_frame(previous_bbox, current_bbox)
        #             for features in person_counter.people_features:
        #                 if features.current:
        #                     if center[0] > features.bbox[0][0] and center[0] < features.bbox[1][0] and \
        #                                     center[1] > features.bbox[1][1] and center[1] < features.bbox[0][1]:
        #                         features.count += 1
        #                         features.current = True
        #                         features.bbox = current_bbox
        #             people_tracker.current_frame_bboxes.remove(current_bbox)
        #
        #
        # for i, previous_bbox in enumerate(people_tracker.previous_frame_bboxes):
        #     if i not in people_tracker.replaced_indices:
        #         people_tracker.previous_frame_bboxes.remove(previous_bbox)
        #         center = (previous_bbox[0][0] + previous_bbox[1][0])/2, (previous_bbox[1][0] + previous_bbox[1][1])/2
        #         for features in person_counter.people_features:
        #             if features.current:
        #                 if center[0] > features.bbox[0][0] and center[0] < features.bbox[1][0] and \
        #                                 center[1] > features.bbox[1][1] and center[1] < features.bbox[0][1]:
        #                     features.current = False
        #
        #
        # for bbox in people_tracker.current_frame_bboxes:
        #     people_tracker.add_to_previous(bbox)
        #     idx = len(person_counter.people_features)
        #
        #     features = PeopleFeatures()
        #     features.id = idx
        #     features.bbox = bbox
        #     features.count = 1
        #     features.current = True
        #     person_counter.add(features)

        cv2.imshow('frame', draw_frame)
        cv2.waitKey(5)


if __name__ == "__main__":
    main()