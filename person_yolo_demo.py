from person_detector.yolo_person_detection  import PersonDetector
from tracking.correlation_tracker import Tracker
import cv2
import numpy as np
from face_detector.face_detector import FaceDetection
from smile_counter.people_counter import PeopleTracker, PeopleCounter, People
from smile_counter.smile_counter import SmileCounter
from sentiment_net.sentiment_net import SmileDetector
import pandas as pd

def main():
    persondetect = PersonDetector("/home/suraj/Downloads/yolov2.weights", "/home/suraj/Downloads/yolov2.cfg", "yolo")
    people_tracker = PeopleTracker()
    person_counter = PeopleCounter()
    face_detector = FaceDetection("Models/haarcascade_frontalface_default.xml")
    smile_detector = SmileDetector()
    smile_counter = SmileCounter()
    cap = cv2.VideoCapture("/home/suraj/Downloads/control_video.mov")
    tracker = Tracker()
    previous_frame = []
    current_frame = None
    frame_count = 0
    while True:
        _, frame = cap.read()
        t0 = cv2.getTickCount()
        # Set previous frame at the start
        if len(previous_frame) == 0:
            previous_frame = np.copy(frame)
        current_frame = frame
        # print current_frame.shape, previous_frame.shape
        # Set frame for drawing purposes
        draw_frame = np.copy(frame)

        # Initialize person detection
        # face_detector.run_facedetector(current_frame)
        persondetect.detect_person(current_frame)
        people_tracker.current_frame_bboxes = persondetect.person_bounding_boxes

        # for faces in people_tracker.current_frame_bboxes:
        #     cv2.rectangle(draw_frame, faces[0], faces[1], (0, 0, 255), 2)
        # cv2.imshow('frame', draw_frame)
        # cv2.waitKey()
        print people_tracker.current_frame_bboxes
        state = []
        bboxes = []

        # if frame_count % 3 == 0:
        for person in person_counter.people:
            if person.current:
                person.current = False
                previous_bbox = person.bbox

                center = (previous_bbox[0][0] + previous_bbox[1][0]) / 2, (previous_bbox[0][1] + previous_bbox[1][1]) / 2

                for current_bbox in people_tracker.current_frame_bboxes:

                    if (center[0] >= current_bbox[0][0] and center[0] <= current_bbox[1][0] and center[1] >= current_bbox[0][1] and center[1] <= current_bbox[1][1]):
                        person.bbox = current_bbox
                        person.current = True
                        people_tracker.current_frame_bboxes.remove(current_bbox)
                        break

        for bbox in people_tracker.current_frame_bboxes:
            max_idx = len(person_counter.people)
            new_person = People()
            new_person.bbox = bbox
            new_person.current = True
            new_person.id = max_idx
            person_counter.add(new_person)
        # else:
        #     for person in person_counter.people:
        #         if person.current:
        #             tracker.initialize_tracker(previous_frame, person.bbox)
        #             tracker.tracker_run(current_frame)
        #             tracked_bbox = tracker.tracked_bbox
        #             person.bbox = tracked_bbox



        if frame_count % 4 == 0:
            for people in person_counter.people:
                if people.current:
                    bbox = people.bbox
                    face_detector.run_facedetector(current_frame)
                    faces = face_detector.faces
                    for face in faces:
                        center_face = (face[0][0] + face[1][0]) / 2, (face[0][1] + face[1][1]) / 2
                        if center_face[0] >= bbox[0][0] and center_face[0] <= bbox[1][0] and center_face[1] >= bbox[0][1] and center_face[1] <= bbox[1][1]:
                            smile_detector.preprocess_image(current_frame[face[0][1]: face[1][1], face[0][0]: face[1][0]])

                            if smile_detector.predict():
                                people.count += 1

        for person in person_counter.people:
            state.append(person.current)
            bboxes.append(person.bbox)
            if person.current:
                people_tracker.previous_frame_bboxes.append(person.bbox)
                cv2.rectangle(draw_frame, person.bbox[0], person.bbox[1], (255, 0, 0), 2)
                cv2.putText(draw_frame, "{0}".format(person.id), person.bbox[0], cv2.FONT_HERSHEY_PLAIN, 8, (255, 255, 255), 5)
                cv2.putText(draw_frame, "Smiles Count: {0}".format(person.count), (person.bbox[0][0], person.bbox[1][1]), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
        print state
        print bboxes


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
        cv2.namedWindow('frame', cv2.WINDOW_FREERATIO)

        cv2.imshow('frame', draw_frame)
        cv2.waitKey(2)
        inf_time = (cv2.getTickCount() - t0)/ cv2.getTickFrequency()
        print "Inference time: {0} ms \n FPS: {1}".format(inf_time * 1000, 1/ inf_time)


if __name__ == "__main__":
    main()