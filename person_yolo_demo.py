from person_detector.yolo_person_detection  import PersonDetector
from tracking.iou_tracking import Tracker
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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    tracker = Tracker()
    previous_frame = []
    current_frame = None
    frame_count = 0
    cv2.namedWindow('frame', cv2.WINDOW_FREERATIO)
    _, frame = cap.read()

    # Select Area to track
    roi = cv2.selectROI('frame', frame)
    while True:
        _, frame = cap.read()
        original = np.copy(frame)
        frame = frame[roi[1]: roi[3], roi[0]: roi[2]]
        t0 = cv2.getTickCount()
        # Set previous frame at the start
        if len(previous_frame) == 0:
            previous_frame = np.copy(frame)
        current_frame = frame

        # Set frame for drawing purposes
        draw_frame = np.copy(frame)

        # Initialize person detection
        persondetect.detect_person(current_frame)
        people_tracker.current_frame_bboxes = persondetect.person_bounding_boxes

        print people_tracker.current_frame_bboxes
        state = []
        bboxes = []

        # if frame_count % 3 == 0:
        for person in person_counter.people:
            if person.current:
                person.current = False
                previous_bbox = person.bbox

                bbox_overlaps = []
                for current_bbox in people_tracker.current_frame_bboxes:
                    overlap = tracker.iou_tracker(previous_bbox, current_bbox)
                    bbox_overlaps.append(overlap)

                if len(bbox_overlaps) != 0:
                    if max(bbox_overlaps) > 0.6:
                        person.history.append(person.bbox)
                        person.bbox = people_tracker.current_frame_bboxes[bbox_overlaps.index(max(bbox_overlaps))]
                        person.current = True
                        people_tracker.current_frame_bboxes.remove(person.bbox)

        for bbox in people_tracker.current_frame_bboxes:
            max_idx = len(person_counter.people)
            new_person = People()
            new_person.bbox = bbox
            new_person.current = True
            new_person.id = max_idx
            person_counter.add(new_person)

        if frame_count % 2 == 0:
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
                cv2.putText(draw_frame, "ID: {0}".format(person.id), person.bbox[0], cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 5)
                cv2.putText(draw_frame, "Smiles: {0}".format(person.count), (person.bbox[0][0], person.bbox[1][1]), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
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

        original[roi[1]: roi[3], roi[0]: roi[2]] = draw_frame
        cv2.imshow('frame', original)
        cv2.waitKey(2)
        inf_time = (cv2.getTickCount() - t0)/ cv2.getTickFrequency()
        print "Inference time: {0} ms \n FPS: {1}".format(inf_time * 1000, 1/ inf_time)


if __name__ == "__main__":
    main()