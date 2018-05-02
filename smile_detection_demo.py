from tracking.iou_tracking import Tracker
import cv2
import numpy as np
import sys
from face_detector.face_detector import FaceDetection
from smile_counter.people_counter import PeopleTracker, PeopleCounter, People
from smile_counter.smile_counter import SmileCounter
from sentiment_net.sentiment_net import SmileDetector
import pandas as pd
from skvideo.io import LibAVWriter

def main():
    time_elapsed = 0
    people_tracker = PeopleTracker()
    person_counter = PeopleCounter()
    face_detector = FaceDetection("Models/haarcascade_frontalface_default.xml")
    smile_detector = SmileDetector()
    smile_counter = SmileCounter()
    cap = cv2.VideoCapture(0)
    writer = LibAVWriter("output.mp4")
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
        total_smile_counter = 0
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

        # Initialize face detection
        face_detector.run_facedetector(current_frame)
        people_tracker.current_frame_bboxes = face_detector.faces

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
            new_person.timestamp = frame_count
            person_counter.add(new_person)



        # if frame_count % 5 ==0:


        for people in person_counter.people:
            print "{0}, {1}, {2}".format(people.bbox, people.current, people.count)
            if people.current:
                face = people.bbox
                smile_detector.preprocess_image(current_frame[face[0][1]: face[1][1], face[0][0]: face[1][0]])
                if smile_detector.predict():
                    people.count += 1

        for person in person_counter.people:
            total_smile_counter += person.count
            state.append(person.current)
            bboxes.append(person.bbox)
            if person.current:
                people_tracker.previous_frame_bboxes.append(person.bbox)
                cv2.rectangle(draw_frame, person.bbox[0], person.bbox[1], (255, 255, 255), 3)
                cv2.putText(draw_frame, "ID: {0}".format(person.id), person.bbox[0], cv2.FONT_HERSHEY_TRIPLEX, 0.75, (0, 255, 0), 2)
                cv2.putText(draw_frame, "SMILES: {0}".format(person.count), (person.bbox[0][0], person.bbox[1][1]), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (0, 255, 0), 2)


        if time_elapsed % 3600 == 0:
            df = pd.DataFrame()
            ids = []
            smile_count = []
            last_bbox = []
            location_history = []
            timestamp = []
            for people in person_counter.people:
                people.history.append(people.bbox)
                ids.append(people.id)
                smile_count.append(people.count)
                last_bbox.append(people.bbox)
                location_history.append(people.history)
                timestamp.append(people.timestamp)

            df["ID"] = ids
            df["Smiles_Detected"] = smile_count
            df["Last_Location"] = last_bbox
            df["Location_History"] =location_history
            df["Timestamp"] = timestamp

            df.to_csv("output.csv", index=False)
        frame_count += 1

        original[roi[1]: roi[3], roi[0]: roi[2]] = draw_frame
        cv2.putText(original, "Total Smiles: {0}".format(total_smile_counter), (0, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        cv2.imshow('frame', original)
        writer_image = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        writer.writeFrame(writer_image)
        ch = 0xFF & cv2.waitKey(2)
        if ch == 27:
            break
        inf_time = (cv2.getTickCount() - t0)/ cv2.getTickFrequency()
        print "Inference time: {0} ms \n FPS: {1}".format(inf_time * 1000, 1/ inf_time)
        time_elapsed += inf_time

    writer.close()
if __name__ == "__main__":
    main()