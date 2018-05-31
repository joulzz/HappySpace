from person_detector.yolo_person_detection  import PersonDetector
from tracking.iou_tracking import Tracker
import cv2
from time import gmtime, strftime

import numpy as np
import sys
from face_detector.face_detector import FaceDetection
from smile_counter.people_counter import PeopleTracker, PeopleCounter, People
from smile_counter.smile_counter import SmileCounter
from sentiment_net.sentiment_net import SmileDetector
import pandas as pd
from skvideo.io import FFmpegWriter
from configuration_module.json_parser import json_parser
import os
import boto3

def main():
    s3 = boto3.resource('s3')

    dir_path = os.path.dirname(os.path.abspath(__file__))

    if len(sys.argv) != 2:
        print(
            "\n Give path to the JSON Configuration File\n Example: python smile_detection_demo.py <full path to json file>")
        return

    tinkerboard_id, skip_frame, display_flag, write_video, remote_upload, running_time, min_face, max_face = json_parser(
        sys.argv[1])

    persondetect = PersonDetector(os.path.realpath(os.path.join(dir_path, "Models/tiny_yolo/yolov2-tiny.weights")), os.path.realpath(os.path.join(dir_path, "Models/tiny_yolo/yolov2-tiny.cfg")), "yolo", 416)
    people_tracker = PeopleTracker()
    person_counter = PeopleCounter()
    face_detector = FaceDetection("Models/haarcascade_frontalface_default.xml")
    smile_detector = SmileDetector()
    smile_counter = SmileCounter()
    start_time = int(strftime("%H%M", gmtime()))

    cap = cv2.VideoCapture(0)
    if write_video:
        writer = FFmpegWriter("output.mp4")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    tracker = Tracker()
    previous_frame = []
    current_frame = None
    frame_count = 0
    if display_flag:
        cv2.namedWindow('frame', cv2.WINDOW_FREERATIO)
    _, frame = cap.read()

    # Select Area to track
    while True:
        current_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        time_elapsed = int(strftime("%H%M", gmtime()))

        total_smile_counter = 0
        _, frame = cap.read()
        original = np.copy(frame)
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
            new_person.timestamp = frame_count
            person_counter.add(new_person)

        if frame_count % (skip_frame+1) == 0:
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
            total_smile_counter += person.count
            state.append(person.current)
            bboxes.append(person.bbox)
            if person.current:
                people_tracker.previous_frame_bboxes.append(person.bbox)
                cv2.rectangle(draw_frame, person.bbox[0], person.bbox[1], (255, 255, 255), 3)
                cv2.putText(draw_frame, "ID: {0}".format(person.id), person.bbox[0], cv2.FONT_HERSHEY_TRIPLEX, 0.75, (0, 255, 0), 2)
                cv2.putText(draw_frame, "SMILES: {0}".format(person.count), (person.bbox[0][0], person.bbox[1][1]), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (0, 255, 0), 2)
        print state
        print bboxes


        if int((time_elapsed - start_time) / 100) > running_time or (time_elapsed - start_time) == -24 + running_time:
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
            if remote_upload:
                data = open(os.path.join(dir_path, 'output.csv'), 'rb')
                s3.Bucket('smile-log').put_object(
                    Key='{0}/{1}.csv'.format(tinkerboard_id, strftime("%Y-%m-%d", gmtime())), Body=data)
            break
            break
        frame_count += 1

        original = draw_frame
        cv2.putText(original, "Total Smiles: {0}".format(total_smile_counter), (0, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        if display_flag:
            cv2.imshow('frame', original)
            ch = 0xFF & cv2.waitKey(2)
            if ch == 27:
                break
        if write_video:
            writer_image = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            writer.writeFrame(writer_image)

        inf_time = (cv2.getTickCount() - t0)/ cv2.getTickFrequency()
        print "Inference time: {0} ms \n FPS: {1}".format(inf_time * 1000, 1/ inf_time)

    writer.close()
if __name__ == "__main__":
    main()