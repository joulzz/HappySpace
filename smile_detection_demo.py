from tracking.iou_tracking import Tracker
import cv2
import numpy as np
from face_detector.face_detector import FaceDetection
from smile_counter.people_counter import PeopleTracker, PeopleCounter, People
from sentiment_net.sentiment_net import SmileDetector
import pandas as pd
from skvideo.io import FFmpegWriter
import gc
import subprocess
from configuration_module.json_parser import json_parser
from time import gmtime, strftime
import sys
import boto3
import os


def main():
    dir_path = os.path.dirname(os.path.abspath(__file__))

    if len(sys.argv)!= 2:
        print("\n Give path to the JSON Configuration File\n Example: python smile_detection_demo.py <full path to json file>")
        return


    # Read parameters from JSON file. Refer to word document for parameter functions
    tinkerboard_id, skip_frame, display_flag, write_video, remote_upload, running_time, min_face, max_face, write_images = json_parser(sys.argv[1])



    # Create instances of required class objects
    people_tracker = PeopleTracker()
    person_counter = PeopleCounter()
    face_detector = FaceDetection(os.path.join(dir_path, "Models/haarcascade_frontalface_default.xml"))
    smile_detector = SmileDetector()
    tracker = Tracker()
    s3 = boto3.resource('s3')

    if display_flag:
        cv2.namedWindow("frame", cv2.WINDOW_FREERATIO)

    cap = cv2.VideoCapture(0)
    if write_video:
        writer = FFmpegWriter(os.path.join(dir_path, "output.mp4"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    previous_frame = []
    frame_count = 0
    _, frame = cap.read()


    # Comment if running on local machine, swapoff swapon required for tinkerboard
    subprocess.check_output(["sudo", "swapoff","-a"])
    subprocess.check_output(["sudo", "swapon","-a"])


    start_time = int(strftime("%H%M", gmtime()))
    inference_time_sum = 0
    average_fps = 0
    while cap.isOpened():
        total_smile_counter = 0
        _, frame = cap.read()
        original = np.copy(frame)
        # frame = frame[roi[1]: roi[3], roi[0]: roi[2]]
        t0 = cv2.getTickCount()
        # Set previous frame at the start
        if len(previous_frame) == 0:
            previous_frame = np.copy(frame)
        current_frame = frame

        # Set frame for drawing purposes
        draw_frame = np.copy(frame)

        # Initialize face detection
        face_detector.run_facedetector(current_frame, min_face, max_face)
        people_tracker.current_frame_bboxes = face_detector.faces

        state = []
        bboxes = []
        current_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        # Here person_counter.people corresponds to previous frame people and people_tracker.current_frame_bboxes to people in current frame
        for person in person_counter.people:
            if person.current:
                person.current = False
                previous_bbox = person.bbox

                bbox_overlaps = []

                # Add overlaps between previous bboxes and current bboxes to an array
                for current_bbox in people_tracker.current_frame_bboxes:
                    overlap = tracker.iou_tracker(previous_bbox, current_bbox)
                    bbox_overlaps.append(overlap)

                if len(bbox_overlaps) != 0:
                    # If overlap is greater than 50%, replace previous bbox with current one
                    if max(bbox_overlaps) > 0.5:
                        person.history.append(person.bbox)
                        person.bbox = people_tracker.current_frame_bboxes[bbox_overlaps.index(max(bbox_overlaps))]
                        person.current = True
                        people_tracker.current_frame_bboxes.remove(person.bbox)


        # Add unreplaced bboxes
        for bbox in people_tracker.current_frame_bboxes:
            max_idx = len(person_counter.people)
            new_person = People()
            new_person.bbox = bbox
            new_person.current = True
            new_person.id = max_idx
            new_person.timestamp = current_time
            person_counter.add(new_person)

        # person_counter.people is now updated to correspond to people in the current frame

        # if frame_count % 5 ==0:
        if frame_count % (skip_frame+1) == 0:
            print "Sentiment Net Run"
            for people in person_counter.people:
                if people.current:
                    face = people.bbox
                    smile_detector.preprocess_image(current_frame[face[0][1]: face[1][1], face[0][0]: face[1][0]])
                    if smile_detector.predict():

                        # Check flag to then save images to the images folder in current directory
                        if write_images:
                            cv2.imwrite(
                            "{0}/{1}_{2}.jpg".format(os.path.join(dir_path, "images"), people.id, people.count),
                            current_frame[face[0][1]: face[1][1], face[0][0]: face[1][0]])

                        people.count += 1

        for person in person_counter.people:
            total_smile_counter += person.count
            # state.append(person.current)
            # bboxes.append(person.bbox)
            if person.current:
                people_tracker.previous_frame_bboxes.append(person.bbox)
                cv2.rectangle(draw_frame, person.bbox[0], person.bbox[1], (255, 255, 255), 3)
                cv2.putText(draw_frame, "ID: {0}".format(person.id), person.bbox[0], cv2.FONT_HERSHEY_TRIPLEX, 0.75, (0, 255, 0), 2)
                cv2.putText(draw_frame, "SMILES: {0}".format(person.count), (person.bbox[0][0], person.bbox[1][1]), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (0, 255, 0), 2)

        inf_time = (cv2.getTickCount() - t0)/ cv2.getTickFrequency()
        time_elapsed = int(strftime("%H%M", gmtime()))
        if int((time_elapsed - start_time) / 100) > running_time or (time_elapsed - start_time) == -24 + running_time:
            frame_count = 0

            # Write to CSV, Create different write parameters
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

            df.to_csv(os.path.join(dir_path, "output.csv"), index=False)
            print("Wrote to CSV")


            if remote_upload:
                data = open(os.path.join(dir_path, 'output.csv'), 'rb')
                s3.Bucket('smile-log').put_object(Key='{0}/{1}.csv'.format(tinkerboard_id, strftime("%Y-%m-%d", gmtime())), Body=data)
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

        if frame_count % 30 == 0:
            average_fps = 1 / (inference_time_sum / 30)
            inference_time_sum = 0

        else:
            inference_time_sum += inf_time

        print "Inference time: {0} ms, FPS Average: {1}, Time Elapsed:{2} ".format(inf_time * 1000, average_fps,
                                                                                   (time_elapsed - start_time) / 100)
        gc.collect()

    if write_video:
        writer.close()


if __name__ == "__main__":
    main()
