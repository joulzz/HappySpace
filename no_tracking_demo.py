import cv2
import numpy as np
from face_detector.face_detector import FaceDetection
from sentiment_net.sentiment_net import SmileDetector
import pandas as pd
from skvideo.io import FFmpegWriter
import gc
import subprocess
from time import gmtime, strftime
from smile_counter.people_counter import PeopleCounter, People
from configuration_module.json_parser import json_parser
import sys
import boto3
import os


def main():
    if len(sys.argv)!= 2:
        print("\n Give path to the JSON Configuration File\n Example: python smile_detection_demo.py <full path to json file>")
        return
    dir_path = os.path.dirname(os.path.abspath(__file__))

    tinkerboard_id, skip_frame, display_flag, write_video, remote_upload, running_time, min_face, max_face = json_parser(sys.argv[1])
    # Keep track of time to store data into csv files
    start_time = int(strftime("%H%M", gmtime()))
    s3 = boto3.resource('s3')

    # Create instances of required class objects

    face_detector = FaceDetection(os.path.join(dir_path, "Models/haarcascade_frontalface_default.xml"))
    smile_detector = SmileDetector()

    if display_flag:
        cv2.namedWindow("preview", cv2.WINDOW_FREERATIO)
    # cv2.namedWindow("preview")
    cap = cv2.VideoCapture(0)
    if write_video:
        writer = FFmpegWriter(os.path.join(dir_path, "output.mp4"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    person_counter = PeopleCounter()
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    frame_count = 0
    _, frame = cap.read()

    subprocess.check_output(["sudo", "swapoff","-a"])
    subprocess.check_output(["sudo", "swapon","-a"])
    total_smile_counter = 0
    max_idx = 0
    while cap.isOpened():
        try:
            _, frame = cap.read()
            original = np.copy(frame)
            t0 = cv2.getTickCount()
            # Set previous frame at the start

            current_frame = frame

            # Set frame for drawing purposes
            draw_frame = np.copy(frame)

            # Initialize face detection
            face_detector.run_facedetector(current_frame, min_face, max_face)
            current_frame_bboxes=[]
            for face in face_detector.faces:
                current_frame_bboxes.append(face)

            current_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())


            for face in current_frame_bboxes:
                cv2.rectangle(draw_frame, face[0], face[1], (255, 255, 255), 3)
                new_person = People()
                new_person.bbox = face
                new_person.current = True
                new_person.id = max_idx
                new_person.timestamp = current_time
                person_counter.add(new_person)
                max_idx += 1


                if frame_count % (skip_frame +1) == 0:
                    smile_detector.preprocess_image(current_frame[face[0][1]: face[1][1], face[0][0]: face[1][0]])

                    if smile_detector.predict():
                        total_smile_counter += 1
                        new_person.count += 1
            inf_time = (cv2.getTickCount() - t0)/ cv2.getTickFrequency()
            time_elapsed = int(strftime("%H%M", gmtime()))
            if int((time_elapsed - start_time) / 100) > running_time or (time_elapsed - start_time) == -24 + running_time :
                frame_count = 0
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
                df["Location_History"] = location_history
                df["Timestamp"] = timestamp

                df.to_csv(os.path.join(dir_path, "output.csv"), index=False)
                print("Wrote to CSV")

                if remote_upload:
                    data = open(os.path.join(dir_path, 'output.csv'), 'rb')
                    s3.Bucket('smile-log').put_object(
                        Key='{0}/{1}.csv'.format(tinkerboard_id, strftime("%Y-%m-%d", gmtime())), Body=data)
                break
            frame_count += 1

            original = draw_frame
            cv2.putText(original, "Total Smiles: {0}".format(total_smile_counter), (0, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
            if write_video:
                writer_image = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
                writer.writeFrame(writer_image)
            if display_flag:
                cv2.imshow("preview",original)
                ch = 0xFF & cv2.waitKey(2)
                if ch == 27:
                    break
            print "Inference time: {0} ms, FPS: {1}, Time Elapsed:{2} ".format(inf_time * 1000, 1/ inf_time, time_elapsed)
            gc.collect()

        except:
            pass
    if write_video:
        writer.close()


if __name__ == "__main__":
    main()
