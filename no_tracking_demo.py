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

def main():
    if len(sys.argv)!= 2:
        print("\n Give path to the JSON Configuration File\n Example: python smile_detection_demo.py <full path to json file>")
        return

    tinkerboard_id, skip_frame, display_flag, write_video, remote_upload, csv_write_frequency = json_parser(sys.argv[1])
    # Keep track of time to store data into csv files
    time_elapsed = 0


    # Create instances of required class objects

    face_detector = FaceDetection("Models/haarcascade_frontalface_default.xml")
    smile_detector = SmileDetector()

    if display_flag:
        cv2.namedWindow("preview", cv2.WINDOW_FREERATIO)
    # cv2.namedWindow("preview")
    cap = cv2.VideoCapture(0)
    if write_video:
     writer = FFmpegWriter("output.mp4")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    person_counter = PeopleCounter()
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    frame_count = 0
    _, frame = cap.read()
    smile_timestamps = []
    smile_bboxes = []
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
            face_detector.run_facedetector(current_frame)
            current_frame_bboxes=[]
            for face in face_detector.faces:
                current_frame_bboxes.append(face)



            for face in current_frame_bboxes:
                cv2.rectangle(draw_frame, face[0], face[1], (255, 255, 255), 3)
                new_person = People()
                new_person.bbox = face
                new_person.current = True
                new_person.id = max_idx
                new_person.timestamp = strftime("%Y-%m-%d %H:%M:%S", gmtime())
                person_counter.add(new_person)
                max_idx += 1


                if frame_count % (skip_frame +1) == 0:
                    smile_detector.preprocess_image(current_frame[face[0][1]: face[1][1], face[0][0]: face[1][0]])

                    if smile_detector.predict():
                        total_smile_counter += 1
                        new_person.count += 1

            if time_elapsed % csv_write_frequency == 0:
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

                df.to_csv("output.csv", index=False)
                print("Wrote to CSV")
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
            inf_time = (cv2.getTickCount() - t0)/ cv2.getTickFrequency()
            print "Inference time: {0} ms, FPS: {1}, Time Elapsed:{2} ".format(inf_time * 1000, 1/ inf_time, float(time_elapsed)/3600)
            time_elapsed = float(time_elapsed)
            time_elapsed += inf_time
            time_elapsed = int(time_elapsed)
            gc.collect()

        except:
            pass
    if write_video:
        writer.close()


if __name__ == "__main__":
    main()
