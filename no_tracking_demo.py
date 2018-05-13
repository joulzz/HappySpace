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

def main():

    # Keep track of time to store data into csv files
    time_elapsed = 0


    # Create instances of required class objects
    people_tracker = PeopleTracker()
    person_counter = PeopleCounter()
    face_detector = FaceDetection("Models/haarcascade_frontalface_default.xml")
    smile_detector = SmileDetector()
    #tracker = Tracker()


    # cv2.namedWindow("frame", cv2.WINDOW_FREERATIO)
    cv2.namedWindow("preview")
    cap = cv2.VideoCapture(0)
    writer = FFmpegWriter("output.mp4")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    previous_frame = []
    frame_count = 0
    _, frame = cap.read()

    #subprocess.check_output(["sudo", "swapoff","-a"])
    #subprocess.check_output(["sudo", "swapon","-a"])

    while cap.isOpened():
        try:
            total_smile_counter = 0
            _, frame = cap.read()
            original = np.copy(frame)
            # frame = frame[roi[1]: roi[3], roi[0]: roi[2]]
            t0 = cv2.getTickCount()
            # Set previous frame at the start

            current_frame = frame

            # Set frame for drawing purposes
            draw_frame = np.copy(frame)

            # Initialize face detection
            face_detector.run_facedetector(current_frame)
            current_frame_bboxes=[]
            if face_detector.faces:
                current_frame_bboxes.append(face_detector.faces)

            state = []

            # if frame_count % 5 ==0:


            for bboxes in current_frame_bboxes:
                face = current_frame_bboxes[bboxes]
                smile_detector.preprocess_image(current_frame[face[0][1]: face[1][1], face[0][0]: face[1][0]])
                if smile_detector.predict():
                    total_smile_counter += 1
                cv2.rectangle(draw_frame, current_frame_bboxes[0], current_frame_bboxes[1], (255, 255, 255), 3)



            if frame_count % 5000 == 0:
                frame_count = 0
                df = pd.DataFrame()
                curent_bboxes=[]
                timestamp = []
                for bboxes in current_frame_bboxes:
                    curent_bboxes.append(bboxes)
                    #timestamp.append(people.timestamp)

                df["Current_Bboxes"] = curent_bboxes
                #df["Timestamp"] = timestamp

                df.to_csv("output.csv", index=False)
            frame_count += 1

            original = draw_frame
            cv2.putText(original, "Total Smiles: {0}".format(total_smile_counter), (0, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

            # cv2.imshow('frame', original)
            # ch = 0xFF & cv2.waitKey(2)
            # if ch == 27:
            #     break
            writer_image = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            #writer.writeFrame(writer_image)
            cv2.imshow("preview",original)
            cv2.waitKey(20)

            inf_time = (cv2.getTickCount() - t0)/ cv2.getTickFrequency()
            print "Inference time: {0} ms, FPS: {1}, Time Elapsed:{2} ".format(inf_time * 1000, 1/ inf_time, float(time_elapsed)/3600)
            time_elapsed += inf_time
            gc.collect()

        except:
            pass

    writer.close()


if __name__ == "__main__":
    main()
