import cv2
import numpy as np
from face_detector.face_detector import FaceDetection
from sentiment_net.sentiment_net import SmileDetector
import pandas as pd
from skvideo.io import FFmpegWriter
import gc
import subprocess

def main():

    # Keep track of time to store data into csv files
    time_elapsed = 0


    # Create instances of required class objects

    face_detector = FaceDetection("Models/haarcascade_frontalface_default.xml")
    smile_detector = SmileDetector()


    cv2.namedWindow("preview", cv2.WINDOW_FREERATIO)
    # cv2.namedWindow("preview")
    cap = cv2.VideoCapture("test.mp4")
    writer = FFmpegWriter("output.mp4")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    previous_frame = []
    frame_count = 0
    _, frame = cap.read()
    smile_timestamps = []
    smile_bboxes = []
    subprocess.check_output(["sudo", "swapoff","-a"])
    subprocess.check_output(["sudo", "swapon","-a"])
    total_smile_counter = 0
    clear = 1
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



            print current_frame_bboxes
            for face in current_frame_bboxes:
                cv2.rectangle(draw_frame, face[0], face[1], (255, 255, 255), 3)

                smile_detector.preprocess_image(current_frame[face[0][1]: face[1][1], face[0][0]: face[1][0]])
                if smile_detector.predict():
                    total_smile_counter += 1

            smile_timestamps.append(frame_count * clear)
            smile_bboxes.append(current_frame_bboxes)
            if frame_count % 500 == 0:
                frame_count = 0
                clear += 1
                df = pd.DataFrame()
                df["Current_Bboxes"] = smile_bboxes
                df["Timestamp"] = smile_timestamps

                df.to_csv("output.csv", index=False)
            frame_count += 1

            original = draw_frame
            cv2.putText(original, "Total Smiles: {0}".format(total_smile_counter), (0, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

            writer_image = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            writer.writeFrame(writer_image)
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
