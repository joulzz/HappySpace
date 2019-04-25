from tracking.iou_tracking import Tracker
import cv2
import numpy as np
from face_detector.face_detector import FaceDetection
from smile_counter.people_counter import PeopleTracker, PeopleCounter, People
from sentiment_net.sentiment_net import SmileDetector
# import pandas as pd
# from skvideo.io import FFmpegWriter
import gc
# import subprocess
from configuration_module.json_parser import json_parser
from time import gmtime, strftime, time,sleep
import sys
import boto3
import os
import multiprocessing as mp
# from blinkstick_led import led_blink
from openvino.inference_engine import IENetwork, IEPlugin
from imutils.video import VideoStream
import imutils
# import picamera
from picamera.array import PiRGBArray
# from gps_module import read_gps_data
# from bicolor_led import smiling_face,straight_face,colour_gauge,colour_gauge_update
# from Adafruit_LED_Backpack import BicolorMatrix8x8

def main():

    dir_path = os.path.dirname(os.path.abspath(__file__))

    if len(sys.argv)!= 2:
        print("\n Give path to the JSON Configuration File\n Example: python smile_detection_demo.py <full path to json file>")
        return


    # Read parameters from JSON file. Refer to word document for parameter functions
    tinkerboard_id, skip_frame, display_flag, write_video, remote_upload, dongle_connection, running_time, min_face, max_face, write_images = json_parser(sys.argv[1])

    if dongle_connection:
        print("Disconnecting via sakis3g (Main)")
        subprocess.check_output(['sudo', '/usr/bin/modem3g/sakis3g', 'disconnect'])
        sleep(10)

    plugin = IEPlugin(device="MYRIAD")
    # Create instances of required class objects
    tracker_iou = Tracker()
    people_tracker = PeopleTracker()
    person_counter = PeopleCounter()

    model_xml = os.path.join(dir_path, "Models/intel_models/face-detection-retail-0004.xml")
    face_detector = FaceDetection(plugin, model_xml)

    emo_model_xml = os.path.join(dir_path, "Models/intel_models/emotions-recognition-retail-0003.xml")
    smile_detector = SmileDetector(plugin, emo_model_xml)
    tracker_types = ['KCF', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[2]


    # tracker = Tracker()
    s3 = boto3.resource('s3')

    cur_request_id = 0
    next_request_id = 1
    is_async_mode = True
    usingPiCamera = True

    if display_flag:
        cv2.namedWindow("frame", cv2.WINDOW_FREERATIO)

    if usingPiCamera:
        # camera = PiCamera()
        # camera.resolution = (640, 480)
        # camera.framerate = 32
        # rawCapture = PiRGBArray(camera, size=(640, 480))
        vs = VideoStream(src=0, usePiCamera=usingPiCamera,resolution=(640, 480), framerate=32).start()
        sleep(2.0)
    else:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 24)

    if write_video:
        writer = FFmpegWriter(os.path.join(dir_path, "output.mp4"))

    previous_frame = []
    frame_count = 0
    # _, frame = cap.read()


    # Comment if running on local machine, swapoff swapon required for tinkerboard
    # subprocess.check_output(["sudo", "swapoff","-a"])
    # subprocess.check_output(["sudo", "swapon","-a"])


    start_time = int(strftime("%H%M", gmtime()))
    start_time_seconds = time()
    inference_time_sum = 0
    average_fps = 0
    time_face = 0
    # led = blinkstick.find_first()
    # led.set_mode(3)
    # subprocess.check_output(['sudo', 'blinkstick', '--set-mode','3'])

    # led = mp.Process(target=led_blink("yellow"), daemon=True)
    # led.start()

    if usingPiCamera:
        cameraCap = vs
        frame = vs.read()
    else:
        cameraCap = cap.isOpened()
        frame = cap.read()

    while cameraCap:
        total_smile_counter = 0
        if not usingPiCamera:
            cameraCap = cap
        if is_async_mode:
            next_frame = cameraCap.read()
            if next_frame.any():
                flag= True
            if not (flag):
                print("Skipping Frame")
                continue
            next_frame = cv2.resize(next_frame, (640, 480))
        else:
            frame = cameraCap.read()
            if frame.any():
                flag = True
            if not (flag):
                print("Skipping Frame")
                continue
            frame = cv2.resize(frame, (640, 480))

        # _, frame = cap.read()
        # frame = np.array(frame, dtype=np.uint8)
        # original = np.copy(frame)
        # frame = frame[roi[1]: roi[3], roi[0]: roi[2]]
        t0 = cv2.getTickCount()
        # Set previous frame at the start
        if len(previous_frame) == 0:
            previous_frame = np.copy(frame)
        #1
        # frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=100)

        #2 Histogram Equalization
        # img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
        # frame = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        current_frame = frame

        # Set frame for drawing purposes
        draw_frame = np.copy(current_frame)

        # Initialize face detection
        # print(np.shape(current_frame),current_frame.dtype)

        if is_async_mode:
            face_detector.preprocessing(next_frame)
            face_detector.asyncCall(next_request_id)
        else:
            face_detector.preprocessing(frame)
            face_detector.asyncCall(cur_request_id)

        if face_detector.awaitResults(cur_request_id,draw_frame):
            people_tracker.current_frame_bboxes = face_detector.faces
            print('Current Frame bboxes',people_tracker.current_frame_bboxes)
        state = []
        bboxes = []
        current_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())

        # Here person_counter.people corresponds to previous frame people and people_tracker.current_frame_bboxes to people in current frame
        # for person in person_counter.people:
        #     if person.current:
        #         person.current = False
        #         previous_bbox= person.bbox
        #         previous_bbox_edited = (person.bbox[0][0], person.bbox[0][1], person.bbox[1][0], person.bbox[1][1])
        #         tracker.init(frame, previous_bbox_edited)
        #         print('Previous bbox', previous_bbox_edited)
        #
        #         bbox_overlaps = []
        #
        #         # Add overlaps between previous bboxes and current bboxes to an array
        #         # for current_bbox in people_tracker.current_frame_bboxes:
        #         #     overlap = tracker.iou_tracker(previous_bbox, current_bbox)
        #         #     bbox_overlaps.append(overlap)
        #
        #
        #         ok, current_bbox_raw= tracker.update(next_frame)
        #         print('Current bbox', current_bbox_raw)
        #         current_bbox= list(map(lambda x: int(x) , current_bbox_raw))
        #         current_bbox = ((current_bbox[0],current_bbox[1]),(current_bbox[2],current_bbox[3]))
        #         # if len(bbox_overlaps) != 0:
        #         #     # If overlap is greater than 50%, replace previous bbox with current one
        #         #     if max(bbox_overlaps) > 0.5:
        #                 # person.gps = read_gps_data()
        #
        #         if ok:
        #             person.history.append(person.bbox)
        #             person.bbox = current_bbox
        #             person.current = True
        #             people_tracker.current_frame_bboxes.remove(previous_bbox)
        #         else:
        #             # Tracking failure
        #             cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0, 0, 255), 2)

        for person in person_counter.people:
            if person.current:
                person.current = False
                previous_bbox = person.bbox
                ok, tracked_bbox_raw = person.tracker.update(next_frame)
                tracked_bbox = list(map(lambda x: int(x) , tracked_bbox_raw))
                tracked_bbox = ((tracked_bbox[0],tracked_bbox[1]),(tracked_bbox[2],tracked_bbox[3]))
                bbox_overlaps = []

                # Add overlaps between previous bboxes and current bboxes to an array
                for current_bbox in people_tracker.current_frame_bboxes:
                    overlap = tracker_iou.iou_tracker(tracked_bbox, current_bbox)
                    bbox_overlaps.append(overlap)

                if len(bbox_overlaps) != 0:
                    # If overlap is greater than 50%, replace previous bbox with current one
                    if max(bbox_overlaps) > 0.5:
                        # person.gps = read_gps_data()
                        person.history.append(person.bbox)
                        person.bbox = people_tracker.current_frame_bboxes[bbox_overlaps.index(max(bbox_overlaps))]
                        person.current = True
                        people_tracker.current_frame_bboxes.remove(previous_bbox)

        # Add unreplaced bboxes
        for bbox in people_tracker.current_frame_bboxes:
            max_idx = len(person_counter.people)
            new_person = People()
            new_person.bbox = bbox
            new_person.current = True
            new_person.id = max_idx
            new_person.timestamp = current_time
            if tracker_type == 'KCF':
                tracker = cv2.TrackerKCF_create()
            if tracker_type == 'MOSSE':
                tracker = cv2.TrackerMOSSE_create()
            if tracker_type == "CSRT":
                tracker = cv2.TrackerCSRT_create()
            new_person.tracker = tracker
            new_person_bbox_edited = (new_person.bbox[0][0], new_person.bbox[0][1], new_person.bbox[1][0], new_person.bbox[1][1])
            new_person.tracker.init(frame,new_person_bbox_edited)
            # Uncomment to log GPS functionality
            # new_person.gps = read_gps_data()
            person_counter.add(new_person)

        # person_counter.people is now updated to correspond to people in the current frame

        # if frame_count % 5 ==0:
        if frame_count % (skip_frame+1) == 0:
            print("Sentiment Net Run")
            for people in person_counter.people:
                if people.current:
                    # tb1 = cv2.getTickCount()
                    # led.blink(name="yellow", delay=10)
                    # time_blinkstick = (cv2.getTickCount() - tb1) / cv2.getTickFrequency()
                    # subprocess.check_output(['sudo', 'blinkstick','--blink','yellow'])
                    face = people.bbox
                    try:
                        face_frame= smile_detector.preprocess_image(frame[face[0][1]: face[1][1], face[0][0]: face[1][0]])
                    except:
                        print("Exception Raised in Resizing Image")
                        continue


                    # Add directory for smiles and non-smiles if they don't exist
                    if not os.path.exists('smile_images'):
                        os.makedirs('smile_images')
                    if not os.path.exists('non_smiles_images'):
                        os.makedirs('non_smiles_images')

                    if people.count == None:
                        people.count = 0
                    # Classify and save as smiles and non-smiles
                    if smile_detector.predict(face_frame):
                        # Displaying smiling face, Change color using [BicolorMatrix8x8.RED, BicolorMatrix8x8.GREEN, BicolorMatrix8x8.YELLOW]
                        # smiling_face(BicolorMatrix8x8.GREEN)
                        # tb2 = cv2.getTickCount()
                        # led.blink(name="green",  delay=10)
                        # time_blinkstick += (cv2.getTickCount() - tb2) / cv2.getTickFrequency()
                        # print("Blinkstick Total time: {0} ms".format(time_blinkstick * 1000))
                        # subprocess.check_output(['sudo', 'blinkstick','--blink', 'green'])
                        # Check flag 'write_images' to then save images to the images folder in current directory
                        if write_images:
                            cv2.imwrite(
                            "{0}/{1}_{2}.jpg".format(os.path.join(dir_path, "smile_images"), people.id, people.count),
                            current_frame[face[0][1]+int((face[1][1]-face[0][1])*(0.55)): face[1][1], face[0][0]: face[1][0]])
                        people.count += 1
                    else:
                        if write_images:
                            if people.non_smiles == 0:
                                cv2.imwrite(
                                    "{0}/{1}_{2}.jpg".format(os.path.join(dir_path, "non_smiles_images"), people.id, people.non_smiles),
                                    current_frame[face[0][1] + int((face[1][1] - face[0][1])*(0.55)): face[1][1],
                                    face[0][0]: face[1][0]])
                        time_straight = int(time())
                        # Change color using [BicolorMatrix8x8.RED, BicolorMatrix8x8.GREEN, BicolorMatrix8x8.YELLOW]
                        # straight_face(BicolorMatrix8x8.YELLOW)
                        people.non_smiles += 1
                    time_face = int(time())

        for person in person_counter.people:
            if person.count != None:
                total_smile_counter += person.count
            # state.append(person.current)
            # bboxes.append(person.bbox)
            if person.current:
                people_tracker.previous_frame_bboxes.append(person.bbox)
                cv2.rectangle(draw_frame, person.bbox[0], person.bbox[1], (255, 255, 255), 1)
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
            # Uncomment to log GPS functionality
            # gps_dd = []
            for people in person_counter.people:
                people.history.append(people.bbox)
                ids.append(people.id)
                smile_count.append(people.count)
                last_bbox.append(people.bbox)
                location_history.append(people.history)
                timestamp.append(people.timestamp)
                # Uncomment to log GPS functionality
                # gps_dd.append(people.gps)

            df["ID"] = ids
            df["Smiles_Detected"] = smile_count
            df["Last_Location"] = last_bbox
            df["Location_History"] =location_history
            df["Timestamp"] = timestamp
            # df["GPS_DD"] = gps_dd

            df.to_csv(os.path.join(dir_path, "output.csv"), index=False)
            print("Wrote to CSV")


            if remote_upload:

                if dongle_connection:
                    print("Connecting via sakis3g (Main)")
                    subprocess.check_output(['sudo','/usr/bin/modem3g/sakis3g','connect'])
                    sleep(10)

                data = open(os.path.join(dir_path, 'output.csv'), 'rb')
                s3.Bucket('smile-log').put_object(Key='{0}/{1}.csv'.format(tinkerboard_id, strftime("%Y-%m-%d", gmtime())), Body=data)

                if os.path.exists('smile_images'):
                    image_path = os.path.join(dir_path, "smile_images")
                    for subdir, dirs, files in os.walk(image_path):
                        for file in files:
                            full_path = os.path.join(subdir, file)
                            with open(full_path, 'rb') as data:
                                s3.Bucket('smile-log').put_object(Key='{0}/{1}_smile_images/{2}'.format(tinkerboard_id,strftime("%Y-%m-%d",gmtime()),full_path[len(image_path) + 1:]),Body=data)

                if os.path.exists('non_smiles_images'):
                    image_path = os.path.join(dir_path, "non_smiles_images")
                    for subdir, dirs, files in os.walk(image_path):
                        for file in files:
                            full_path = os.path.join(subdir, file)
                            with open(full_path, 'rb') as data:
                                s3.Bucket('smile-log').put_object(
                                    Key='{0}/{1}_non_smiles_images/{2}'.format(tinkerboard_id,strftime("%Y-%m-%d", gmtime()),full_path[len(image_path) + 1:]),Body=data)
            break

        frame_count += 1

        original = draw_frame
        cv2.putText(original, "Total Smiles: {0}".format(total_smile_counter), (0, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        time_elapsed_seconds = int(time() - start_time_seconds)
        time_gauge = int(time())
        # print str(time_elapsed_seconds) +" "+str(time_smile)+" "+str(time_straight)+" "+str(time_gauge)
        # print str(int(time_straight - time_gauge))+" "+str(int(time_smile - time_gauge))

        # Displaying Colour Gauge
        # if abs(int(time_gauge - time_face)) % 3600 == 0 and time_face != 0:
        #     print("Color Gauge")
            # colour_gauge_update(total_smile_counter)

        # Display tracker type on frame
        cv2.putText(frame, tracker_type + "Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

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

        if is_async_mode:
            cur_request_id, next_request_id = next_request_id, cur_request_id
            frame = next_frame

        print("Inference time: {0} ms, FPS Average: {1}, Time Elapsed:{2} ".format(inf_time * 1000, average_fps,
                                                                                   (time_elapsed - start_time) / 100))
        # gc.collect()

    if write_video:
        writer.close()

    if usingPiCamera:
        vs.stop()
    else:
        cap.release()

    cv2.destroyAllWindows()

    del face_detector.exec_net
    del plugin


if __name__ == "__main__":
    main()

