from person_detector.ssd_person_detection import PersonDetector
from tracking.optical_flow_tracker import Tracker
import cv2
import numpy as np
from face_detector.face_detector import FaceDetection
from smile_counter.people_counter import PeopleCounter
from smile_counter.smile_counter import SmileCounter
from sentiment_net.sentiment_net import SmileDetector

def person_counter():
    persondetect = PersonDetector("Models/ssd/ssd.pb", "Models/ssd/ssd.pbtxt", "ssd")
    people_counter = PeopleCounter()
    face_detector = FaceDetection("Models/haarcascade_frontalface_default.xml")
    smile_detector = SmileDetector()
    smile_counter = SmileCounter()

    cap = cv2.VideoCapture(1)
    tracker = Tracker()
    tracking_set = True
    # k = 0
    while True:
        _, frame = cap.read()
        # if k < 500:
        #     k += 1
        #
        #     continue
        draw_frame = np.copy(frame)
        persondetect.detect_person(frame)
        detected_bboxes = persondetect.person_bounding_boxes
        current_boxes = detected_bboxes[:]

        if len(detected_bboxes) + len(people_counter.detected_bbox) == 0:
            tracking_set = True
            cv2.imshow('frame', frame)
            cv2.waitKey(5)
            continue

        if tracking_set:
            tracker.initialize_tracker(frame)
            for bbox in detected_bboxes:
                people_counter.add(bbox)
                smile_counter.add()

            tracking_set = False
            continue
        else:
            tracker.run_tracker(frame)

        for count, old_bbox in enumerate(people_counter.detected_bbox):
            for new_bbox in detected_bboxes:
                new_count = 0
                feature_count = 0
                old_count = len(tracker.good_old_features)
                for i, (new, old) in enumerate(zip(tracker.good_new_features, tracker.good_old_features)):
                    a, b = old.ravel()
                    c, d = new.ravel()
                    cv2.circle(draw_frame, (a, b), 2, (0, 255, 0), 2)
                    if a > old_bbox[0][0] and a < old_bbox[1][0] and b > old_bbox[1][1] and b < old_bbox[0][1] and \
                                    c > new_bbox[0][0] and c < new_bbox[1][0] and d > new_bbox[1][1] and d < \
                            new_bbox[0][1]:
                        cv2.circle(draw_frame, (c, d), 2, (255, 255, 0), 2)
                        new_count += 1
                    if a > new_bbox[0][0] and a < new_bbox[1][0] and b > new_bbox[1][1] and b < new_bbox[0][1]:
                        feature_count += 1


                if float(new_count) /old_count > 0.2:
                    # cv2.rectangle(draw_frame, new_bbox[0], new_bbox[1], (0, 0, 255), 2)
                    if new_bbox in detected_bboxes:
                        detected_bboxes.remove(new_bbox)
                        people_counter.replace(old_bbox, new_bbox)
                # if float(new_count)/ old_count < 0.2:
                #     people_counter.detected_bbox[count] = ((0, 0), (0, 0))

                if feature_count < 5:
                    if new_bbox in detected_bboxes:
                        detected_bboxes.remove(new_bbox)

        if len(detected_bboxes) != 0:
            for bbox in detected_bboxes:
                people_counter.add(bbox)
                smile_counter.add()


        for i, person in enumerate(people_counter.detected_bbox):
            if person not in current_boxes:
               continue

            face_detector.run_facedetector(frame)
            for face in face_detector.faces:
                (x, y, w, h) = face
                cropped_face = frame[y: y+h, x: x+ w]
                face = (x, y), (x + w, y + h)
                cv2.rectangle(draw_frame, face[0], face[1], (0, 255, 0), 2)
                smile_detector.preprocess_image(cropped_face)
                if smile_detector.predict():
                    smile_counter.add_smile(face, person, i)
            cv2.rectangle(draw_frame, person[0], person[1], (255, 0, 0), 2)
            cv2.putText(draw_frame, "Person: {0}".format(i), (person[0][0], person[1][1]), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1)
            print smile_counter.smiles_array
            cv2.putText(draw_frame, "Smiles: {0}".format(smile_counter.smiles_array[i]), person[0], cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1)

        print "Total People Detected: {0}".format(people_counter.people_detected)
        cv2.imshow('frame', draw_frame)
        cv2.waitKey(5)


if __name__ == "__main__":
    person_counter()