from person_detector.ssd_person_detection import PersonDetector
from tracking.optical_flow_tracker import Tracker
import cv2
import numpy as np

if __name__ == "__main__":

    tracked_bboxes = []
    persondetect = PersonDetector("/home/suraj/Repositories/YoloSockets/deploy/models/ssd/ssd.pb", "/home/suraj/Repositories/YoloSockets/deploy/models/ssd/ssd.pbtxt", "ssd")
    cap = cv2.VideoCapture(1)
    tracker = Tracker()
    tracking_set = True
    while True:
        _, frame = cap.read()
        draw_frame = np.copy(frame)
        persondetect.detect_person(frame)
        detected_bboxes = persondetect.person_bounding_boxes
        for bbox in detected_bboxes:
            cv2.rectangle(draw_frame, bbox[0], bbox[1], (255, 0, 0), 2)
        if len(detected_bboxes) == 0:
            cv2.imshow('frame', frame)
            cv2.waitKey(10)
            continue

        new_bboxes = []
        if tracking_set:
            tracker.initialize_tracker(frame)
            tracking_set = False 
            continue
        else:
            tracker.run_tracker(frame)

        # for bbox in detected_bboxes:
        for i, (new, old) in enumerate(zip(tracker.good_new_features, tracker.good_old_features)):
            a, b = new.ravel()
            c, d = old.ravel()
            cv2.line(draw_frame, (a, b), (c, d), (0, 0, 255), 2)
            cv2.circle(draw_frame, (a, b), 5, (255, 0, 0), -1)



        #     for bbox1 in detected_bboxes:
        #         for bbox2 in tracked_bboxes:
        #             tracked_box_center = [(bbox2[0][0] + bbox2[1][0])/2, (bbox2[0][1] + bbox2[1][1])/2]
        #             if (tracked_box_center[0] > bbox1[0][0] and tracked_box_center[0] < bbox1[1][0] and tracked_box_center[1] < bbox1[0][1] and tracked_box_center[1] > bbox1[1][1]):
        #                 new_bboxes.append(bbox1)
        #
        # tracker.initialize_tracker(frame, new_bboxes)
        #
        # tracker.tracker_run(frame)
        # # tracked_bboxes = []
        # print (len(tracker.tracked_bboxes))
        # for bbox in tracker.tracked_bboxes:
        #     bbox = cv2.boxPoints(bbox).astype(int)
        #     cv2.rectangle(draw_frame, tuple(bbox[1]), tuple(bbox[-1]), (0, 0, 255), 2)
        #     tracked_bboxes.append([tuple(bbox[1]), tuple(bbox[-1])])
        #
        # print "Total People in frame: {0}".format(len(tracked_bboxes))
        #
        cv2.imshow('frame', draw_frame)
        cv2.waitKey(10)