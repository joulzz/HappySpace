import cv2
import numpy as np


class PersonDetector:
    def __init__(self, model_binary, model_config, model_type):
        self.model_binary = model_binary
        self.model_config = model_config
        self.person_bounding_boxes = []
        print dir(self)
        if hasattr(self, "load_{0}_model".format(model_type)):

            model_loader = getattr(self, "load_{0}_model".format(model_type))
            model_loader()

    def load_ssd_model(self):
        print "Loaded DNN model"
        self.net = cv2.dnn.readNetFromTensorflow(self.model_binary, self.model_config)

    def detect_person(self, frame):
        self.person_bounding_boxes= []
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5, True, False)
        self.net.setInput(blob)
        self.detections = self.net.forward()

        rows = frame.shape[0]
        cols = frame.shape[1]
        for i in range(self.detections.shape[2]):
            confidence = self.detections[0, 0, i, 2]
            if confidence > 0.6:
                class_id = int(self.detections[0, 0, i, 1])

                xLeftBottom = int(self.detections[0, 0, i, 3] * cols)
                yLeftBottom = int(self.detections[0, 0, i, 4] * rows)
                xRightTop = int(self.detections[0, 0, i, 5] * cols)
                yRightTop = int(self.detections[0, 0, i, 6] * rows)
                if class_id == 1:
                    self.person_bounding_boxes.append(((xLeftBottom, yRightTop), (xRightTop, yLeftBottom)))

        for person in self.person_bounding_boxes:
            for person2 in self.person_bounding_boxes:
                if person == person2:
                    continue
                center = (person2[0][0] + person2[1][0])/2, (person2[0][1] + person2[1][1])/2
                area1 = (person[1][0] - person[0][0]) * (person[0][1] - person[1][1])
                area2 = (person2[1][0] - person2[0][0]) * (person2[0][1] - person2[1][1])



                if center[0] > person[0][0] and center[0] < person[1][0] and center[1] > person[1][1] and center[1] < person[0][1]:
                    if area1 > area2:
                        self.person_bounding_boxes.remove(person2)
                    elif area2 > area1:
                        self.person_bounding_boxes.remove(person)
                    else:
                        self.person_bounding_boxes.remove(person)
                    # cv2.rectangle(frame, , , (255, 0, 0), 2)
                    # cv2.imshow('frame', frame)
                    # cv2.waitKey(10)


if __name__ == "__main__":
    persondetect = PersonDetector("../Models/ssd/ssd.pb", "../Models/ssd/ssd.pbtxt", "ssd")
    cap = cv2.VideoCapture(0)
    i =0
    while True:
        _, frame = cap.read()
        persondetect.detect_person(frame)
        draw_frame = np.copy(frame)
        for bbox in persondetect.person_bounding_boxes:
            cv2.rectangle(draw_frame, bbox[0], bbox[1], (255, 0, 0), 2)
        cv2.imshow('frame', draw_frame)
        cv2.waitKey(10)
        i += 1