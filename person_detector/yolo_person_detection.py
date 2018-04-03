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

    def load_yolo_model(self):
        print "Loaded DNN model"
        self.net = cv2.dnn.readNetFromDarknet(self.model_config, self.model_binary)

    def detect_person(self, frame):
        self.person_bounding_boxes= []
        blob = cv2.dnn.blobFromImage(frame, 1./255, (416, 416), (0, 0, 0), True, False)
        self.net.setInput(blob)
        self.detections = self.net.forward()

        rows = frame.shape[1]
        cols = frame.shape[0]

        for detection in self.detections:
            confidences = detection[5:]

            class_id = np.argmax(confidences)
            if class_id != 0:
                continue
            confidence = confidences[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * rows)
                center_y = int(detection[1] * cols)
                width = int(detection[2] * rows)
                height = int(detection[3] * cols)
                left = center_x - width / 2
                top = center_y - height / 2
                right = center_x + width/2
                bottom = center_y + height/2
                self.person_bounding_boxes.append(((int(left), int(top)), (int(right), int(bottom))))


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