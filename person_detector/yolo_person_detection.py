import cv2
import numpy as np


class PersonDetector:
    def __init__(self, model_binary, model_config, model_type, size):
        self.model_binary = model_binary
        self.size = size
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
        blob = cv2.dnn.blobFromImage(frame, 1./255, (self.size, self.size), (0, 0, 0), True, False)
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