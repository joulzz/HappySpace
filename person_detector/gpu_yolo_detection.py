from darkflow.net.build import TFNet

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
        options = {"model": "/home/suraj/Repositories/SmileDetection/Models/yolo/yolov2.cfg",
                   "load": "/home/suraj/Repositories/SmileDetection/Models/yolo/yolov2.weights", "threshold": 0.4,
                   "gpu": 1}

        self.net = TFNet(options)

    def detect_person(self, frame):
        self.person_bounding_boxes= []
        self.detections = self.net.return_predict(frame)

        for detection in self.detections:
            confidence = detection['confidence']

            label = detection['label']
            if label != 'person':
                continue
            if confidence > 0.5:
                left = detection['topleft']['x']
                top = detection['topleft']['y']
                right = detection['bottomright']['x']
                bottom = detection['bottomright']['y']
                self.person_bounding_boxes.append(((int(left), int(top)), (int(right), int(bottom))))
