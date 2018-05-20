from em_model import EMR
import cv2
import numpy as np
from sklearn.externals import joblib
import os

class SmileDetector:
    def __init__(self):
        dir_path = os.path.dirname(os.path.abspath(__file__))
        self.network = EMR(os.path.abspath(os.path.join(dir_path, "../Models/sentiment_net/sentiment_net")))
        self.network.build_network()
        self.final_layer = joblib.load(os.path.abspath(os.path.join(dir_path, "../Models/svm_model.pkl")))

    def preprocess_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.image = cv2.resize(image, (48, 48), interpolation = cv2.INTER_CUBIC) / 255.

    def predict(self):
        result = self.network.predict(self.image)
        output = self.final_layer.predict(result)
        if output:
            return False
        else:
            return True