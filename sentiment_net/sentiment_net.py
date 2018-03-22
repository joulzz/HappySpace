from em_model import EMR
import cv2
import numpy as np
from sklearn.externals import joblib
class SmileDetector:
    def __init__(self):
        self.network = EMR("Models/sentiment_net/sentiment_net")
        self.network.build_network()
        self.final_layer = joblib.load("Models/svm_model.pkl")

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