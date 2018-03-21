from em_model import EMR
import cv2
import numpy as np

class SmileDetector:
    def __init__(self):
        self.network = EMR("Models/sentiment_net/sentiment_net")
        self.network.build_network()


    def preprocess_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.image = cv2.resize(image, (48, 48), interpolation = cv2.INTER_CUBIC) / 255.

    def predict(self):
        result = self.network.predict(self.image)
        if result is not None:
            if result[0][6] < 0.6:
                result[0][6] = result[0][6] - 0.12
                result[0][:3] += 0.01
                result[0][4:5] += 0.04
            maxindex = np.argmax(result[0])
            font = cv2.FONT_HERSHEY_SIMPLEX
            if maxindex == 3:
                return True
            else:
                return False
        else:
            return False