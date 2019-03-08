from em_model import EMR
import cv2
import numpy as np
from sklearn.externals import joblib
import os
from openvino.inference_engine import IENetwork

class SmileDetector:
    def __init__(self, plugin, emo_model_xml):
        emo_model_bin = os.path.splitext(emo_model_xml)[0] + ".bin"
        self.emo_net = IENetwork(model=emo_model_xml, weights=emo_model_bin)

        self.emotion_input_blob = next(iter(self.emo_net.inputs))
        self.emotion_out_blob = next(iter(self.emo_net.outputs))
        self.emo_net.batch_size = 1

        self.emo_size = self.emo_net.inputs[self.emotion_input_blob].shape
        print("Batch Size: ", self.emo_size[0])

        self.exec_emo_net = plugin.load(network=self.emo_net)

        self.images_emotion_net = np.zeros(shape=self.emo_size)
        del self.emo_net



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