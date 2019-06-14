
import cv2
import numpy as np
import os
from openvino.inference_engine import IENetwork

class GAPredictor:
    def __init__(self, plugin, ga_model_xml):
        ga_model_bin = os.path.splitext(ga_model_xml)[0] + ".bin"
        self.ga_net = IENetwork(model=ga_model_xml, weights=ga_model_bin)

        self.ga_input_blob = next(iter(self.ga_net.inputs))
        self.ga_out_blob = next(iter(self.ga_net.outputs))
        self.ga_net.batch_size = 1

        self.ga_size = self.ga_net.inputs[self.ga_input_blob].shape
        print("Batch Size: ", self.ga_size[0])

        self.exec_ga_net = plugin.load(network=self.ga_net)

        self.images_ga_net = np.zeros(shape=self.ga_size)
        del self.ga_net



    def preprocess_image(self, face):
        if face.shape[:-1] != (self.ga_size[2], self.ga_size[3]):
            face_frame = cv2.resize(face, (self.ga_size[3], self.ga_size[2]))

        face_frame = face_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        return face_frame

    def predict(self, face_frame):

        res = self.exec_ga_net.infer(inputs={self.ga_input_blob: face_frame})
        res = res[self.ga_out_blob]
        print(res)
        return res

        # result = self.network.predict(self.image)
        # output = self.final_layer.predict(result)
        # if output:
        #     return False
        # else:
        #     return True