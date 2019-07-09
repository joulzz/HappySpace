import cv2
import numpy as np
import os
from openvino.inference_engine import IENetwork

class FaceReidentification:
    def __init__(self, plugin, fr_model_xml):
        fr_model_bin = os.path.splitext(fr_model_xml)[0] + ".bin"
        self.fr_net = IENetwork(model=fr_model_xml, weights=fr_model_bin)

        self.fr_input_blob = next(iter(self.fr_net.inputs))
        self.fr_out_blob = next(iter(self.fr_net.outputs))
        self.fr_net.batch_size = 1

        self.fr_size = self.fr_net.inputs[self.fr_input_blob].shape
        print("Batch Size: ", self.fr_size[0])

        self.exec_fr_net = plugin.load(network=self.fr_net)

        self.images_fr_net = np.zeros(shape=self.fr_size)

        del self.ga_net

