
import cv2
import numpy as np
import os
from openvino.inference_engine import IENetwork

class GAPredictor:
    """
       Class used to predict the Age and Gender of person using the age-gender-recognition-retail-0013 model

       Attributes
       ----------
       gender_list:
           Labels list for mapping between output vector and gender.
       ga_net:
           Gender-Age network description that is sent to plugin to load
       ga_input_blob:
           The name of the node in the network where the image is passed as input
       ga_out_blob:
            The name of the node in the network where the result vector is received as outpuit

       ga_size:
            Size of the image expected at the input node
       exec_ga_net:
            The in-memory network loaded that executes a forward pass over the network

       """
    def __init__(self, plugin, ga_model_xml):
    
        """ Description

        Initializes and loads the gender-age prediction network into memory

        :param plugin: Inference Engine Plugin
    
        :type ga_model_xml: string
        :param ga_model_xml: XML path of the ga xml description

        """    
        self.gender_list = ["female", "male"]

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
    
        """ Description

        Preprocess the image by resizing and transposing as required by the Gender-Age prediction network in IE format.
    
        :type face: numpy array
        :param face: Image representing crop area of the detected face.

        :rtype face_frame:
        :returns face_frame: Returns processed numpy array ready to be sent for inference

        """
        if face.shape[:-1] != (self.ga_size[2], self.ga_size[3]):
            face_frame = cv2.resize(face, (self.ga_size[3], self.ga_size[2]))

        face_frame = face_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        return face_frame

    def predict(self, face_frame):

    
        """ Description

        Runs the Gender Age Prediction on the input and provides the age, gender output

        :param face_frame: The numpy array returned by the preprocess_image function

        :rtype age: float
        :returns age: Age in float format

        :rtype age: string
        :returns age: Gender as either "male" or "female"

        """
        res = self.exec_ga_net.infer(inputs={self.ga_input_blob: face_frame})
        age = int(res['age_conv3'].reshape(-1) * 100)
        gender = self.gender_list[np.argmax(res['prob'].reshape(-1, 2))]

        return age, gender

        # result = self.network.predict(self.image)
        # output = self.final_layer.predict(result)
        # if output:
        #     return False
        # else:
        #     return True