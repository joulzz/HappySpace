import cv2
import numpy as np
import os
from openvino.inference_engine import IENetwork

def align_face(face_frame, landmarks):
    """ Description

    Function is used to align the detected face to increase the accuracy of the generated feature vector
    :param face_frame: Variable that holds the detected face frame
    :param landmarks: Variable holds the facial landmarks detected
    :return: Returns the aligned face
    """
   
    left_eye, right_eye, tip_of_nose, left_lip, right_lip = landmarks
    
    # compute the angle between the eye centroids
    dy = right_eye[1] - left_eye[1]     # right eye, left eye Y
    dx = right_eye[0] - left_eye[0]  # right eye, left eye X
    angle = np.arctan2(dy, dx) * 180 / np.pi
    # compute center (x, y)-coordinates (i.e., the median point)
    # between the two eyes in the input image
    ##eyes_center = ((right_eye[0] + left_eye[0]) // 2, (right_eye[1] + left_eye[1]) // 2)
    
    ## center of face_frame
    center = (face_frame.shape[0] // 2, face_frame.shape[1] // 2)
    h, w, c = face_frame.shape
    
    # grab the rotation matrix for rotating and scaling the face
    M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    aligned_face = cv2.warpAffine(face_frame, M, (w, h))
    
    return aligned_face

class FaceReidentification:
    def __init__(self, plugin, landmark_model_xml, fr_model_xml):

        """ Description

        :param plugin: Inference Engine Plugin
        :param landmark_model_xml:  XML path of the landmarks-regression-retail-0009 model xml description
        :param fr_model_xml: XML path of the model face-reidentification-retail-0071 xml description

        """
        landmark_model_bin = os.path.splitext(landmark_model_xml)[0] + ".bin"
        self.landmark_net = IENetwork(model=landmark_model_xml, weights=landmark_model_bin)

        self.landmark_input_blob = next(iter(self.landmark_net.inputs))
        self.landmark_out_blob = next(iter(self.landmark_net.outputs))
        self.landmark_net.batch_size = 1

        self.landmark_size = self.landmark_net.inputs[self.landmark_input_blob].shape
        print("Batch Size: ", self.landmark_size[0])

        self.exec_landmark_net = plugin.load(network=self.landmark_net)

        self.images_landmark_net = np.zeros(shape=self.landmark_size)

        del self.landmark_net

        fr_model_bin = os.path.splitext(fr_model_xml)[0] + ".bin"
        self.fr_net = IENetwork(model=fr_model_xml, weights=fr_model_bin)

        self.fr_input_blob = next(iter(self.fr_net.inputs))
        self.fr_out_blob = next(iter(self.fr_net.outputs))
        self.fr_net.batch_size = 1

        self.fr_size = self.fr_net.inputs[self.fr_input_blob].shape
        print("Batch Size: ", self.fr_size[0])

        self.exec_fr_net = plugin.load(network=self.fr_net)

        self.images_fr_net = np.zeros(shape=self.fr_size)

        del self.fr_net

    
    def preprocess_image(self, face):
        """ Description

        Function preprocesses the incoming image and aligns it for a more accurate face vector result
        :return:  Returns the aligned transposed face
        """
        if face.shape[:-1] != (self.landmark_size[2], self.landmark_size[3]):
            face_frame = cv2.resize(face, (self.landmark_size[3], self.landmark_size[2]))

        in_frame = face_frame.transpose((2, 0, 1))
        res = self.exec_landmark_net.infer(inputs={self.landmark_input_blob: in_frame})[self.landmark_out_blob].reshape(1, 10)[0]
        facial_landmarks = np.zeros((5, 2)) 
        for i in range(res.size // 2):
            normed_x = res[2 * i]
            normed_y = res[2 * i + 1]
            x_lm = face_frame.shape[1] * normed_x
            y_lm = face_frame.shape[0] * normed_y
            facial_landmarks[i] = (x_lm, y_lm)
        cv2.imshow('before_align', face)

        aligned_face = align_face(face, facial_landmarks)
        cv2.imshow('after_align', aligned_face)
        cv2.waitKey(10)
        if aligned_face.shape[:-1] != (self.fr_size[2], self.fr_size[3]):
            aligned_face_frame = cv2.resize(face, (self.fr_size[3], self.fr_size[2]))

        out_frame = aligned_face_frame.transpose((2, 0, 1))
        return out_frame


    def predict(self, face_frame):

        """ Description

        Runs the Face Reidentification model on the input image and provides a (1, 256, 1, 1) output feature vector

        :return: Returns the predicted face vector

        """
        res = self.exec_fr_net.infer(inputs={self.fr_input_blob: face_frame})[self.fr_out_blob]
        feature_vec = res.reshape(1, 256)
        return feature_vec