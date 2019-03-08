from em_model import EMR
import cv2
import numpy as np
from sklearn.externals import joblib
import os
from openvino.inference_engine import IENetwork

class SmileDetector:
    def __init__(self, plugin, emo_model_xml):
        self.emotions_list = ["neutral", "happy", "sad", "surprise", "anger"]
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



    def preprocess_image(self, face):
        if face.shape[:-1] != (self.emo_size[2], self.emo_size[3]):
            try:
                face_frame = cv2.resize(face, (self.emo_size[3], self.emo_size[2]))
            except:
                print("Exception Raised in Resizing Image")
            face_frame = face_frame.transpose((2, 0, 1))

        face_frame = face_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        return face_frame

    def predict(self, face_frame):

        res = self.exec_emo_net.infer(inputs={self.emotion_input_blob: face_frame})
        res = res[self.emotion_out_blob]
        for i, emotion_out in enumerate(res.reshape(-1, 5)):
            emotion_idx = int(np.argmax(emotion_out))
            print("Iteration number:", i, "Confidence:", emotion_out[emotion_idx])
            if emotion_out[emotion_idx] > 0.6:
                emotion = self.emotions_list[emotion_idx]
                if emotion == 'happy':
                    return True
                else:
                    return False


        # result = self.network.predict(self.image)
        # output = self.final_layer.predict(result)
        # if output:
        #     return False
        # else:
        #     return True