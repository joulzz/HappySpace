import cv2
import os
from openvino.inference_engine import IENetwork
from logging import log
import numpy as np

class FaceDetection:
    def __init__(self, plugin, model_xml):
        self.faces = []
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        # Create networks
        self.net = IENetwork(model=model_xml, weights=model_bin)
        log.info("Preparing input blobs")
        self.input_blob = next(iter(self.net.inputs))
        self.out_blob = next(iter(self.net.outputs))

        # Read and pre-process input images
        self.size = self.net.inputs[self.input_blob].shape

        log.info("Loading model to the plugin")
        self.exec_net = plugin.load(network=self.net, num_requests=2)

        self.images = np.ndarray(shape=(self.size))

        del self.net

    def preprocessing(self, frame):
        if frame.shape[:-1] != (self.size[2], self.size[3]):
            in_frame = cv2.resize(frame, (self.size[3], self.size[2]))
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        self.images[0] = in_frame

    def asyncCall(self, request_id):
        self.exec_net.start_async(request_id=request_id, inputs={self.input_blob: self.images})

    def awaitResults(self, request_id, frame):
        if self.exec_net.requests[self.request_id].wait(-1) == 0:
            res = self.exec_net.requests[request_id].outputs[self.out_blob]
            face_images_array = []
            coordinates_array = []
            emo_cur_request_id = 0
            self.face_count = 0
            # print("Face Detection Run Time: {} ms".format((time() - t0) * 1000))
            # res = res[out_blob]
            for detection in res[0][0].reshape(-1, 7):
                confidence = float(detection[2])
                xmin = int(detection[3] * frame.shape[1])
                ymin = int(detection[4] * frame.shape[0])
                xmax = int(detection[5] * frame.shape[1])
                ymax = int(detection[6] * frame.shape[0])

                if confidence > 0.5:
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=2)
                    face = frame[ymin:ymax, xmin:xmax]
                    self.faces.append(xmin, xmax, ymin, ymax)
                    print("Face Detection Confidence:", confidence)

        return True



