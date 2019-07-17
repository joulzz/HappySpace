import cv2
import os
from openvino.inference_engine import IENetwork
from logging import log
import numpy as np

class FaceDetection:
    """
       Class used to detect faces using the face-detection-retail-0004 model

       Attributes
       ----------
       net:
        Face Detection network description that is sent to plugin to load
       input_blob:
        The name of the node in the network where the image is passed as input
       out_blob:
        The name of the node in the network where the result vector is received as output
       size:
        Size of the image expected at the input node
       exec_net:
        The in-memory network loaded that executes a forward pass over the network

    """
    def __init__(self, plugin, model_xml):

        """ Description

        :param plugin: Inference Engine Plugin
        :param model_xml: XML path of the face detection xml description

        """
        self.faces = []
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        # Create networks
        self.net = IENetwork(model=model_xml, weights=model_bin)
        print("Preparing input blobs")
        self.input_blob = next(iter(self.net.inputs))
        self.out_blob = next(iter(self.net.outputs))

        # Read and pre-process input images
        self.size = self.net.inputs[self.input_blob].shape

        print("Loading model to the plugin")
        self.exec_net = plugin.load(network=self.net, num_requests=2)

        self.images = np.ndarray(shape=(self.size))

        del self.net

    def preprocessing(self, frame):

        """ Description

        Preprocess the image by resizing and transposing as required by the Face Detection network in IE format

        """
        if frame.shape[:-1] != (self.size[2], self.size[3]):
            in_frame = cv2.resize(frame, (self.size[3], self.size[2]))
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        self.images[0] = in_frame

    def asyncCall(self, request_id):

        """ Description

        Function to start Asynchronour request with the provided request id.

        :param request_id: Used to receive either the current or next request id from the main script dependent on the
        is_async_mode in the main script

        """
        self.exec_net.start_async(request_id=request_id, inputs={self.input_blob: self.images})

    def awaitResults(self, request_id, frame):

        """ Description

        Function awaits results from the asynchronous call to populate the detected faces as a part of the current_frame_bboxes

        :param request_id: Request ID currently awaiting the network's results
        :param frame: Takes in the frame used for drawing
        :return: Returns True if the request is satisfied to populate person data
        """
        if self.exec_net.requests[request_id].wait(-1) == 0:
            res = self.exec_net.requests[request_id].outputs[self.out_blob]
            face_images_array = []
            coordinates_array = []
            emo_cur_request_id = 0
            self.face_count = 0
            self.faces=[]
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
                    self.faces.append([(xmin, ymin), (xmax, ymax)])
                    print("Face Detection Confidence:", confidence)

        return True



