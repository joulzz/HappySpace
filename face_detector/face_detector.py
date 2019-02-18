import cv2

class FaceDetection:
    def __init__(self, path):
        self.faces = []
        self.cascade_classifier = cv2.CascadeClassifier(path)

    def run_facedetector(self, image, min_size, max_size):
        self.faces = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if len(max_size) != 0:
            faces = self.cascade_classifier.detectMultiScale(gray, 1.3, 5, minSize=tuple(min_size), maxSize=tuple(max_size))
        else:
            faces = self.cascade_classifier.detectMultiScale(gray, 1.3, 5, minSize=tuple(min_size))

        for face in faces:
            self.faces.append(((face[0], face[1]), (face[0] + face[2], face[1] + face[3])))
