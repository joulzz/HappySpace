import cv2

class FaceDetection:
    def __init__(self, path):
        self.cascade_classifier = cv2.CascadeClassifier(path)

    def run_facedetector(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.faces = self.cascade_classifier.detectMultiScale(gray, 1.3, 5)