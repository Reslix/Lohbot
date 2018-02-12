"""
Here we have the model or models that deal with facial recognition. This class will be more involved as it will
keep track of identified and unidentified persons. Each individual has a unique feature embedding, and this
class will keep a reference database between identities and embeddings.

We will likely be using FaceNet, which is a siamese neural network that can idenitfy people with a single image.
Multiple images are still required for training, but we can simultaneously classify persons whose identities we are
certain of as well as prepare training data. (probably remotely)

"""
import torch
import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')

class Face(object):

    def __init__(self, image, bounds, eyes):
        """
        We initalize a face object with these parameters because this object is only used to track in-environment faces.

        :param image: image is the extracted face photo in rgb color and put in a list, with subsequent images being
            used for training and such for identification. This will populate to different levels depending on
            whether or not the person is already in the face database.
        :param bounds: bounds is a tuple with (x,y,w,h)
        :param eyes: the coordinates of the eyes in [(x,y),(x,y),...(if they somehow have more...ew)]
        """
        self.images = [image]
        self.bounds = bounds
        self.eyes = eyes


class Faces(object):

    def __init__(self):
        self.faces = []
        self.current = []

    def detect_in_current_frame(self, image):
        """
        Returns faces
        :param image:
        :return:
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)

            self.current.append(Face(image[x:x+w, y:y+h], (x,y,w,h), eyes))

        for (x,y,w,h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 1)

    def track_faces(self):
        """
        After detecting the faces in a current frame, we need to match them with existing faces or add them to the
        database.
        :return:
        """
        pass


class FaceNet(object):
    """
    Inspired by FaceNet, we implement a scale-invariant siamese deep network used to
    """

    def forward(self, model, batch):
        out = model(batch)
        return out

    def load_model(self):
        pass


def train_model(name="facemodel"):
