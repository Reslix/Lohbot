"""
This model is still ambiguous, but we want to identify individuals as well as their poses. We are most interested in
our primary target's pose and location, of course. The
"""
import cv2
import imutils
from imutils.object_detection import non_max_suppression
import numpy as np


class PersonDetector():
    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect_person(self, image):

        image = imutils.resize(image, width=min(400, image.shape[1]))
        (rects, weights) = self.hog.detectMultiScale(image, winStride=(4, 4),
                                                padding=(8, 8), scale=1.05)
        # apply non-maxima suppression to the bounding boxes using a
        # fairly large overlap threshold to try to maintain overlapping
        # boxes that are still people
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
