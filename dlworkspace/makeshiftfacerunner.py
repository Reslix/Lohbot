from dlworkspace.facialrecognition import Faces
import dlworkspace.facialrecognition as fr

import cv2

class CameraRunner():

    def __init__(self):
        self.faces = Faces()
        self.capture = cv2.VideoCapture(0)
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.frame = None

        self.faces.set_model(fr.load_model())

    def close(self):
        self.faces.save_model()
        self.faces.save_identities()
        self.faces.save_identities()
        self.capture.release()
        cv2.destroyAllWindows()
        del self.faces

    def step_frame(self):
        ret,frame = self.capture.read()
        if ret is True:
            self.frame = frame

    def facial(self):
        self.faces.detect_in_current_frame(self.frame)
        self.faces.track_faces()


    def crude_positioning(self):


