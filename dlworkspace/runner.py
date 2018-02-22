import time
from facial import Faces, imshow

import cv2

delay = 30
class CameraRunner():
    def __init__(self):
        self.faces = Faces()
        self.cap = cv2.VideoCapture(1)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.cap.set(cv2.CAP_PROP_FPS, 15)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
        print(self.cap.get(cv2.CAP_PROP_FOURCC))
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.frame = None
        self.im = None
        print(self.width, self.height)

    def close(self):
        self.faces.save_model()
        self.faces.save_identities()
        self.faces.save_identities()
        self.cap.release()
        cv2.destroyAllWindows()
        del self.faces

    def step_frame(self):
        ret, frame = self.cap.read()
        if ret is True:
            self.frame = frame

    def capture(self, i, name):
        image = self.faces.detect_in_current_frame(self.frame)
        for face in self.faces.detected_faces:
            cv2.putText(image, str(i), (face.bounds[0], face.bounds[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0),
                        2, cv2.LINE_AA)
        self.im = imshow(image, im=self.im)
        self.faces.single_face_capture(name)

    def prepare_capture(self, i):
        image = self.faces.detect_in_current_frame(self.frame)
        for face in self.faces.detected_faces:
            cv2.putText(image, "Prepare for Capture " + str(delay - i), (face.bounds[0], face.bounds[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0),
                        2, cv2.LINE_AA)
        self.im = imshow(image, im=self.im)

    def facial(self):
        image = self.faces.detect_in_current_frame(self.frame)
        self.faces.track_faces()

        for face in self.faces.detected_faces:
            if face.name is not "":
                cv2.putText(image, face.name, (face.bounds[0], face.bounds[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0),
                            2, cv2.LINE_AA)
        self.im = imshow(image, im=self.im)

    def crude_positioning(self):
        pass


if __name__ == "__main__":

    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Deals with all camera related stuff')
    parser.add_argument('-n', required=False, action="store",
                        help='Name of person')
    parser.add_argument('--frames', required=False,
                        metavar="Number of frame captures", default=100,
                        help='Number of frame captures')
    args = parser.parse_args()

    c = CameraRunner()
    print(args)
    if args.n is None:
        for i in range(delay):
            c.step_frame()
            c.prepare_capture(i)
        
        while True:
            c.step_frame()
            c.facial()

    if args.n is not None:
        for i in range(delay):
            c.step_frame()
            c.prepare_capture(i)

        for i in range(args.frames):
            c.step_frame()
            c.capture(i, args.n)

        c.close()
