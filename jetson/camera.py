from threading import Thread

import cv2

from balltrack import track_tennis_ball
from show import imshow

from facial import Faces

delay = 30


class Camera:
    def __init__(self, id=0, height=720, width=1280, fps=30):
        self.cap = cv2.VideoCapture(id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        self.success, self.image = self.cap.read()
        self.stopped = False

    def update(self):
        while True:
            if self.stopped:
                return

            self.success, image = self.cap.read()
            self.image = cv2.flip(image, -1)

    def start(self):
        Thread(target=self.update, args=()).start()

    def read_rgb(self):
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        return self.success, image

    def release(self):
        self.stopped = True
        self.cap.release()


class CameraRunner():
    def __init__(self, camera=0):
        self.faces = Faces()
        self.camera = Camera(camera)
        self.camera.start()
        self.frame = None
        self.im = None

    def close(self):
        self.faces.save_model()
        self.faces.save_identities()
        self.faces.save_identities()
        self.camera.release()
        cv2.destroyAllWindows()
        del self.faces

    def track_tennis_ball(self):
        center, radius, image = track_tennis_ball(self.frame)
        self.im = imshow(image, self.im)
        return center, radius, image

    def step_frame(self):

        ret, frame = self.camera.read_rgb()
        if ret is True:
            self.frame = frame

    def capture(self, i, name):
        image = self.faces.detect_in_current_frame(self.frame)
        for face in self.faces.detected_faces:
            cv2.putText(image, str(i), (face.bounds[0], face.bounds[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0),
                        2, cv2.LINE_AA)
        self.im = imshow(image, im=self.im)

        self.faces.single_face_capture(name)

    def prepare_face_capture(self, i):
        #note, face detection takes up 45% of the runtime of this function
        image = self.faces.detect_in_current_frame(self.frame)
        for face in self.faces.detected_faces:
            cv2.putText(image, "Prepare for Capture " + str(delay - i), (face.bounds[0], face.bounds[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0),
                        2, cv2.LINE_AA)
        #note, this part takes up 45% of the runtime of the function
        self.im = imshow(image, im=self.im)

    def face_recog(self):
        image = self.faces.detect_in_current_frame(self.frame)

        # this component takes between 1 and 33% of the runtime of this function, .01 to .1 seconds
        self.faces.track_faces()

        for face in self.faces.detected_faces:
            if face.name is not "":
                cv2.putText(image, face.name, (face.bounds[0], face.bounds[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0),
                            2, cv2.LINE_AA)
        self.im = imshow(image, im=self.im)

    def estimate_pose(self):
        pass

    def crude_positioning(self):
        pass
