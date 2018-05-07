from threading import Thread

import cv2
from facial import Faces

from balltrack import track_tennis_ball

delay = 30

face_cascade = cv2.CascadeClassifier('cascades_cuda/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('cascades_cuda/haarcascade_eye.xml')


class Camera:
    def __init__(self, id=0, height=720, width=1280, fps=30):
        self.cap = cv2.VideoCapture(id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.success, self.image = self.cap.read()
        self.stopped = False
        # self.calib = pickle.load('calibration.pickle')
        # self.newcameramtx, self.roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(width,height),1,(width,height))

    def update(self):
        while True:
            if self.stopped:
                return

            self.success, image = self.cap.read()
            self.image = image

    def start(self):
        Thread(target=self.update, args=()).start()

    def read_rgb(self):
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        return self.success, image

    def undistort(self, image):
        return cv2.undistort(image, self.calib[1])

    def release(self):
        self.stopped = True
        self.cap.release()

    def get_jpg(self):
        return cv2.imencode('.jpg', self.image)[1].tostring()

class TrackingCameraRunner():
    """
    Deprecated, facial recognition is not a priority atm
    """

    def __init__(self, camera=0):
        self.tracker = cv2.TrackerKCF_create()
        self.tracking = False
        self.camera = Camera(camera)
        self.camera.start()
        self.frame = None
        self.faces = Faces()
        self.im = None

    def close(self):
        self.camera.release()
        cv2.destroyAllWindows()

    def capture_face(self, name):
        coord = self.track_face()
        if coord is not None:
            self.faces.add_face(self.frame,coord)
            self.faces.single_face_capture(name)

    def track_tennis_ball(self):
        center, radius, image = track_tennis_ball(self.frame)
        cv2.circle(self.camera.image, (int(x), int(y)), int(radius),
                   (0, 255, 255), 2)
        cv2.circle(self.camera.image, center, 5, (0, 0, 255), -1)
        return center, radius, image

    def step_frame(self):
        ret, frame = self.camera.read_rgb()
        if ret is True:
            self.frame = frame

    def track_face(self):
        image = self.frame.copy()
        if self.tracking:
            tracking, face = self.tracker.update(image)
            face = [int(i) for i in face]
            self.tracking = tracking
            print('\r tracking',end="")
        else:
            rects = self.detect_faces()
            rects = sorted(rects, key=lambda x: x[2] * x[3], reverse=True)
            face = rects[0] if len(rects) else None
            if face is not None:
                self.tracker = cv2.TrackerKCF_create()
                self.tracker.init(image, face)
                self.tracking = True
                print('\r reinitializing',end="")
            print('\r detecting', end="")
        if face is not None:
            cv2.rectangle(self.camera.image, (face[0], face[1]), (face[0] + face[2], face[1] + face[3]), (0, 255, 255), 2)

        return face

    def detect_faces(self):
        gray = cv2.cvtColor(self.frame, cv2.COLOR_RGB2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.5, 5, minSize=(80, 80))
        rects = []
        for (x, y, w, h) in faces:
            x = max(0,x - w // 6)
            y = max(0,y - h // 4)
            w = min(w + w // 3,1200)
            h = min(h + h // 2,719)
            eyes = eye_cascade.detectMultiScale(gray[y:y + h, x:x + w])
            if eyes is not None and len(eyes) > 0:
                rects.append((x, y, w, h))

        return rects

    def open_pose(self):
        """
        The api for pyopenpose is completely borked
        :return:
        """
        pass
