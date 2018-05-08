from threading import Thread

import cv2
import time
from facial import Faces

from balltrack import track_tennis_ball

delay = 30

face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')


class Camera:
    def __init__(self, id=0, height=1080, width=1920, fps=30):
        self.cap = cv2.VideoCapture(id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(w,h)
        self.success, self.image = self.cap.read()
        self.stopped = False

    def update(self):
        while True:
            if self.stopped:
                return

            self.success, image = self.cap.read()
            self.image = image[180:900, 320:1600]

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

class NonTrackingCameraRunner():
    def __init__(self, camera=0):
        self.camera = Camera(camera)
        self.camera.start()
        self.frame = None
        self.im = None

    def close(self):
        self.camera.release()
        cv2.destroyAllWindows()

    def step_frame(self):
        ret, frame = self.camera.read_rgb()
        if ret is True:
            self.frame = frame

    def track_face(self):
        return None, None

    def get_jpg(self):
        return cv2.imencode('.jpg', self.frame)[1].tostring()


class TrackingCameraRunner():
    """
    HAHAH FACIAL RECOGNITION IS BACK
    """

    def __init__(self, camera=0):
        self.tracker = cv2.TrackerKCF_create()
        self.tracking = False
        self.camera = Camera(camera)
        self.camera.start()
        self.frame = None
        #self.faces = Faces()
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
        if center is not None:
            cv2.circle(self.frame, (int(center[0]), int(center[1])), int(radius),
                   (0, 255, 255), 2)
            cv2.circle(self.frame, center, 5, (0, 0, 255), -1)
        return center, radius, image

    def step_frame(self):
        ret, frame = self.camera.read_rgb()
        if ret is True:
            self.frame = frame

    def track_face(self):
        if self.tracking:
            tracking, face = self.tracker.update(self.frame)
            face = [int(i) for i in face]
            self.tracking = tracking
            print('\r tracking',end="")
        else:
            rects = self.detect_faces()
            rects = sorted(rects, key=lambda x: x[2] * x[3], reverse=True)
            face = rects[0] if len(rects) else None
            if face is not None:
                self.tracker = cv2.TrackerKCF_create()
                self.tracker.init(self.frame, face)
                self.tracking = True
                print('\r reinitializing',end="")
            print('\r detecting', end="")
        if face is not None:
            cv2.rectangle(self.frame, (face[0], face[1]), (face[0] + face[2], face[1] + face[3]), (0, 255, 255), 2)
            # Andrew this is the face recog thing
            # self.faces.add_face(self.frame,face)
            # self.faces.track_faces()
        #else:
            # self.faces.detected_faces = []

        return face, []#self.faces.detected_faces

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

    def get_jpg(self):
        return cv2.imencode('.jpg', self.frame)[1].tostring()
