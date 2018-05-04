from threading import Thread

import cv2

from balltrack import track_tennis_ball
from show import imshow

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


class EnetCameraRunner():
    """
    Segments
    """
    pass


class PersonCameraRunner():
    """
    Identify human shaped things using
    """

    def __init__(self, camera=0):
        self.camera = Camera(camera)
        self.camera.start()
        self.frame = None
        self.im = None
        self.detector = detector.detector

    def step_frame(self):
        ret, frame = self.camera.read_rgb()
        if ret is True:
            self.frame = frame

    def step_imshow_frame(self):
        rects, image = self.detector(self.frame)
        for (x, y, w, h) in rects:
            cv2.rectangle(image, (x, y), (w, h), (0, 255, 255), 2)
        self.im = imshow(image, im=self.im)

    def track_people(self):
        """
        We obtain the bounding boxes of all the people in the scene and update their positions.
        We do this by tracking the location of the centers of their boxes and comparing them with the locations of the
        previous frame with a small prediction.
        We have a vector of the x and y coordinates of the old and new centers.
        :return:
        """
        pass


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
        self.im = None

    def close(self):
        self.camera.release()
        cv2.destroyAllWindows()

    def track_tennis_ball(self):
        center, radius, image = track_tennis_ball(self.frame)
        self.im = imshow(image, self.im)
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
            print('tracking')
        else:
            rects = self.detect_faces()
            rects = sorted(rects, key=lambda x: x[2] * x[3], reverse=True)
            face = rects[0] if len(rects) else None
            if face is not None:
                self.tracker = cv2.TrackerKCF_create()
                self.tracker.init(image, face)
                self.tracking = True
                print('reinitializing')
            print('detecting')

        return face, image

    def detect_faces(self):
        gray = cv2.cvtColor(self.frame, cv2.COLOR_RGB2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.5, 5, minSize=(80, 80))
        rects = []
        for (x, y, w, h) in faces:
            x = max(0,x - w // 6)
            y = max(0,y - h // 4)
            w = min(w + w // 3,1280)
            h = min(h + h // 2,720)
            eyes = eye_cascade.detectMultiScale(gray[y:y + h, x:x + w])
            if eyes is not None and len(eyes) > 0:
                rects.append((x, y, w, h))

        return rects
