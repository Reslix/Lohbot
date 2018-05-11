from threading import Thread

import cv2

from balltrack import track_tennis_ball

delay = 30


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

    def release(self):
        self.stopped = True
        self.cap.release()

    def get_jpg(self):
        return cv2.imencode('.jpg', self.image)[1].tostring()


class TrackingCameraRunner():

    def __init__(self, camera=0):
        self.tracker = cv2.TrackerKCF_create()
        self.tracking = False
        self.camera = Camera(camera)
        self.camera.start()
        self.frames = []
        self.frame = None
        #self.faces = Faces()
        self.im = None

    def close(self):
        self.camera.release()
        for i,frame in enumerate(self.frames):
            cv2.imwrite("frame{}.jpg".format(i), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.destroyAllWindows()


    def track_tennis_ball(self):
        center, radius = track_tennis_ball(self.frame)
        if center is not None:
            cv2.circle(self.frame, (int(center[0]), int(center[1])), int(radius),
                   (0, 255, 255), 2)
            cv2.circle(self.frame, center, 5, (0, 0, 255), -1)
        return center, radius

    def step_frame(self):
        if self.frame is not None:
            self.frames = self.frames +  [self.frame]
        ret, frame = self.camera.read_rgb()
        if ret is True:
            self.frame = frame
    
    def get_jpg(self):
        return cv2.imencode('.jpg', cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR))[1].tostring()
