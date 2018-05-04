#!/usr/bin/env python3
import math, time
from threading import Thread

from flask import Flask, render_template, Response

from serial_io import SerialIO
from camera import TrackingCameraRunner
from show import imshow
import cv2
import fasteners

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


def gen(camera):
    while True:
        frame = camera.get_jpg()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == "__main__":



    #We have a single instance of our serial communicator
    ard = SerialIO()
    ard.start()

    c = TrackingCameraRunner(0)
    camera = c.camera
    Thread(target=app.run(), args=('localhost','80')).start()
    im = None
    tcenterx = 640
    tsize = 160
    while True:
        with fasteners.InterProcessLock('ALEXA_COMMAND.txt.lock'):
            with open('ALEXA_COMMAND.txt') as file:
                command = file.read().strip()
        if command == 'follow':
            c.step_frame()
            rect, image = c.track_face()
            if rect is not None:
                cv2.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 255), 2)
                im = imshow(image, im=im)

                center = (rect[0]+rect[2]//2, rect[1]+rect[3]//2)
                size = math.sqrt(rect[2]**2+rect[3]**2)

                differential = (tcenterx - center[0]) // 3
                distance = tsize - size
                left = distance + differential
                right = distance - differential
                ard.direct(int(left), int(right))
            else:
                ard.stop()
                im = imshow(image, im=im)
        elif command == 'stop':
            ard.stop()
        elif command == 'openpose':
            #TODO
            pass
        else:
            print('undefined command')

