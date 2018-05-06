#!/usr/bin/env python3
import math, time
from multiprocessing import Process

from serial_io import SerialIO
from camera import TrackingCameraRunner
from flask_streaming_server import start_streaming_server
from show import imshow
import cv2
import fasteners


if __name__ == "__main__":

    c = TrackingCameraRunner(0)
    camera = c.camera
    c.step_frame()
    # Start web server to stream camera image
    p = Process(target=start_streaming_server, args=(camera,))
    p.start()

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

                print('Moving')
            else:
                print('Stopping')
                im = imshow(image, im=im)
        elif command == 'stop':
            print('stop')
        elif command == 'openpose':
            print('Openpose')
            pass
        else:
            print('undefined command')
        time.sleep(1)
