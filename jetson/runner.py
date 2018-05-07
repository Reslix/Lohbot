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

    #We have a single instance of our serial communicator
    ard = SerialIO()
    ard.start()

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
        c.step_frame()
        with fasteners.InterProcessLock('ALEXA_COMMAND.txt.lock'):
            with open('ALEXA_COMMAND.txt') as file:
                command = file.read().strip()
        if command == 'follow':
            rect = c.track_face()
            if rect is not None:

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

