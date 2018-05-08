#!/usr/bin/env python3
import math, time
from datetime import datetime
from multiprocessing import Manager, Process

from serial_io import SerialIO
from camera import NonTrackingCameraRunner, TrackingCameraRunner
from flask_streaming_server import start_streaming_server
from manager import ImageManager
from show import imshow
import cv2
import fasteners


if __name__ == "__main__":

    #We have a single instance of our serial communicator
    ard = SerialIO()
    ard.start()

    # Initialize camera
    c = TrackingCameraRunner(0)
    # c = NonTrackingCameraRunner(0)
    camera = c.camera

    c.step_frame()

    # Send image to manager
    manager = ImageManager(address=('', 11579), authkey=b'password')
    ImageManager.register('get_dict')
    try:
        manager.connect()
        print("Connected to manager.")
        manager.get_dict().update([('camera', camera)])
    except ConnectionRefusedError:
        print("No connection to manager.")

    im = None
    tcenterx = 640
    tsize = 160

    prevtime = datetime.now()
    while True:
        newtime = datetime.now()
        difference = newtime-prevtime
        print("%d %d", difference.seconds, difference.microseconds)
        prevtime = newtime
        c.step_frame()
        # im = imshow(camera.image,im=im)
        with fasteners.InterProcessLock('ALEXA_COMMAND.txt.lock'):
            with open('ALEXA_COMMAND.txt') as file:
               command = file.read().strip()
        if command == 'follow':
            rect, faceObj = c.track_face()
            if (faceObj is not None) and (len(faceObj) != 0):
                print(faceObj[0].name)
                manager.get_dict().update([('name', faceObj[0].name)])
            else:
                manager.get_dict().pop('name', None)
            if rect is not None:
                center = (rect[0]+rect[2]//2, rect[1]+rect[3]//2)
                size = math.sqrt(rect[2]**2+rect[3]**2)
                differential = (tcenterx - center[0]) // 3
                left = differential
                left = max(-30, min(30, left))
                right = -left
                print('({}, {})'.format(left, right))
                ard.direct(int(right), int(left))

                manager.get_dict().update([('state', 'follow - moving')])
            else:
                ard.stop()
                manager.get_dict().update([('state', 'follow - stopping')])

            # Update mangager with shared image
            #encoded = c.get_jpg()
            #manager.get_dict().update([('encoded', encoded)])
        elif command == 'stop':
            ard.stop()
            print('stop')
            #encoded = c.get_jpg()
            #manager.get_dict().update([('encoded', encoded)])
            manager.get_dict().update([('state', 'stopping')])
            manager.get_dict().pop('name', None)
        elif command == 'openpose':
            #TODO no more
            print('Openpose')
            manager.get_dict().update([('state', 'Openpose')])
        else:
            print('undefined command')
            manager.get_dict().update([('state', 'undefined command')])
