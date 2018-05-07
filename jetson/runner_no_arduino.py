#!/usr/bin/env python3
import math, time
from multiprocessing import Manager, Process

from serial_io import SerialIO
from camera import TrackingCameraRunner
from flask_streaming_server import start_streaming_server
from manager import ImageManager
from show import imshow
import cv2
import fasteners


if __name__ == "__main__":

    # Initialize camera
    c = TrackingCameraRunner(0)
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
        print("No connection to  manager.")

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
                print('Moving')
                manager.get_dict().update([('state', 'follow: moving')])
            else:
                print('Stopping')
                manager.get_dict().update([('state', 'follow: stopping')])

        elif command == 'stop':
            print('stop')
            manager.get_dict().update([('state', 'stopping')])
        elif command == 'openpose':
            #TODO no more
            print('Openpose')
            manager.get_dict().update([('state', 'Openpose')])
        else:
            print('undefined command')
            manager.get_dict().update([('state', 'undefined command')])

        time.sleep(1)
