#!/usr/bin/env python3
import math, time
from multiprocessing import Manager, Process

from serial_io import SerialIO
from camera import TrackingCameraRunner
from flask_streaming_server import start_streaming_server
from flask_streaming_server import ImageManager
from show import imshow
import cv2
import fasteners


if __name__ == "__main__":

    #We have a single instance of our serial communicator
    ard = SerialIO()
    ard.start()

    # Initialize camera
    c = TrackingCameraRunner(0)
    camera = c.camera

    c.step_frame()

    # Shared Manager object
    image_dictionary = Manager().dict()
    ImageManager.register('get_dict', callable=lambda:image_dictionary)
    manager = ImageManager()
    manager.start()
    manager.get_dict().update([('camera', camera)])

    # Start web server to stream camera image
    p = Process(target=start_streaming_server, args=(manager,))
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

                manager.get_dict().update([('state', 'follow: moving')])
            else:
                ard.stop()
                manager.get_dict().update([('state', 'follow: stopping')])

            # Update mangager with shared image
            # encoded = cv2.imencode('.jpg', image)[1].tostring()
            # manager.get_dict().update([('encoded', encoded)])
        elif command == 'stop':
            ard.stop()
            print('stop')
            # encoded = camera.get_jpg()
            # manager.get_dict().update([('encoded', encoded)])
            manager.get_dict().update([('state', 'stopping')])
        elif command == 'openpose':
            #TODO no more
            print('Openpose')
            manager.get_dict().update([('state', 'Openpose')])
        else:
            print('undefined command')
            manager.get_dict().update([('state', 'undefined command')])
