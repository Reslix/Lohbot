#!/usr/bin/env python3

from camera import TrackingCameraRunner
from show import imshow


if __name__ == "__main__":
    im = None
    
    c = TrackingCameraRunner(0)
    while True:    
        c.step_frame()            
        im = imshow(c.frame,im=im)
        c.capture_face('r1')

