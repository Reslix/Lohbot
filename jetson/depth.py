import cv2
from show import imshow
from camera import Camera
import time
import numpy as np

cameraR = Camera(1)
cameraL = Camera(2)

cameraL.start()
cameraR.start()

stereo = cv2.StereoBM_create(16, 41)
im = None

while True:
    imL = cameraL.read_gray()[1]
    imR = cameraR.read_gray()[1]
    disp = np.array(stereo.compute(imL, imR)) 
    
    vis = np.concatenate((np.concatenate((imL,imR), axis=0), disp), axis=0)
    im = imshow(vis, im)
    time.sleep(.01)
