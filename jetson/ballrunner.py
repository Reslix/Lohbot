from newcamera import TrackingCameraRunner
from serial_io import SerialIO
from show import imshow
import cv2
import math

print("Initializing serial connection with Arduino")
ard = SerialIO()
ard.start()
print("Initializing camera")
c = TrackingCameraRunner(0)

print("Tracking Ball...")
tcenterx = 640
tradius = 40
speed = 40
im = None
last = (0, cv2.getTickCount())
try:
    while True:
        c.step_frame()
        center, radius = c.track_tennis_ball()
        #im = imshow(c.frame, im=im)
        if center:
            # This should al lbe in cm...
            distance = 2131/(radius ** 1.02)
            horizontal = 3.5/radius * (tcenterx - center[0])
            oh = horizontal/distance
            if oh < -1:
                oh = -1
            elif oh > 1:
                oh = 1
            angle = math.asin(oh)
            print('distance: {}\nhorizontal: {}\nangle: {}'.format(distance, horizontal, angle)) 
            #TODO make sure angle is in degrees
            '''
            if angle < -.1:
                differential = 89.1 * angle + 43.7
            elif angle > .1:
                differential = 126 * angle - 28.9
            else:
                differential = 0
            differential *= 1
            '''
            differential = angle * 130
            deriv = (angle - last[0])/((cv2.getTickCount() - last[1])/cv2.getTickFrequency())
            print('deriv: {}'.format(deriv))
            last = (angle, cv2.getTickCount())
            differential = differential + 0.01 * deriv
            translate = (distance - 30)/4
            left = translate + differential
            left = max(-speed, min(speed, left))
            right = translate - differential
            right = max(-speed, min(speed, right))*1.2
            print(left,right)
            ard.direct(int(right), int(left))
        else:
            ard.stop()
            print("stop")
except KeyboardInterrupt:
    ard.stop()
    c.close()
    pass
