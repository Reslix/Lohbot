import math, time

from serial_io import SerialIO
from camera import TrackingCameraRunner
from show import imshow
import cv2

"""
The object structure will be as follows: 
    
    Motion()
        Serial()
        Map()
            Mapper()
                Various Sensors()
                    Serial()
                    Segmenter()
                    Object()
                    Depth()
    CameraRunner()
        Face()
        Pose()
        Object()
        Depth()
        Segmenter()
    Alexa()
    ...
    ...

    Ideally we have all the classes declared in here so we can reduce redundancy.
    
"""


# (0, 0) is top left
# If ball if past left_threshold (perentage of camera screen), move left. Between 0 and 1.
left_threshold = 0.35
# If ball is past right_threshold (perentage of camera screen), move right. Between 0 and 1.
right_threshold = 0.65
# Move forward if ball radius < this value
ball_radius_min = 20.0
# Move backwards if ball radius > this value
ball_radius_max = 30.0

if __name__ == "__main__":

    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Deals with all camera related stuff')
    parser.add_argument('-n', required=False, action="store",
                        help='Name of person')
    parser.add_argument('--frames', required=False,
                        metavar="Number of frame captures", default=100,
                        help='Number of frame captures')
    args = parser.parse_args()

    #We have a single instance of our serial communicator
    ard = SerialIO()
    ard.start()

    c = TrackingCameraRunner(0)
    im = None
    tcenterx = 640
    tsize = 160
    while True:
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

        """
        center, radius, image = c.track_tennis_ball()
        # print(str(image.shape[0]))
        height = image.shape[0]
        width = image.shape[1]

        # First, turn if necessary
        # Move forward if far away, move backwards if too close
        if center:
            print('center: ' + str(center) + ' radius: ' + str(radius))
            if center[0] > width*right_threshold:
                print('Turn right')
                ard.right()
                time.sleep(0.5)
                print('And stopping')
                ard.stop()

            elif center[0] < width*left_threshold:
                print('Turn left')
                ard.left()
                time.sleep(0.5)
                print('And stopping')
                ard.stop()

            else:
                if (radius < ball_radius_min):
                    print('Move forward')
                    ard.forward()

                elif (radius > ball_radius_max):
                    print('Move backwards')
                    ard.backward()

                else:
                    print('Not moving')
                    ard.stop()

        else:
            print("No ball found")
            ard.stop()

        time.sleep(0.5)
        """


