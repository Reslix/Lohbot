import math, time

from serial_io import SerialIO
from camera import CameraRunner

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


delay = 30

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

    c = CameraRunner(1)
    while True:
        c.step_frame()
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
                ard.write(b'r')
                time.sleep(0.5)
                print('And stopping')
                ard.write(b's')

            elif center[0] < width*left_threshold:
                print('Turn left')
                ard.write(b'l')
                time.sleep(0.5)
                print('And stopping')
                ard.write(b's')

            else:
                if (radius < ball_radius_min):
                    print('Move forward')
                    ard.write(b'f')

                elif (radius > ball_radius_max):
                    print('Move backwards')
                    ard.write(b'b')

                else:
                    print('Not moving')
                    ard.write(b's')

        else:
            print("No ball found")
            ard.write(b's')

        time.sleep(0.5)


    """
    c = CameraRunner()
    print(args)
    if args.n is None:
        for i in range(delay):
            c.step_frame()
            c.prepare_face_capture(i)

        while True:
            c.step_frame()
            c.face_recog()

    if args.n is not None:
        for i in range(delay):
            c.step_frame()
            c.prepare_face_capture(i)

        for i in range(args.frames):
            c.step_frame()
            c.capture(i, args.n)

        c.close()
        
    """

