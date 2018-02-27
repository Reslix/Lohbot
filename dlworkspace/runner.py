from camera import CameraRunner

delay = 30

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
