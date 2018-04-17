from camera import PersonCameraRunner


runner = PersonCameraRunner()
while True:
    runner.step_frame()
    runner.step_imshow_frame()
