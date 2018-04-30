from camera import FaceCameraRunner
from serial_io import SerialIO


print("Initializing serial connection with Arduino")
ard = SerialIO()
ard.start()
print("Initializing camera")
c = FaceCameraRunner(1)

print("Tracking Ball...")
tcenterx = 640
tradius = 30
while True:
    c.step_frame()
    center, radius, image = c.track_tennis_ball()
    if center:
        differential = (tcenterx - center[0])//3
        distance = tradius - radius
        left = distance + differential
        right = distance - differential
        ard.direct(int(left), int(right))
    else:
        ard.stop()
