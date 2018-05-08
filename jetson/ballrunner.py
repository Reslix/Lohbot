from camera import TrackingCameraRunner
from serial_io import SerialIO


print("Initializing serial connection with Arduino")
ard = SerialIO()
ard.start()
print("Initializing camera")
c = TrackingCameraRunner(0)

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
        left = max(-30, min(30, left))
        right = distance - differential
        right = max(-30, min(30, right))
        print(left,right)
        ard.direct(int(right), int(left))
    else:
        ard.stop()
