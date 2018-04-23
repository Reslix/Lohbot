from camera import FaceCameraRunner
from serial_io import SerialIO


print("Initializing serial connection with Arduino")
ard = SerialIO()
ard.start()
print("Initializing camera")
c = CameraRunner(0)

print("Tracking Ball...")
tcenterx = 640
tradius = 30
while True:
    c.step_frame()
    center, radius, image = c.track_tennis_ball()
    differential = (tcenterx - center[0])/3
    distance = tradius - radius
    left = distance + differential
    right = distance - differential
    ard.direct(left, right)