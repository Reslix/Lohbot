from camera import TrackingCameraRunner
from serial_io import SerialIO
from show import imshow
from flask_streaming_server import start_streaming_server

print("Initializing serial connection with Arduino")
ard = SerialIO()
ard.start()
print("Initializing camera")
c = TrackingCameraRunner(0)

# Send image to manager
manager = ImageManager(address=('', 11579), authkey=b'password')
ImageManager.register('get_dict')
try:
    manager.connect()
    print("Connected to manager.")
    manager.get_dict().update([('camera', camera)])
except ConnectionRefusedError:
    print("No connection to manager.")

print("Tracking Ball...")
tcenterx = 640
tradius = 30
im = None
while True:
    c.step_frame()
    center, radius, image = c.track_tennis_ball()
    #im = imshow(image, im=im)
    if center:
        differential = (tcenterx - center[0])//3
        distance = tradius - radius
        left = distance + differential
        left = max(-30, min(30, left))
        right = distance - differential
        right = max(-30, min(30, right))
        print(left,right)
        ard.direct(int(right), int(left))
        encoded = c.get_jpg()
        manager.get_dict().update([('encoded', encoded)])
        manager.get_dict().update([('state', 'follow - moving')])
    else:
        ard.stop()
        print("stop")
        manager.get_dict().update([('state', 'follow - stopping')])
