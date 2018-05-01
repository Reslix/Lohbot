from serial_io import SerialIO
import sys
import time

i = int(sys.argv[1])
ard = SerialIO()
ard.start()

current_milli_time = lambda: int(round(time.time() * 1000))

t_end = current_milli_time() + 250
print('begin turn')
while current_milli_time() < t_end:
    ard.direct(-i,i)
print('end turn')
ard.stop()
