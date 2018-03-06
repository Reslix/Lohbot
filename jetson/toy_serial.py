#!/usr/bin/env python

import serial
import time

ser = serial.Serial('/dev/ttyACM0', 9600)
print(ser.name)
commands = [b's', b'f', b'b', b'l', b'r', b'x', b'y', b'z']
print("start up {}".format(ser.read(1)))
for c in commands:
    print('ser.write(): {}'.format(ser.write(c)))
    print("ser.read(): {}".format(ser.read(1)))
    time.sleep(1)


