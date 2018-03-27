#!/usr/bin/env python3

import serial
import time

ser = serial.Serial('/dev/ttyACM0', 9600)
print(ser.name)
commands = [b's', b'f', b'b', b'l', b'r', b'x', b'y', b'z']
print("start up {}".format(ser.read(1)))
for c in commands:
    if c == b'f':
        ser.forward(10)
        print(ser.read(1))
    elif c == b'b':
        ser.backward(10)
        print(ser.read(1))
    elif c == b'r':
        ser.right(10)
        print(ser.read(1))
    elif c == b'l':
        ser.left(10)
        print(ser.read(1))
    else:
        print(ser.read(c[0]))
    time.sleep(1)


