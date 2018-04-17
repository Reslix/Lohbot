#!/usr/bin/env python3

#script to (kind of test) python serial comms

from serial_io import SerialIO

import random
import time

r = random.Random()
ard = SerialIO()

ard.start()

print(ard.read('left'))

dirs = ['left', 'right', 'middle']

while 1:
    # might need struct packing for actual data
    '''
    print('left: {}\tright: {}\tmiddle: {}'.format(
    ard.read('left')[0],
    ard.read('right')[0],
    ard.read('middle')[0])
    )
    '''
    ard.forward()
    #print('trying to go forward')
    '''
    if ard.check == b'E':
        ard.stop()
        print('stopping because error {}'.format(ard.check))
    '''
    print('ard.check: {}'.format(ard.check_return()))
    time.sleep(0.5)
    ard.stop()
