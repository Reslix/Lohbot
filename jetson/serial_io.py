#!/usr/bin/env python3

import serial
import threading
from threading import Thread

class SerialIO():

    # methods we should call to move the robot around
    # should probably go in a wrapper class but that 
    # would be too enterprise for us

    def forward(self, speed = 48):
        self.write(b'f', bytes([speed]))
                
    def backward(self, speed = 48):
        self.write(b'b', bytes([speed]))

    def left(self, speed = 48):
        self.write(b'l', bytes([speed]))

    def right(self, speed = 48):
        self.write(b'r', bytes([speed]))

    # Direct drive the motors, 
    # left and right are the speeds of the motors
    # and are in the range [-255, 255] and are integers
    def direct(self, left = 48, right = 48):
        # decode the directions based off the sign of the inputs
        dirs = 0
        dirs += (left >= 0) 
        dirs += (right >= 0) << 1

        # arduino no like negative numbers
        if left < 0:
            left *= -1
        if right < 0:
            right *= -1

        self.lock.acquire()
        self.buffer[0] = b'd'
        self.buffer[1] = bytes([dirs])
        self.buffer[2] = bytes([left])
        self.buffer[3] = bytes([right])
        self.lock.release()

    def stop(self):
        self.write(b's')


    def __init__(self, com=None, baud=115200, delay=10):
        # initialize data structure to keep shizz
        # put lockfile, write buffer
        self.baud = baud
        self.delay = delay
        self.running = 1
        # buffer is [command, speed]
        self.buffer = [0, 0, 0, 0]
        self.check = 0
        self.lock = threading.RLock()
        self.distances = {'left': 201,'right': 201,'middle': 201} # start off with error values for distance

        # better than silently failing
        self.ser = serial.Serial('/dev/ttyACM1', self.baud, timeout = self.delay)
        # wait for startup time
        # Toggle DTR to reset Arduino
        self.ser.setDTR(False)
        # toss any data already received, see
        # http://pyserial.sourceforge.net/pyserial_api.html#serial.Serial.flushInput
        self.ser.flushInput()
        self.ser.setDTR(True)
        print(self.ser.read())

    def start(self):
        # start the thread that constantly does serial reading
        # if the serial port doesn't exist the thing crashes sometimes
        Thread(target=self.update, args=()).start()

    def read(self, dir):
        """
        dir = one of  'left', 'right', 'middle'
        """
        # returns the data structure, be sure to check lockfile
        return self.distances[dir]
        
    def check_return(self):
        # Return current check (write back from arudino)
        self.lock.acquire()
        c = self.check
        self.lock.release()
        return c

    def write(self, m, speed = 48):
        """
        One of b'f', b'b', b's', b'l', b'r'
        second arguement is the motor speed (optional)
        """

    	# Don't want to let the user poll the sensors themselves
        if m not in [b'x', b'y', b'z']:
            self.lock.acquire()
            self.buffer[0] = m
            self.buffer[1] = speed
            self.lock.release()

        # sends the write buffer, be sure to check lockfile. Returns delay between when the write was sent and when
        # this was called.

    def update(self):
        while self.running:
            # getting distances via polling now
            # or not because reasons
            '''
            self.ser.write(b'x')
            self.distances['left'] = self.ser.read()
            self.ser.write(b'y')
            self.distances['middle'] = self.ser.read()
            self.ser.write(b'z')
            self.distances['right'] = self.ser.read()
            '''

            # do the handshakes to read, write if necessary, then delay
            self.lock.acquire()
            if self.buffer[0]:
                # Write command then speed
                self.ser.write(self.buffer[0])
                # print('writing: {}'.format(self.buffer[0]))
                if self.buffer[0] in [b'f', b'b', b'l', b'r']:
                    self.ser.write(self.buffer[1])
                    # print('also writing: {}'.format(self.buffer[1][0]))
                # wait for arduino to write back
                if self.buffer[0] == b'd':
                    self.ser.write(self.buffer[1])
                    self.ser.write(self.buffer[3])
                    self.ser.write(self.buffer[3])
                c = self.ser.read(1)
                self.check = c
                # jenky way to check if message in buffer
                self.buffer[0] = 0
            self.lock.release() 
            


