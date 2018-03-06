#!/usr/bin/env python3

import serial
import threading

class SerialIO():

    def __init__(self, com=None, baud=9600, delay=10):
        # initialize data structure to keep shizz
        # put lockfile, write buffer
        self.baud = baud
        self.delay = delay
        self.running = 1
        self.buffer = 0;
        self.lock = thread.RLock()
        self.distances = {'left': 201,'right': 201,'left': 201} # start off with error values for distance

        # better than silently failing
        self.ser = serial.Serial('/dev/ttyACM0', self.baud, timeout = self.delay)

    def start(self):
        # start the thread that constantly does serial reading
        # if the serial port doesn't exist the thing crashes sometimes
        Thread(target=self.update, args=()).start()

    def read(self):
        # returns the data structure, be sure to check lockfile
        if self.ser:
            return self.ser.read(1)
        else:
            None
        

    def write(self, m):
        if m not in [b'x', b'y', b'z']:
            self.lock.aquire()
            self.buffer = m
            self.lock.release()

        # sends the write buffer, be sure to check lockfile. Returns delay between when the write was sent and when
        # this was called.

    def update(self):
        while self.running:
            # getting distances via polling now
            self.distances['left'] = ser.write('x')
            self.distances['middle'] = ser.write('y')
            self.distances['right'] = ser.write('z')

            # do the handshakes to read, write if necessary, then delay
            self.lock.aquire()
            if self.buffer:
                ser.write(self.buffer) 
                self.buffer = 0
            self.lock.release() 
           


