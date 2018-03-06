#!/usr/bin/env python3

import serial
import threading
from threading import Thread

class SerialIO():

    def __init__(self, com=None, baud=9600, delay=10):
        # initialize data structure to keep shizz
        # put lockfile, write buffer
        self.baud = baud
        self.delay = delay
        self.running = 1
        self.buffer = 0;
        self.lock = threading.RLock()
        self.distances = {'left': 201,'right': 201,'middle': 201} # start off with error values for distance

        # better than silently failing
        self.ser = serial.Serial('/dev/ttyACM0', self.baud, timeout = self.delay)
        # wait for startup time
        # self.ser.read()

    def start(self):
        # start the thread that constantly does serial reading
        # if the serial port doesn't exist the thing crashes sometimes
        Thread(target=self.update, args=()).start()

    def read(self, dir):
        # returns the data structure, be sure to check lockfile
        return self.distances[dir]
        

    def write(self, m):
        if m not in [b'x', b'y', b'z']:
            self.lock.acquire()
            self.buffer = m
            self.lock.release()

        # sends the write buffer, be sure to check lockfile. Returns delay between when the write was sent and when
        # this was called.

    def update(self):
        while self.running:
            # getting distances via polling now
            self.ser.write(b'x')
            self.distances['left'] = self.ser.read()
            self.ser.write(b'y')
            self.distances['middle'] = self.ser.read()
            self.ser.write(b'z')
            self.distances['right'] = self.ser.read()

            # do the handshakes to read, write if necessary, then delay
            self.lock.acquire()
            if self.buffer:
                ser.write(self.buffer) 
                self.buffer = 0
            self.lock.release() 
           


