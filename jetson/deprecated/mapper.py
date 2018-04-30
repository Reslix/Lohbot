"""
We obtain mapping information from two sources: the ultrasonic ping sensors, and whatever we obtain from the cameras.

"""
import numpy as np

class Mapper(object):
    """
    Keeps track of the robots location in relation to mapped objects/obstacles.
    """
    def __init__(self):
         self.map = []