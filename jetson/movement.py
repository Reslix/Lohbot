
class Motion():

    def __init__(self, serial, map):
        self.serial = serial
        self.map = map

    def move_rel(self):
        """
        Move to a new location relative to robot, returns new relative position and error
        :return:
        """
        pass

    def move(self):
        """
        Sends command to move forward at a speed.
        :return:
        """
        pass

    def turn(self):
        """
        Turns at a speed in a direction.
        :return:
        """
        pass

    def waypoint_add_global(self):
        """
        Adds absolute locations according to the map.
        :return:
        """
        pass

    def waypoint_add_rel(self):
        """
        Adds relative waypoint, which means it'll have a noisy space on the global map
        :return:
        """
        pass

    def reckon(self):
        """
        Figures out the location of the robot in map.
        :return:
        """
