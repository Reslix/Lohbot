"""
This class (TODO: move to better location) keeps track of the environments and objects within it.

Our map will be a 2D cartesian grid containing points that we observe from the environment. As
we ideally want to stay 30 centimeters away from any static obstacle, and about 50 cm away from people, we can basically
use 30x30 blocks (or circles, I suppose) of no-go zones.

Whenever we take a depth augmented and segmented image of a scene, we use each pixel in the image that is of an object
of interest to draw points on the map. Points that are not within 30cm of another point will be discarded as noise.
(The final value will be adjusted based on experimentation.) Since we don't want to have a large number of points in
memory, we will find the boundaries of these points to attempt to find
"""


class Map(object):

    def __init__(self):
        """
        We set the ground truth here. The current orientation of the robot is [0 0] and the position is [0 0] as well.

        Points are stored in a dictonary with the format of [x, y, type, read_angle]
        """

    def reckon_accel(self, accel_read, read_time=0):
        """
        Given accelerometer readings, update the location of the robot in the map.
        Obviously, we need to know this over two separate captures to properly calculate
        :param accel_read:
        :param read_time:
        :return:
        """
        if read_time == 0:
            self.last_read_time = read_time
        pass

    def ping_update(self, pings):
        """
        Given distances from the three ping sensors on the robot, update the map. This distance will be taken more
        seriously than from the images, mainly because there can be things at ground level that we don't identify with
        the best accuracy. We can be sure that the robot won't be able to cross an obstacle detected by the ping sensor,
        however.

        If a distance is between the (maximum range - buffer), then we add or update a point at that distance as a
        certain obstacle. If later on we sweep the same area to find there isn't anything we remove any points if the
        obstacle was an image determined static obstacle or a moving obstacle.

        If a distance is beyond the buffer we put a tentative obstacle there, and clear any uncertain obstacles in
        between.

        :param pings:
        :return:
        """
        pass

    def clustering_monotone_chain(self):
        """
        In order to simplify the geometry of our map and to get rid of extraneous points, we use an augmented
        monotone chain algorithm to identify all the potential areas where we cannot move the robot to. Our minimum
        chain length is 2, or a line. Single points won't have any extraordinary geometries assigned to them.

        As we are only able to see certain angles of objects, we don't create any
        :return:
        """
        pass