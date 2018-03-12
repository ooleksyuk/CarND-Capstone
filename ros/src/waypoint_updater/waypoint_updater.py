#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32

import math
import sys
import tf

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
MAX_SPEED = 20.0


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)


        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.tf_listener = tf.TransformListener()
        self.red_light_waypoint = None
        self.next_waypoint = -1
        self.current_speed = 0.0
        self.trajectory_target_speed = 0.0
        self.trajectory_acceleration = 0.0

        # Current pose
        self.pose_stamped = None

        # Base waypoints
        self.waypoints_stamped = None

        rospy.spin()

    def pose_cb(self, msg):
        self.pose_stamped = msg
        # rospy.loginfo("waypoint_updater:pose_cb:self.pose_stamped %s", self.pose_stamped)

        # Do nothing until all messages have been recieved
        if self.waypoints_stamped == None or self.red_light_waypoint == None:
            return

        # Find the closest waypoint to the current pose
        next_waypoint = self.get_closest_waypoint(self.pose_stamped.pose)

        # The next waypoint should be ahead of the current pose.
        # i.e. x co-ordinate of the next waypoint in car's reference frame should be positive
        transformed_waypoint = self.transform_to_car_frame(self.waypoints_stamped.waypoints[next_waypoint].pose)

        if transformed_waypoint != None and transformed_waypoint.pose.position.x <= 0.0:
            next_waypoint += 1

        num_waypoints = len(self.waypoints_stamped.waypoints)

        if next_waypoint > num_waypoints:
            next_waypoint -= num_waypoints

        # Calculate a trajectory
        self.calculate_trajectory(next_waypoint)

        # Construct the set of subsequent waypoints
        next_wps = [None] * LOOKAHEAD_WPS

        for _wp, wp in enumerate(range(next_waypoint, next_waypoint + LOOKAHEAD_WPS)):
            next_wps[_wp] = self.waypoints_stamped.waypoints[wp if (wp < num_waypoints) else (wp - num_waypoints)]
            self.set_waypoint_velocity(next_wps, _wp, self.get_trajectory_speed_at_waypoint(_wp))

        # This is a make shift way of working out the current speed
        if self.next_waypoint != next_waypoint:
            self.next_waypoint = next_waypoint
            self.current_speed = self.get_waypoint_velocity(next_wps[0])

        # Construct final_waypoints message
        lane = Lane()
        lane.waypoints = next_wps
        lane.header.frame_id = self.waypoints_stamped.header.frame_id
        lane.header.stamp = rospy.Time(0)

        self.final_waypoints_pub.publish(lane)

    def waypoints_cb(self, msg):
        if self.waypoints_stamped != None:
            return

        self.waypoints_stamped = msg

        for i in range(len(self.waypoints_stamped.waypoints)):
            self.waypoints_stamped.waypoints[i].pose.header.frame_id = self.waypoints_stamped.header.frame_id
            self.set_waypoint_velocity(self.waypoints_stamped.waypoints, i, 0.0)

    def traffic_cb(self, msg):
        self.red_light_waypoint = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def distance2(self, pose1, pose2):
        """Calculate the square of the Eucleadian distance bentween the two poses given

        Args:
            pose1: given Pose
            pose2: given Pose

        Returns:
            float: square of the Eucleadian distance bentween the two poses given

        """
        dist2 = (pose1.position.x-pose2.position.x)**2 + (pose1.position.y-pose2.position.y)**2
        return dist2

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO This function is currently O(n). It can be improved to O(log n)
        if self.waypoints_stamped == None:
            return None

        dist_min = sys.maxsize
        wp_min = None

        for wp in range(len(self.waypoints_stamped.waypoints)):
            dist = self.distance2(pose, self.waypoints_stamped.waypoints[wp].pose.pose)

            if dist < dist_min:
                dist_min = dist
                wp_min = wp

        return wp_min

    def transform_to_car_frame(self, pose_stamped):
        """Transform the given pose to car co-ordinate frame

        Args:
            pose: given PoseStamped object

        Returns:
            PoseStamped: a PoseStamped object which is car co-ordinate frame equivalent
                  of the given pose. (None if the tranformation failed)

        """
        try:
            self.tf_listener.waitForTransform("base_link", "world", rospy.Time(0), rospy.Duration(0.02))
            transformed_pose_stamped = self.tf_listener.transformPose("base_link", pose_stamped)
        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            try:
                self.tf_listener.waitForTransform("base_link", "world", rospy.Time(0), rospy.Duration(1.0))
                transformed_pose_stamped = self.tf_listener.transformPose("base_link", pose_stamped)
            except (tf.Exception, tf.LookupException, tf.ConnectivityException):
                transformed_pose_stamped = None
                rospy.logwarn("Failed to transform pose")

        return transformed_pose_stamped

    def calculate_trajectory(self, next_waypoint):
        """Calculate a trajectory starting at the given waypoint so that the car
           comes to a halt at the next stopline in case of a red light and
           travels at speed limit otherwise

        Args:
            next_waypoint: index of the next waypoint

        """
        num_waypoints = len(self.waypoints_stamped.waypoints)

        self.trajectory_target_speed = MAX_SPEED
        waypoints_to_target = 20

        if self.red_light_waypoint > 0:
            self.trajectory_target_speed = 0.0
            waypoints_to_target = 0 if (self.red_light_waypoint >= next_waypoint) else num_waypoints
            waypoints_to_target += (self.red_light_waypoint - next_waypoint)

        # The 'acceleration' here is speed/distance (m/s/m) not speed/time (m/s^2)
        # This is only a crude way of getting the car to stop somewhat smoothly
        # I am assuming that the waypoints are spaced reasonably evenly
        self.trajectory_acceleration = (self.trajectory_target_speed - self.current_speed) / waypoints_to_target if (waypoints_to_target != 0) else 0.0

    def get_trajectory_speed_at_waypoint(self, waypoint):
        """Get the expected speed at the given waypoint as per the
           current trajectory

        Args:
            waypoint: index of the waypoint

        Returns:
            float: trajectory speed at the given waypoint

        """
        # The +1 is to make the speed at 0th waypoin non-zero. Waypoint follower seem to want this.
        return self.current_speed + (waypoint + 1) * self.trajectory_acceleration

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
