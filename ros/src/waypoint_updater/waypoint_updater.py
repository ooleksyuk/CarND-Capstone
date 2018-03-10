#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint

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


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below


        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.tf_listener = tf.TransformListener()

        # Current pose
        self.pose_stamped = None

        # Base waypoints
        self.waypoints_stamped = None

        rospy.spin()

    def pose_cb(self, msg):
        # TODO: Implement
        self.pose_stamped = msg
        # rospy.loginfo("waypoint_updater:pose_cb:self.pose_stamped %s", self.pose_stamped)

        if self.waypoints_stamped == None:
            return

        num_waypoints = len(self.waypoints_stamped.waypoints)

        # Find the closest waypoint to the current pose
        dist_min = sys.maxsize;
        wp_min = None

        x_ego = self.pose_stamped.pose.position.x;
        y_ego = self.pose_stamped.pose.position.y;

        for i in range(num_waypoints):
            waypoint = self.waypoints_stamped.waypoints[i]

            x_wp = waypoint.pose.pose.position.x;
            y_wp = waypoint.pose.pose.position.y;

            dist = (x_ego-x_wp)**2 + (y_ego-y_wp)**2

            if dist < dist_min:
                dist_min = dist
                wp_min = i

        # The next waypoint should be ahead of the current pose.
        # i.e. x co-ordinate of the next waypoint in car's reference frame should be positive
        transformed_waypoint = self.transform_to_car_frame(self.waypoints_stamped.waypoints[wp_min].pose)

        if transformed_waypoint != None and transformed_waypoint.pose.position.x <= 0.0:
            wp_min += 1

        # Construct the set of subsequent waypoints
        next_wps = [None] * LOOKAHEAD_WPS

        for wp in range(wp_min, wp_min + LOOKAHEAD_WPS):
            next_wps[wp - wp_min] = self.waypoints_stamped.waypoints[wp if (wp < num_waypoints) else (wp - num_waypoints)]

        lane = Lane()
        lane.waypoints = next_wps
        lane.header.frame_id = self.waypoints_stamped.header.frame_id
        lane.header.stamp = rospy.Time(0)

        self.final_waypoints_pub.publish(lane)

    def waypoints_cb(self, msg):
        if self.waypoints_stamped != None:
            return

        self.waypoints_stamped = msg;

        for i in range(len(self.waypoints_stamped.waypoints)):
            self.waypoints_stamped.waypoints[i].pose.header.frame_id = self.waypoints_stamped.header.frame_id
            self.set_waypoint_velocity(self.waypoints_stamped.waypoints, i, 20)

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

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


    def transform_to_car_frame(self, pose_stamped):
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

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
