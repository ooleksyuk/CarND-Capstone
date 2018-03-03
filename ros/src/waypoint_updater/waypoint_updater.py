#!/usr/bin/env python

import rospy
import numpy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from tf.transformations import euler_from_quaternion, quaternion_from_euler

import math

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
        self.base_waypoints = None
        self.last_ts = None

        rospy.spin()

    def pose_cb(self, msg):
        # stay still until base_waypoints is defined
        if self.base_waypoints is None:
            return
        rospy.loginfo(rospy.get_caller_id() + ": current pose received")
        # debug
        print("pose_cb message: ", msg)

        current_ts = msg.header.stamp.to_sec()
        # debug
        print("ts: ", current_ts)
        current_pos = msg.pose.position # x, y, z cartesian coordinate
        current_ori = msg.pose.orientation # x, y, z, w quaternion
        (roll, pitch, yaw) = euler_from_quaternion ([current_ori.x,current_ori.y,current_ori.z,current_ori.w])
        # debug
        print("roll, pitch, yaw: ", roll, pitch, yaw)

        # publish dummy
        if self.last_ts is not None:
            dt = current_ts - self.last_ts
            print("dt: ", dt)
            v = 1
            v_x = v * numpy.cos(yaw)
            v_y = v * numpy.sin(yaw)

            final_waypoints = Lane()
            final_waypoints.header = msg.header
            for i in range(LOOKAHEAD_WPS):
                dummy_pose = PoseStamped()
                dummy_pose.pose.position.x = msg.pose.position.x + i * v_x * dt
                dummy_pose.pose.position.y = msg.pose.position.y + i * v_y * dt
                dummy_pose.pose.position.z = msg.pose.position.z
                dummy_pose.pose.orientation.x = msg.pose.orientation.x
                dummy_pose.pose.orientation.y = msg.pose.orientation.y
                dummy_pose.pose.orientation.z = msg.pose.orientation.z
                dummy_pose.pose.orientation.w = msg.pose.orientation.w

                dummy_waypoint = Waypoint()
                dummy_waypoint.pose = dummy_pose
                final_waypoints.waypoints.append(dummy_waypoint)
            # print(final_waypoints)
            self.final_waypoints_pub.publish(final_waypoints)
            rospy.loginfo(rospy.get_caller_id() + ": {} final waypoints published".format(len(final_waypoints.waypoints)))

        # update ts
        self.last_ts = current_ts


    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints.waypoints
        rospy.loginfo(rospy.get_caller_id() + ": {} base waypoints received".format(len(self.base_waypoints)))
        # debug
        print("base waypoints: ", self.base_waypoints[0])

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


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
