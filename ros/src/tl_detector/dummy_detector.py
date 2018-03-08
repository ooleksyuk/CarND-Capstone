#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml

STATE_COUNT_THRESHOLD = 3

class DummyDetector(object):
    
    #---------------------------------------------------------------------------
    # Constructor
    #---------------------------------------------------------------------------
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.stop_line_positions = None
        self.light_stops = {}   # Correlation between lights and stop positions
        self.correlated = False
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic
        light in 3D map space and helps you acquire an accurate ground truth
        data source for the traffic light classifier by sending the current
        color state of all traffic lights in the simulator. When testing on the
        vehicle, the color state will not be available. You'll need to rely on
        the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray,
                                self.traffic_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint',
                                                      Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    #---------------------------------------------------------------------------
    # Callbacks
    #---------------------------------------------------------------------------
    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, msg):
        # Get them only once
        if self.waypoints is None:
            self.waypoints = [wp.pose.pose.position for wp in msg.waypoints]

    def traffic_cb(self, msg):
        self.lights = msg.lights
    
        if self.waypoints == None:
            return
    
        # List of positions that correspond to the line to stop in front of for
        # a given intersection
        self.stop_line_positions = self.config['stop_line_positions']
    
        # Associate the stop lines with the traffic lights
        # (just do this once, the first time we get the method to execute)
        if self.correlated == False:
            self.correlate_lights_and_stop_positions()
            self.correlated = True
    
        # Get the closest waypoint to the position of the car
        if self.pose:
            car_wp_index = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)
            rospy.loginfo("car is at waypoint: %s", car_wp_index)
        else:
            return

        # TODO: Locate the next upcoming red traffic light stop line
        closest_stop_index = 0
        for light in self.lights:
            if light.state == TrafficLight.GREEN:
                continue
            
            # FIX: Get the waypoint index closest to the light
            stop_wp_index = self.get_closest_waypoint(self.light_stops[light][0], self.light_stops[light][1])
  
            if  stop_wp_index > car_wp_index and \
                stop_wp_index < closest_stop_index:
                closest_light_index = stop_wp_index

        rospy.loginfo("closest red/yellow light is at waypoint: %s",
                      closest_light_index)

        # TODO: Correlate the discovered traffic light with the stop line to
        
        # TODO: publish the result of the function call
        # Implement the loop logic to publish until rospy.is_shutdown()

    #---------------------------------------------------------------------------
    # Methods
    #---------------------------------------------------------------------------
    def correlate_lights_and_stop_positions(self):
        
        rospy.loginfo("correlate_lights_and_stop_positions")
        
        for light in self.lights:
            min_dist = float('inf')
            matching_index = 0
            for i, stop_line_position in enumerate(self.stop_line_positions):
                dx = light.pose.pose.position.x - stop_line_position[0]
                dy = light.pose.pose.position.y - stop_line_position[1]
                dist = pow(dx, 2) + pow(dy, 2)
                if dist < min_dist:
                    min_dist = dist
                    matching_index = i
            self.light_stops[light] = self.stop_line_positions[matching_index]

        for light in self.lights:
            rospy.loginfo("%s - %s", light, self.light_stops[light] )

    def get_closest_waypoint(self, x, y):
        """
        Identifies the closest path waypoint to the given position
        https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        
        Args:
            x, y

        Returns:
            int: index of the closest waypoint in self.waypoints
        """
        min_dist = float('inf')
        closest_waypoint_index = 0  # Index to return

        # TODO: Check if the self.waypoints is null before processing

        for i, wp in enumerate(self.waypoints):
            # d^2 = (x1 - x2)^2 + (y1 - y2)^2
            dist = pow(x - wp.x, 2) + pow(y - wp.y, 2)

            # Update the minimum distance and update the index
            if dist < min_dist:
                min_dist = dist
                closest_waypoint_index = i
    
        # Return the index of the closest waypoint in self.waypoints
        return closest_waypoint_index
