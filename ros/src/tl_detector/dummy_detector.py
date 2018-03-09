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
    """
    Detecting the status of the traffic lights based on simulator information
    
    Caution!!!
    - For development and testing purposes only, this does not work on Carla!
    - For Carla use the live detector found in the live_detector.py
    
    Notes:
    /vehicle/traffic_lights provides you with the location of the traffic
    light in 3D map space and helps you acquire an accurate ground truth
    data source for the traffic light classifier by sending the current
    color state of all traffic lights in the simulator. When testing on the
    vehicle, the color state will not be available. You'll need to rely on
    the position of the light and the camera image to predict it.
    """
    
    #---------------------------------------------------------------------------
    # Constructor
    #---------------------------------------------------------------------------
    
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.stop_line_positions = None
        self.light_stops = {} # Correlation between lights and stop positions
        self.correlated = False # Controls the correlation process
        self.lights = []
        
        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
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
            # Need the waypoints for further processing
            return
    
        # List of positions of the lines to stop in front of intersections
        self.stop_line_positions = self.config['stop_line_positions']
    
        # Associate the stop lines with the traffic lights. This is done once
        if self.correlated == False:
            self.correlate_lights_and_stop_positions()
            self.correlated = True

        # Get the closest waypoint to the position of the car
        if self.pose:
            car_wp_index = self.get_closest_waypoint(self.pose.pose.position.x,
                                                     self.pose.pose.position.y)
            rospy.loginfo("car @ %s", car_wp_index)
        else:
            # Cannot continue without knowing the pose of the car itself
            return

        # Locate the next upcoming red traffic light stop line waypoint index
        closest_stop_index = len(self.waypoints) - 1
        for light in self.lights:
            # Green light is not an obstacle!
            if light.state == TrafficLight.RED:
                
                # Get the stop line from the light
                light_x = light.pose.pose.position.x
                light_y = light.pose.pose.position.y
                stop_line = self.light_stops[(light_x, light_y)]
                rospy.loginfo("found red @ %s",
                              self.get_closest_waypoint(light_x, light_y))

                # Get the waypoint index closest to the stop line
                stop_line_x = stop_line[0]
                stop_line_y = stop_line[1]
                stop_wp_index = self.get_closest_waypoint(stop_line_x,
                                                          stop_line_y)
      
                if  stop_wp_index > car_wp_index and \
                    stop_wp_index < closest_stop_index:
                    closest_stop_index = stop_wp_index

        rospy.loginfo("closest stop @ %s", closest_stop_index)
        
        # Publish the result
        self.upcoming_red_light_pub.publish(closest_stop_index)

    #---------------------------------------------------------------------------
    # Methods
    #---------------------------------------------------------------------------

    def correlate_lights_and_stop_positions(self):
        """
        Assign the closest stop line position to each of the traffic lights.
        The operation is supposed to be done only once
        """
        for light in self.lights:
            # Reset the minimum distance and the index we search for
            min_dist = float('inf')
            matching_index = 0
            
            for i, stop_line_position in enumerate(self.stop_line_positions):
                # Calculate the Euclidean distance
                dx = light.pose.pose.position.x - stop_line_position[0]
                dy = light.pose.pose.position.y - stop_line_position[1]
                dist = pow(dx, 2) + pow(dy, 2)
                
                # Update the minimum distance and matching index
                if dist < min_dist:
                    min_dist = dist
                    matching_index = i
        
            # Correlate each light position (x, y) with the closest stop line
            x = light.pose.pose.position.x
            y = light.pose.pose.position.y
            self.light_stops[(x, y)] = self.stop_line_positions[matching_index]

            rospy.loginfo("light = (%s, %s) : stop = (%s, %s)",
                          x, y, self.stop_line_positions[matching_index][0],
                          self.stop_line_positions[matching_index][1])

    def get_closest_waypoint(self, x, y):
        """
        Identifies the closest path waypoint to the given position
        https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        
        Args:
            float x
            float y

        Returns:
            int: index of the closest waypoint in self.waypoints
        """
        min_dist = float('inf')
        closest_waypoint_index = 0  # Index to return

        for i, wp in enumerate(self.waypoints):
            # d^2 = (x1 - x2)^2 + (y1 - y2)^2
            dist = pow(x - wp.x, 2) + pow(y - wp.y, 2)
            
            # Update the minimum distance and update the index
            if dist < min_dist:
                min_dist = dist
                closest_waypoint_index = i
    
        # Return the index of the closest waypoint in self.waypoints
        return closest_waypoint_index
