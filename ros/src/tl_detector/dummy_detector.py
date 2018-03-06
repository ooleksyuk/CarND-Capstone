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
        self.camera_image = None
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
                                
        # TODO: Remove camera subscription
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

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
    
        # TODO: Consider moving all the process_traffic_lights here, else:
    
        # TODO: Call process_traffic_lights, like so:
        light_wp, state = self.process_traffic_lights()
        
        # TODO: publish the result of the function call
        # Implement the loop logic to publish until rospy.is_shutdown()

    def image_cb(self, msg):
        """
        Identifies red lights in the incoming camera image and publishes the
        index of the waypoint closest to the red light's stop line to
        /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera
        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    #---------------------------------------------------------------------------
    # Methods
    #---------------------------------------------------------------------------
    def get_closest_waypoint(self, pose):
        """
        Identifies the closest path waypoint to the given position
        https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        
        Args:
            pose (Pose): position to match a waypoint to
                         in the form: pose.x, pose.y

        Returns:
            int: index of the closest waypoint in self.waypoints
        """
        min_dist = float('inf')
        closest_waypoint_index = 0  # Index to return

        # TODO: Check if the self.waypoints is null before processing

        for i, wp in enumerate(self.waypoints):
            # d^2 = (x1 - x2)^2 + (y1 - y2)^2
            dist = pow(pose.x - wp.x, 2) + pow(pose.y - wp.y, 2)

            # Update the minimum distance and update the index
            if dist < min_dist:
                min_dist = dist
                closest_waypoint_index = i
    
        # Return the index of the closest waypoint in self.waypoints
        return closest_waypoint_index

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        # Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """
        Finds closest visible traffic light, if one exists, and determines its
        location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a
                 traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        light = None

        # List of positions that correspond to the line to stop in front of for
        # a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            rospy.loginfo("self.pose.pose.position %s", self.pose.pose.position)
            car_position = self.get_closest_waypoint(self.pose.pose.position)
    
        # TODO: Associate the stop lines with the traffic lights
        # (just do this once, the first time we get the method to execute)
    
        #-------------------------------------------------------------
        # TODO: Find the closest visible traffic light (if one exists)
        #-------------------------------------------------------------
        # Step 1. Get the car location
        
        # Step 2. Locate the next upcoming traffic light
            # Set a big minimum distance to begin
            # For all the traffic lights:
                # Calculate the distance from the car's position
                # Check if the status of the traffic light is red
                # Update the minimum distance and traffic light index
        
        # TODO: Correlate the discovered traffic light with the stop line to
        # return to the callback
        
        # Return result to callback
        if light:
            state = self.get_light_state(light)
            return light_wp, state
        self.waypoints = None
        return -1, TrafficLight.UNKNOWN
