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
import math
# import tensorflow
import cv2
import yaml
import sys
import numpy as np
from keras.models import load_model, model_from_json
from keras.utils.generic_utils import get_custom_objects
from keras import backend

STATE_COUNT_THRESHOLD = 3
TRAFFIC_LIGHT_VISIBLE_DISTANCE = 250  # 250m
SMOOTH = 1.
LABELS = list(enumerate(['Red', 'Yellow', 'Green', 'None', 'None']))


def dice_coef(y_true, y_pred):
    y_true_f = backend.flatten(y_true)
    y_pred_f = backend.flatten(y_pred)
    intersection = backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + SMOOTH) / (backend.sum(y_true_f) + backend.sum(y_pred_f) + SMOOTH)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose_stamped = None
        self.waypoints_stamped = None
        self.camera_image = None
        self.lights = None
        self.has_image = False
        self.light_classifier = TLClassifier()
        self.tf_listener = tf.TransformListener()
        self.prev_light_loc = None

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        self.lights_wp = []
        self.stoplines_wp = []

        self.simulated_detection = True
        self.camera_callback_count = 0

        self.simulated_detection = rospy.get_param('~simulated_detection', 0)
        self.tl_detection_interval_frames = rospy.get_param('~tl_detection_interval_frames', 1)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        # Classifier Setup
        rospy.loginfo("[TL_DETECTOR] Loading TLClassifier model")
        self.light_classifier = TLClassifier()
        model = load_model(self.config['tl']['tl_classification_model'])
        resize_width = self.config['tl']['classifier_resize_width']
        resize_height = self.config['tl']['classifier_resize_height']
        self.light_classifier.setup_classifier(model, resize_width, resize_height)
        self.invalid_class_number = 3

        # Detector setup
        rospy.loginfo("[TL_DETECTOR] Loading TLDetector model")
        custom_objects = {'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef}
        # self.detector_model = load_model(self.config['tl']['tl_detection_model'],
        #                                  custom_objects=custom_objects)
        # load json and create model
        get_custom_objects().update(custom_objects)
        json_file = open(self.config['tl']['tl_detector_model_json'], 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.detector_model = model_from_json(loaded_model_json)
        # load weights into new model
        self.detector_model.load_weights(self.config['tl']['tl_detection_model'])
        self.detector_model._make_predict_function()
        rospy.loginfo("[TL_DETECTOR] Loaded models from disk")

        # self.detector_model._make_predict_function()
        self.resize_width = self.config['tl']['detector_resize_width']
        self.resize_height = self.config['tl']['detector_resize_height']

        self.resize_height_ratio = self.config['camera_info']['image_height'] / float(self.resize_height)
        self.resize_width_ratio = self.config['camera_info']['image_width'] / float(self.resize_width)
        self.middle_col = self.resize_width / 2
        self.is_carla = self.config['tl']['is_carla']
        self.projection_threshold = self.config['tl']['projection_threshold']
        self.projection_min = self.config['tl']['projection_min']
        self.color_mode = self.config['tl']['color_mode']

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()

        rospy.spin()

    def pose_cb(self, msg):
        self.pose_stamped = msg

    def waypoints_cb(self, msg):
        if self.waypoints_stamped is not None:
            return

        self.waypoints_stamped = msg

        for i in range(len(self.waypoints_stamped.waypoints)):
            self.waypoints_stamped.waypoints[i].pose.header.frame_id = self.waypoints_stamped.header.frame_id

        self.calculate_traffic_light_waypoints()

    def traffic_cb(self, msg):
        if self.simulated_detection is True:
            self.lights = msg.lights
            self.calculate_traffic_light_waypoints()

            light_wp, state = self.process_traffic_lights()
            self.publish_upcoming_red_light(light_wp, state)
        else:
            if self.lights is not None:
                return

            self.lights = msg.lights
            self.calculate_traffic_light_waypoints()

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """

        self.camera_callback_count += 1

        if self.camera_callback_count < self.tl_detection_interval_frames:
            return
        
        self.camera_callback_count = 0

        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        '''
        self.publish_upcoming_red_light(light_wp, state)

    def publish_upcoming_red_light(self, light_wp, state):
        """Publishes the index of the waypoint closest to the red light's 
            stop line to /traffic_waypoint

        Args:
            light_wp: waypoint of the closest traffic light
            state: state of the closest traffic light 

        """

        '''
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

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO This function is currently O(n). It can be improved to O(log n)
        if self.waypoints_stamped is None:
            return None

        dist_min = sys.maxsize
        wp_min = None

        for wp in range(len(self.waypoints_stamped.waypoints)):
            dist = self.distance2(pose, self.waypoints_stamped.waypoints[wp].pose.pose)

            if dist < dist_min:
                dist_min = dist
                wp_min = wp

        return wp_min

    def extract_image(self, pred_image_mask, image):
        # rospy.loginfo("[TL_DETECTOR] Detecting TL...extract_image()")
        if np.max(pred_image_mask) < self.projection_min:
            return None

        row_projection = np.sum(pred_image_mask, axis=1)
        row_index = np.argmax(row_projection)

        if np.max(row_projection) < self.projection_threshold:
            return None

        zero_row_indexes = np.argwhere(row_projection <= self.projection_threshold)
        top_part = zero_row_indexes[zero_row_indexes < row_index]
        top = np.max(top_part) if top_part.size > 0 else 0
        bottom_part = zero_row_indexes[zero_row_indexes > row_index]
        bottom = np.min(bottom_part) if bottom_part.size > 0 else self.resize_height

        roi = pred_image_mask[top:bottom, :]
        column_projection = np.sum(roi, axis=0)

        if np.max(column_projection) < self.projection_min:
            return None

        non_zero_column_index = np.argwhere(column_projection > self.projection_min)

        index_of_column_index = np.argmin(np.abs(non_zero_column_index - self.middle_col))
        column_index = non_zero_column_index[index_of_column_index][0]

        zero_column_indexes = np.argwhere(column_projection == 0)
        left_side = zero_column_indexes[zero_column_indexes < column_index]
        left = np.max(left_side) if left_side.size > 0 else 0
        right_side = zero_column_indexes[zero_column_indexes > column_index]
        right = np.min(right_side) if right_side.size > 0 else self.resize_width
        return image[int(top * self.resize_height_ratio):int(bottom * self.resize_height_ratio),
                     int(left * self.resize_width_ratio):int(right * self.resize_width_ratio)]

    def detect_traffic_light(self, cv_image):
        # rospy.loginfo("[TL_DETECTOR] Detecting TL...detect_traffic_light()")
        resize_image = cv2.resize(cv_image, (self.resize_width, self.resize_height))
        resize_image = cv2.cvtColor(resize_image, cv2.COLOR_RGB2GRAY)
        resize_image = resize_image[..., np.newaxis]
        if self.is_carla:
            mean = np.mean(resize_image)  # mean for data centering
            std = np.std(resize_image)  # std for data normalization

            resize_image -= mean
            resize_image /= std

        image_mask = self.detector_model.predict(resize_image[None, :, :, :], batch_size=1)[0]
        image_mask = (image_mask[:, :, 0] * 255).astype(np.uint8)
        return self.extract_image(image_mask, cv_image)

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        if self.simulated_detection is True:
            if self.lights is None or light >= len(self.lights):
                rospy.loginfo("[TL_DETECTOR] No TL is detected. None")
                return TrafficLight.UNKNOWN
            state = self.lights[light].state
            return state

        if not self.has_image:
            self.prev_light_loc = None
            rospy.loginfo("[TL_DETECTOR] No TL is detected. None")
            return TrafficLight.UNKNOWN

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, self.color_mode)
        tl_image = self.detect_traffic_light(cv_image)
        if tl_image is not None:
            state = self.light_classifier.get_classification(tl_image)
            state = state if (state != self.invalid_class_number) else TrafficLight.UNKNOWN
            return state
        else:
            rospy.loginfo("[TL_DETECTOR] No TL is detected. None")
            return TrafficLight.UNKNOWN

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if self.pose_stamped is None or len(self.stoplines_wp) == 0:
            rospy.loginfo("[TL_DETECTOR] No TL is detected. None")
            return -1, TrafficLight.UNKNOWN

        # find the closest visible traffic light (if one exists)
        light = self.get_closest_visible_traffic_light(self.pose_stamped.pose)

        if light is None:
            rospy.loginfo("[TL_DETECTOR] No TL is detected. None")
            return -1, TrafficLight.UNKNOWN
        state = self.get_light_state(light)
        rospy.loginfo("[TL_DETECTOR] Nearest TL-state is: %s", LABELS[state][1])

        return self.stoplines_wp[light], state

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

    def get_closest_stopline_pose(self, pose):
        """Finds closest stopline to the given Pose

        Args:
            pose: given Pose

        Returns:
            Pose: a Pose object whose position is that of the closest stopline

        """
        stop_line_positions = self.config['stop_line_positions']

        dist_min = sys.maxsize
        stop_line_min = None

        for stop_line_position in stop_line_positions:
            stop_line_pose = Pose()
            stop_line_pose.position.x = stop_line_position[0]
            stop_line_pose.position.y = stop_line_position[1]
            stop_line_pose.position.z = 0.0

            dist = self.distance2(pose, stop_line_pose)

            if dist < dist_min:
                dist_min = dist
                stop_line_min = stop_line_pose

        return stop_line_min

    def calculate_traffic_light_waypoints(self):
        """Populate traffic light waypoints and stopline waypoints arrays if they are not already populated

            self.lights_wp contains the closest waypoints to corresponding trafic lights in self.lights
            self.stoplines_wp contains the waypoints of stoplines corrsponding to trafic lights in self.lights

        """
        if self.waypoints_stamped is not None and self.lights is not None and len(self.lights_wp) == 0:
            for i in range(len(self.lights)):
                stopline = self.get_closest_stopline_pose(self.lights[i].pose.pose)
                self.stoplines_wp.append(self.get_closest_waypoint(stopline))
                self.lights_wp.append(self.get_closest_waypoint(self.lights[i].pose.pose))

                # rospy.logwarn("calculate_traffic_light_waypoints: %d %f:%f %f:%f %d", self.lights_wp[i], self.lights[i].pose.pose.position.x, self.lights[i].pose.pose.position.y,
                #     stopline.position.x, stopline.position.y, self.stoplines_wp[i])

    def get_closest_visible_traffic_light(self, pose):
        """Finds closest visible traffic light to the given Pose

        Args:
            pose: given Pose

        Returns:
            int: index the closest visible traffic light (None if none exists)

        """
        if self.waypoints_stamped is None or self.lights is None or len(self.lights_wp) == 0:
            return None

        num_lights = len(self.lights_wp)

        dist_min = sys.maxsize
        light_min = None

        for light in range(num_lights):
            dist = self.distance2(pose, self.waypoints_stamped.waypoints[self.lights_wp[light]].pose.pose)

            if dist < dist_min:
                dist_min = dist
                light_min = light

        transformed_waypoint = self.transform_to_car_frame(self.waypoints_stamped.waypoints[self.lights_wp[light_min]].pose)

        if transformed_waypoint is not None and transformed_waypoint.pose.position.x <= 0.0:
            light_min += 1

        if light_min >= num_lights:
            light_min -= num_lights

        dist2 = self.distance2(pose, self.waypoints_stamped.waypoints[self.lights_wp[light_min]].pose.pose)

        if dist2 > (TRAFFIC_LIGHT_VISIBLE_DISTANCE**2):
            return None

        return light_min


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
