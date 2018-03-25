from styx_msgs.msg import TrafficLight
import cv2
import tensorflow as tf
import numpy as np

import rospy

from keras.models import load_model, model_from_json
from keras.utils.generic_utils import get_custom_objects
from keras import backend


from cv_bridge import CvBridge
import cv2

from tl_classifier import TLClassifier

SMOOTH = 1.


def dice_coef(y_true, y_pred):
    y_true_f = backend.flatten(y_true)
    y_pred_f = backend.flatten(y_pred)
    intersection = backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + SMOOTH) / (backend.sum(y_true_f) + backend.sum(y_pred_f) + SMOOTH)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


class UnetClassifier(TLClassifier):
    def __init__(self, model_config, input_config):
        self.input_source = self.INPUT_SOURCE_IMAGE

        self.model = None
        self.width = 0
        self.height = 0
        self.channels = 3
        self.graph = None

        self.image = None
        self.has_image = False
        self.bridge = CvBridge()
        self.invalid_class_number = 3

        self.width = model_config['tl']['classifier_resize_width']
        self.height = model_config['tl']['classifier_resize_height']
        rospy.loginfo("[TL_DETECTOR] Loading TLClassifier model")
        self.model = load_model(model_config['tl']['tl_classification_model'])
        self.channels = 3
        # necessary work around to avoid troubles with keras
        self.graph = tf.get_default_graph()

        rospy.loginfo("[TL_DETECTOR] Loading TLDetector model")
        custom_objects = {'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef}
        # self.detector_model = load_model(model_config['tl']['tl_detection_model'], custom_objects=custom_objects)
        # load json and create model
        json_file = open(model_config['tl']['tl_detector_model_json'], 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.detector_model = model_from_json(loaded_model_json)
        # load weights into new model
        self.detector_model.load_weights(model_config['tl']['tl_detection_model'])
        get_custom_objects().update(custom_objects)
        self.detector_model._make_predict_function()
        rospy.loginfo("[TL_DETECTOR] Loaded models from disk")

        # self.detector_model._make_predict_function()
        self.resize_width = model_config['tl']['detector_resize_width']
        self.resize_height = model_config['tl']['detector_resize_height']

        self.resize_height_ratio = input_config['camera_info']['image_height'] / float(self.resize_height)
        self.resize_width_ratio = input_config['camera_info']['image_width'] / float(self.resize_width)
        self.middle_col = self.resize_width / 2
        self.is_carla = model_config['tl']['is_carla']
        self.projection_threshold = model_config['tl']['projection_threshold']
        self.projection_min = model_config['tl']['projection_min']
        self.color_mode = model_config['tl']['color_mode']

    def feed_image(self, image):
        self.image = image
        self.has_image = True

    def get_classification(self, light):
        if not self.has_image:
            rospy.loginfo("[TL_DETECTOR UNET] has_image is None: No TL is detected. None")
            return TrafficLight.UNKNOWN

        cv_image = self.bridge.imgmsg_to_cv2(self.image, self.color_mode)
        tl_image = self.detect_traffic_light(cv_image)
        if tl_image is not None:
            state = self.get_color_classification(tl_image)
            state = state if (state != self.invalid_class_number) else TrafficLight.UNKNOWN
            rospy.loginfo("[TL_DETECTOR UNET] Nearest TL-state is: %s", TLClassifier.LABELS[state][1])
            return state
        else:
            rospy.loginfo("[TL_DETECTOR UNET] tl_image is None: No TL is detected. None")
            return TrafficLight.UNKNOWN

    def get_color_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        resized = cv2.resize(image, (self.width, self.height))
        resized = resized / 255.;  # Normalization

        # necessary work around to avoid troubles with keras
        with self.graph.as_default():
            predictions = self.model.predict(resized.reshape((1, self.height, self.width, self.channels)))
            color = predictions[0].tolist().index(np.max(predictions[0]))
            tl = TrafficLight()
            tl.state = color
            return tl.state


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