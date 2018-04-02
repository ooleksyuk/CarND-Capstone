from styx_msgs.msg import TrafficLight
import cv2
import numpy as np

import rospy
import tensorflow

from cv_bridge import CvBridge
import cv2

import timeit

from tl_classifier import TLClassifier


class UnetTensorflowClassifier(TLClassifier):
    def __init__(self, model_config, input_config):
        self.input_source = self.INPUT_SOURCE_IMAGE

        self.image = None
        self.has_image = False
        self.bridge = CvBridge()
        self.invalid_class_number = 3

        rospy.loginfo("[UNET_TENSORFLOW_DETECTOR] Loading TLClassifier model")

        self.classfication_graph = self.load_graph(model_config['tl']['tl_classification_graph'])
        self.classfication_input_operation = self.classfication_graph.get_operation_by_name("import/" + model_config['tl']['classification_input_operation'])
        self.classfication_output_operation = self.classfication_graph.get_operation_by_name("import/" + model_config['tl']['classification_output_operation'])
        self.classfication_session = tensorflow.Session(graph=self.classfication_graph)

        self.classifier_input_width = model_config['tl']['classifier_resize_width']
        self.classifier_input_height = model_config['tl']['classifier_resize_height']
        self.classifier_input_channels = 3


        rospy.loginfo("[UNET_TENSORFLOW_DETECTOR] Loading TLDetector model")

        self.detection_graph = self.load_graph(model_config['tl']['tl_detection_graph'])
        self.detection_input_operation = self.detection_graph.get_operation_by_name("import/" + model_config['tl']['detection_input_operation'])
        self.detection_output_operation = self.detection_graph.get_operation_by_name("import/" + model_config['tl']['detection_output_operation'])
        self.detection_session = tensorflow.Session(graph=self.detection_graph)

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
            rospy.loginfo("[TL_DETECTOR UNET TENSORFLOW] has_image is None: No TL is detected. None")
            return TrafficLight.UNKNOWN

        delta_time = timeit.default_timer()
        cv_image = self.bridge.imgmsg_to_cv2(self.image, self.color_mode)
        tl_image = self.detect_traffic_light(cv_image)
        if tl_image is not None:
            state = self.get_color_classification(tl_image)
            state = state if (state != self.invalid_class_number) else TrafficLight.UNKNOWN
            delta_time = timeit.default_timer() - delta_time
            rospy.loginfo("[TL_DETECTOR UNET TENSORFLOW] Nearest TL-state is: %s dt %f", TLClassifier.LABELS[state][1], delta_time)
            return state
        else:
            rospy.loginfo("[TL_DETECTOR UNET TENSORFLOW] tl_image is None: No TL is detected. None")
            return TrafficLight.UNKNOWN

    def get_color_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        resized = cv2.resize(image, (self.classifier_input_width, self.classifier_input_height))
        resized = resized / 255.;  # Normalization

        predictions = self.classfication_session.run(self.classfication_output_operation.outputs[0], {self.classfication_input_operation.outputs[0]: resized.reshape((1, self.classifier_input_height, self.classifier_input_width, self.classifier_input_channels))})
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

        image_mask = self.detection_session.run(self.detection_output_operation.outputs[0], {self.detection_input_operation.outputs[0]: resize_image[None, :, :, :]})[0]
        image_mask = (image_mask[:, :, 0] * 255).astype(np.uint8)
        return self.extract_image(image_mask, cv_image)

    def load_graph(self, model_file):
        graph = tensorflow.Graph()
        graph_def = tensorflow.GraphDef()

        with open(model_file, "rb") as f:
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            tensorflow.import_graph_def(graph_def)

        return graph