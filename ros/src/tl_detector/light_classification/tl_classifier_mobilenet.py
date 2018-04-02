from styx_msgs.msg import TrafficLight
from tl_classifier import TLClassifier

from cv_bridge import CvBridge
import cv2

import rospy
import tensorflow
import numpy as np

import timeit

class MobilenetClassifier(TLClassifier):
    def __init__(self, model_config, input_config):
        self.input_source = self.INPUT_SOURCE_IMAGE

        self.image = None
        self.bridge = CvBridge()
        self.color_mode = model_config['tl']['color_mode']

        self.graph = self.load_graph(model_config['tl']['tl_classification_graph'])
        self.labels = self.load_labels(model_config['tl']['tl_classification_labels'])
        self.input_height = model_config['tl']['classifier_resize_height']
        self.input_width = model_config['tl']['classifier_resize_width']
        self.input_mean = model_config['tl']['classifier_input_mean']
        self.input_std = model_config['tl']['classifier_input_std']
        self.input_operation = self.graph.get_operation_by_name("import/" + model_config['tl']['input_operation'])
        self.output_operation = self.graph.get_operation_by_name("import/" + model_config['tl']['output_operation'])

        self.sess = tensorflow.Session(graph=self.graph)


    def feed_image(self, image):
        self.image = image

    def get_classification(self, light):
        if self.image == None:
            rospy.loginfo("[TL_DETECTOR MOBILENET] has_image is None: No TL is detected. None")
            return TrafficLight.UNKNOWN

        cv_image = self.bridge.imgmsg_to_cv2(self.image, self.color_mode)

        delta_time = timeit.default_timer()
        state = self.get_color_classification(cv_image)
        delta_time = timeit.default_timer() - delta_time
        rospy.loginfo("[TL_DETECTOR MOBILENET] Nearest TL-state is: %s dt %f", TLClassifier.LABELS[state][1], delta_time)

        return state

    def load_graph(self, model_file):
        graph = tensorflow.Graph()
        graph_def = tensorflow.GraphDef()

        with open(model_file, "rb") as f:
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            tensorflow.import_graph_def(graph_def)

        return graph

    
    def load_labels(self, label_file):
        label_dict = {'green':TrafficLight.GREEN,
                     'no':TrafficLight.UNKNOWN,
                     'red':TrafficLight.RED,
                     'yellow':TrafficLight.YELLOW}
        labels = []
        proto_as_ascii_lines = tensorflow.gfile.GFile(label_file).readlines()
        for l in proto_as_ascii_lines:
            labels.append(label_dict[l.rstrip()])

        return labels

    def read_tensor(self, image, input_height=128, input_width=128, input_mean=128, input_std=128):
        
        float_caster = tensorflow.cast(image, tensorflow.float32)
        dims_expander = tensorflow.expand_dims(float_caster, 0)
        resized = tensorflow.image.resize_bilinear(dims_expander, [input_height, input_width])
        normalized = tensorflow.divide(tensorflow.subtract(resized, [input_mean]), [input_std])
        sess = tensorflow.Session()
        result = sess.run(normalized)

        return result

    def get_color_classification(self, image):
        resized = image.astype('float')
        resized = (cv2.resize(resized, (self.input_width, self.input_height)))
        resized -= self.input_mean
        resized /= self.input_std
        resized = np.expand_dims(resized, axis=0)

        # resized = self.read_tensor(image)

        # with tensorflow.Session(graph=self.graph) as sess:
        results = self.sess.run(self.output_operation.outputs[0], {self.input_operation.outputs[0]: resized})

        results = np.squeeze(results)
        state = results.argsort()[-1:][::-1][0]

        # rospy.loginfo("state %s %f", self.labels[state], results[state])

        return self.labels[state]