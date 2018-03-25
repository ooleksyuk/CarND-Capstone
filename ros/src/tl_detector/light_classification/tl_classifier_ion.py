from styx_msgs.msg import TrafficLight
from tl_classifier import TLClassifier

import rospy

class IonClassifier(TLClassifier):
    def __init__(self, model_config, input_config):
        self.lights = None
        self.input_source = self.INPUT_SOURCE_SIM

    def feed_image(self, image):
        pass

    def feed_trafficlights(self, lights):
        self.lights = lights

    def get_classification(self, light):
        if self.lights is None or light is None or light >= len(self.lights):
            return TrafficLight.UNKNOWN

        state = self.lights[light].state
        rospy.loginfo("[TL_DETECTOR ION] Nearest TL-state is: %s", TLClassifier.LABELS[state][1])

        return state