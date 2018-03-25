from styx_msgs.msg import TrafficLight


class TLClassifier(object):
    INPUT_SOURCE_IMAGE = 0
    INPUT_SOURCE_SIM = 1

    LABELS = list(enumerate(['Red', 'Yellow', 'Green', 'None', 'None']))

    def __init__(self, model_config, input_config):
        self.image = None
        self.has_image = False
        self.input_source = None

    def feed_image(self, image):
        self.image = image
        self.has_image = True

    def feed_trafficlights(self, lights):
        pass

    def get_classification(self, light):
        return TrafficLight.UNKNOWN