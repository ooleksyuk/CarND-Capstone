class hysteresis:
    def __init__(self, lower_thershold, upper_threshold, init, lower_limit=0.0, upper_limit=0.0):
        self.upper_threshold = upper_threshold
        self.lower_thershold = lower_thershold
        self.upper_limit = upper_limit
        self.lower_limit = lower_limit
        self.history = init

    def output(self, input):
        if input > self.history:
            if input < self.upper_threshold:
                self.history = self.upper_limit
            else:
                self.history = input
        else:
            if input < self.lower_thershold:
                self.history = self.lower_limit
            else:
                self.history = input

        return self.history