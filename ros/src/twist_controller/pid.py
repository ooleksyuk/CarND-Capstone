
MIN_NUM = float('-inf')
MAX_NUM = float('inf')


class PID(object):
    def __init__(self, kp, ki, kd, max_input=MAX_NUM, mn=MIN_NUM, mx=MAX_NUM):
        # PID parameters
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min = mn
        self.max = mx
        # This is a crude approximation
        self.max_abs_u = (abs(self.kp) + abs(self.ki) + abs(self.kd)) * abs(max_input)

        # Controller state
        self.t = None
        self.error = 0.0
        self.integral = 0.0

    def reset(self):
        self.t = None

    def step(self, target, current, t):
        # PID controller requires delta t and derivative of error. 
        # i.e. at least one timestep must have elapsed.
        if self.t == None:
            self.t = t
            self.integral = 0.0
            self.error = target - current
            return 0.0

        delta_t = t - self.t

        # calculate error components
        error = target - current
        integral = max(MIN_NUM, min(MAX_NUM, self.integral + error * delta_t)) 
        derivative = (error - self.error) / delta_t

        # control input = P * error + I * integral(error) + D * derivative(error)
        control = self.kp * error + self.ki * integral + self.kd * derivative
        control = max(self.min, min(self.max, control))

        # Record state
        self.t = t
        self.error = error
        self.integral = integral

        return control
