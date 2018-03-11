from yaw_controller import YawController
from lowpass import LowPassFilter
from pid import PID

import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, *args, **kwargs):
        # Steering control: yaw controller followed by  a low pass filter
        self.yaw_control = YawController(
                kwargs['wheel_base'],
                kwargs['steer_ratio'],
                kwargs['min_speed'],
                kwargs['max_lat_accel'],
                kwargs['max_steer_angle'])
        self.lpf = LowPassFilter(0.2,0.1)

        # Speed control: PID controller
        pid_gains = kwargs['throttle_gains']
        self.pid = PID(pid_gains[0], pid_gains[1], pid_gains[2])

    def control(self, target, current):
        # Speed control: PID controller step
        u = self.pid.step(target.linear.x, current.linear.x, rospy.get_time())

        if u > 0:
            # Positive control input - accelerating
            throttle = max(0.0, min(1.0,u))
            brake = 0.0
        else:
            # Negative control input - decelerating
            throttle = 0.0
            brake = 10.0

        # Steering control: yaw controller step followed by  a low pass filter step
        steering = self.yaw_control.get_steering(target.linear.x, target.angular.z, current.linear.x)
        steering = self.lpf.filt(steering)

        return throttle, brake, steering

    def reset(self, *args, **kwargs):
        self.pid.reset()
