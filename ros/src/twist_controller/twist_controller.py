from yaw_controller import YawController
from lowpass import LowPassFilter
from pid import PID

import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, *args, **kwargs):
        self.yaw_control = YawController(
                kwargs['wheel_base'],
                kwargs['steer_ratio'],
                kwargs['min_speed'],
                kwargs['max_lat_accel'],
                kwargs['max_steer_angle'])

        pid_gains = kwargs['throttle_gains']
        self.pid = PID(pid_gains[0], pid_gains[1], pid_gains[2])
        self.lpf = LowPassFilter(0.2,0.1)

        self.t = None

    def control(self, target, current):
        t = rospy.get_time()

        if self.t == None:
            self.t = t
            return 0.0, 0.0, 0.0

        dt = t - self.t
        vx_error = target.linear.x - current.linear.x
        u = self.pid.step(vx_error, dt)
        #rospy.logwarn("control: error %f dt %f u %f", vx_error, dt, u)
        if u > 0:
            throttle = max(0.0, min(1.0,u))
            brake = 0.0
        else:
            throttle = 0.0
            brake = 10.0

        steering = self.yaw_control.get_steering(target.linear.x, target.angular.z, current.linear.x)
        steering = self.lpf.filt(steering)
        self.t = t
        # Return throttle, brake, steer
        return throttle, brake, steering

    def reset(self, *args, **kwargs):
        self.t = rospy.get_time()
        self.pid.reset()
