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
        self.lpf = LowPassFilter(kwargs['steering_tau'], 1.0/kwargs['sample_rate'])

        # Speed control: PID controller
        pid_gains = kwargs['throttle_gains']
        self.pid = PID(pid_gains[0], pid_gains[1], pid_gains[2], kwargs['max_speed'])

        total_mass = kwargs['vehicle_mass'] + kwargs['fuel_capacity']*GAS_DENSITY
        self.max_brake_torque = total_mass * kwargs['decel_limit'] * kwargs['wheel_radius']
        self.min_brake = -1.0 * kwargs['brake_deadband']

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
            brake = self.max_brake_torque * min(self.min_brake, u/self.pid.max_abs_u)

        # Steering control: yaw controller step followed by  a low pass filter step
        steering = self.yaw_control.get_steering(target.linear.x, target.angular.z, current.linear.x)
        steering = self.lpf.filt(steering)

        # rospy.logwarn("current %04.3f target %04.3f target w %04.3f u %04.3f %04.3f:%04.3f:%04.3f",
        #     current.linear.x, target.linear.x, target.angular.z, u, throttle, brake, steering)

        return throttle, brake, steering

    def reset(self, *args, **kwargs):
        self.pid.reset()
