#!/usr/bin/env python
import rospy
from std_msgs.msg import String,Int16
from sensor_msgs.msg import Joy
import numpy as np

class RemoteControl(object):
    def __init__(self):
        self.max_throttle = 200
        self.joy_deadband = 0.1
        self.last_steer = 1500
        self.last_throttle = 1500
	self.fixed_throttle = 1600


        rospy.init_node('remote_control')
        self.steering_pub = rospy.Publisher('/steering_value_us', Int16, queue_size=1024)
        self.throttle_pub = rospy.Publisher('/throttle_value_us', Int16, queue_size=1024)
        rospy.Subscriber("/joy", Joy, self.joy_cb)
        rospy.spin()


    def map_value(self, joy_value, max_displacement):
        value_sign = np.sign(joy_value)
        value = abs(joy_value)
        if value < self.joy_deadband:
            return 1500
        else:
            value -= self.joy_deadband
            return 1500 + max_displacement * value_sign * value / (1.0 - self.joy_deadband)


    def joy_cb(self, msg):
	if msg.axes[2] >= 0.5:
            throttle = self.fixed_throttle
        else:
	    throttle = self.map_value(msg.axes[1], 200)
        
        steer = self.map_value(-msg.axes[2], 350)

	if throttle != self.last_throttle:
            self.throttle_pub.publish(throttle)
            self.last_throttle = throttle

        if steer != self.last_steer:
            self.steering_pub.publish(steer)
            self.last_steer = steer


if __name__ == '__main__':
 try:
     RemoteControl()
 except rospy.ROSInterruptException:
     pass
