#!/usr/bin/env python
import rospy
from std_msgs.msg import String,Int16,Bool
from sensor_msgs.msg import Joy
import numpy as np


class ButtonHandler(object):
    def __init__(self, button_idx, topic_name, on_true=None):
        self.publisher = rospy.Publisher(topic_name, Bool, queue_size=128)
        self.button_idx = button_idx
        self.state = False
        self.on_true = on_true


    def eval(self, msg):
        new_state = msg.buttons[self.button_idx] == 1
        
        if self.state != new_state:
            self.state = new_state
            self.publisher.publish(new_state)
            if self.state and self.on_true is not None:
                self.on_true()
    

class RemoteControl(object):
    def __init__(self):
        self.max_throttle = 200
        self.joy_deadband = 0.1
        self.last_steer = 1500
        self.last_throttle = 1500
        self.fixed_throttle_slow = 1610
        self.fixed_throttle_fast = 1650

        rospy.init_node('remote_control')
        self.publisher = {}
        self.publisher["steering"] = rospy.Publisher('/steering_value_us', Int16, queue_size=128)
        self.publisher["throttle"] = rospy.Publisher('/throttle_value_us', Int16, queue_size=128)
        self.publisher["stop"] = rospy.Publisher('/stop_signal', Bool, queue_size=128)
        rospy.Subscriber("/joy", Joy, self.joy_cb)

        self.button_handlers = []
        self.add_button_handler(3, "/autonomous_signal")     
        self.add_button_handler(4, "/record_telemetry")     
        self.add_button_handler(1, "/stop_signal", self.on_all_stop)

        rospy.spin()


    def add_button_handler(self, button_idx, topic_name, on_true=None):
        self.button_handlers.append(ButtonHandler(button_idx, topic_name, on_true=on_true))


    def map_value(self, joy_value, max_displacement):
        value_sign = np.sign(joy_value)
        value = abs(joy_value)
        if value < self.joy_deadband:
            return 1500
        else:
            value -= self.joy_deadband
            return 1500 + max_displacement * value_sign * value / (1.0 - self.joy_deadband)


    def on_all_stop(self):
        self.publisher["throttle"].publish(1500)
        self.publisher["steering"].publish(1500)


    def joy_cb(self, msg):
        for handler in self.button_handlers:
           handler.eval(msg) 


        if msg.axes[2] < 0.0:
            throttle = self.fixed_throttle_slow
        elif msg.axes[5] < 0.0:
            throttle = self.fixed_throttle_fast
        else:
            throttle = self.map_value(msg.axes[1], 200)
            
        steer = self.map_value(-msg.axes[3], 350)

        if throttle != self.last_throttle:
            self.publisher["throttle"].publish(throttle)
            self.last_throttle = throttle

        if steer != self.last_steer:
            self.publisher["steering"].publish(steer)
            self.last_steer = steer


if __name__ == '__main__':
 try:
     RemoteControl()
 except rospy.ROSInterruptException:
     pass
