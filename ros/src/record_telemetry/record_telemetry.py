#!/usr/bin/python

import rospy
import rosbag
import datetime
from std_msgs.msg import Int16, Bool
from sensor_msgs.msg import Image
from threading import Lock


def bag_filename(dir="."):
    new_file_name = datetime.datetime.now().isoformat('_').replace(":","-") + ".bag"
    if dir != ".":
        new_file_name = dir + '/' + new_file_name

    return new_file_name  


def make_int_msg(value):
    msg = Int16()
    msg.data = value
    return msg


class RecordTelemetry(object):
    def __init__(self):
        self.last_steering_us = 1500
        self.last_throttle_us = 1500
        self.lock = Lock()
        self.bag = None
        rospy.init_node('record_telemetry')
        rospy.Subscriber("/steering_value_us", Int16, self.steering_callback)
        rospy.Subscriber("/throttle_value_us", Int16, self.throttle_callback)
        rospy.Subscriber("/front_camera/image_warped", Image, self.image_callback)
        rospy.Subscriber("/record_telemetry", Bool, self.record_callback)
        
        rospy.spin()

    
    def record_callback(self, msg):
        self.lock.acquire()
        if msg.data and self.bag is None:
            self.bag = rosbag.Bag(bag_filename(), 'w')
            self.bag.write("/steering_value_us", make_int_msg(self.last_steering_us))
            self.bag.write("/throttle_value_us", make_int_msg(self.last_throttle_us))
        elif not msg.data and self.bag is not None:
            self.bag.close()
            self.bag = None
        self.lock.release()


    def steering_callback(self, msg):
        self.last_steering_us = msg.data
        self.lock.acquire()
        if self.bag is not None:
            self.bag.write("/steering_value_us", msg) 
        self.lock.release()


    def throttle_callback(self, msg):
        self.last_throttle_us = msg.data
        self.lock.acquire()
        if self.bag is not None:
            self.bag.write("/throttle_value_us", msg) 
        self.lock.release()


    def image_callback(self, msg):
        self.lock.acquire()
        if self.bag is not None:
            self.bag.write("/front_camera/image_warped", msg)
        self.lock.release()


if __name__ == "__main__":
    try:
        RecordTelemetry()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start telemetry recorder node.')

