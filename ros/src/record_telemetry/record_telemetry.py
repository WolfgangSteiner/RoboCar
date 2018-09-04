#!/usr/bin/python

import rospy
from datetime import datetime
from std_msgs.msg import Int16, Float32, Bool
from sensor_msgs.msg import Image
from threading import Lock
import os
import pickle
import gzip
import cv_bridge

hostname = os.uname()[1]


def pickle_filename(dir="/home/wolfgang/RoboCar/data"):
    now = datetime.now()
    dst_dir = os.path.join(dir, datetime.date(now).isoformat())
    new_file_name = now.isoformat('_').replace(":","-").split('.')[0] + ".pgz"
    os.system("mkdir -p %s" % dst_dir)
    new_file_name = os.path.join(dst_dir, new_file_name)

    return new_file_name  


def make_float_msg(value):
    msg = Float32()
    msg.data = value
    return msg


def make_int_msg(value):
    msg = Int16()
    msg.data = value
    return msg


class RecordTelemetry(object):
    def __init__(self):
        self.bridge = cv_bridge.CvBridge()
        self.last_steering = 0.0
        self.last_throttle = 0.0
        self.last_steering_offset = 0.0
        self.lock = Lock()
        self.pickle = None
        rospy.init_node('record_telemetry')
        rospy.Subscriber("/steering_value", Float32, self.steering_callback)
        rospy.Subscriber("/steering_offset", Float32, self.steering_offset_callback)
        rospy.Subscriber("/throttle_value", Float32, self.throttle_callback)
        rospy.Subscriber("/record_telemetry", Bool, self.record_callback)
        rospy.Subscriber("/front_camera/image_processed", Image, self.image_processed_callback)
        rospy.spin()

    
    def record_callback(self, msg):
        self.lock.acquire()
        if msg.data and self.pickle is None:
            self.pickle = gzip.open(pickle_filename(), 'wb')
        elif not msg.data and self.pickle is not None:
            self.pickle.close()
            self.pickle = None
        self.lock.release()


    def steering_callback(self, msg):
        self.lock.acquire()
        self.last_steering = msg.data
        self.lock.release()

    
    def steering_offset_callback(self, msg):
        self.lock.acquire()
        self.last_steering_offset = msg.data
        self.lock.release()


    def throttle_callback(self, msg):
        self.lock.acquire()
        self.last_throttle = msg.data
        self.lock.release()


    def current_steering(self):
        return max(-1.0, min(1.0, self.last_steering + self.last_steering_offset))


    def image_processed_callback(self, msg): 
        self.lock.acquire()
        if self.pickle is not None:
            img = self.bridge.imgmsg_to_cv2(msg)
            pickle.dump((img, self.current_steering(), self.last_throttle), self.pickle)               
        self.lock.release()


if __name__ == "__main__":
    try:
        RecordTelemetry()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start telemetry recorder node.')

