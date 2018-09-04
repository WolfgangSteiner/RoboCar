#!/usr/bin/python

from cv_bridge import CvBridge
import cv2
import numpy as np
import rospy
import imageutils
from sensor_msgs.msg import Image
from std_msgs.msg import Int16, Float32, Bool
from Common import preprocess_image, normalize_image
from keras.models import load_model
import keras
import tensorflow as tf
from threading import Lock
import time
import hashlib

model_file = "/home/wolfgang/.ros/model.h5"


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()



class LowPassFilter(object):
    def __init__(self, filter_constant, initial_value=0.0):
        self.last_value = initial_value
        self.a = filter_constant

    def __call__(self, value):
        new_value = value * self.a + self.last_value * (1.0 - self.a)
        self.last_value = new_value
        return new_value


class Perception(object):
    def __init__(self):
        self.model = None
        self.is_autonomous = False
        self.model_md5 = ""
        self.graph = None
        self.bridge = CvBridge()
        self.timer = None
        self.steering_filter = LowPassFilter(0.1)
        self.latency_filter = LowPassFilter(0.1)
        self.lock = Lock()
        self.throttle_value_max = 0.35
        self.throttle_value_min = 0.35

        self.last_timestamp = time.time()
        
        self.publishers = {}
        self.publishers["steering"] = rospy.Publisher("/steering_value", Float32)
        self.publishers["throttle"] = rospy.Publisher("/throttle_value", Float32)
        self.publishers["perception_latency"] = rospy.Publisher("/perception_latency", Float32)
        self.publishers["perception_fps"] = rospy.Publisher("/perception_fps", Float32)

        rospy.init_node("perception", log_level=rospy.INFO)
        rospy.Subscriber("/front_camera/image_processed", Image, self.front_camera_callback,  queue_size = 1, buff_size=2**24)
        rospy.Subscriber("/autonomous_signal", Bool, self.autonomous_signal_callback)
        rospy.Subscriber("/stop_signal", Bool, self.stop_all_callback)

        rospy.Subscriber("/throttle_value_max", Float32, self.throttle_value_max_callback)
        rospy.Subscriber("/throttle_value_min", Float32, self.throttle_value_min_callback)



        rospy.loginfo("Done initializing perception node.")
        rospy.spin()


    def load_model(self):
        if self.model_md5 != md5(model_file):
            if self.model:
                self.model = None
                self.graph = None
                keras.backend.clear_session()

            rospy.loginfo("Loading model.")
	    self.model = load_model(model_file)
            self.model._make_predict_function()
	    self.graph = tf.get_default_graph()
            self.model_md5 = md5(model_file)


    def autonomous_signal_callback(self, msg):
        if msg.data:
            self.timer = rospy.Timer(rospy.Duration(0.25), self.on_enter_autonomous_mode, oneshot=True)
        elif self.timer is not None:
            self.timer.shutdown()
            self.timer = None


    def throttle_value_max_callback(self, msg):
        if msg.data:
          self.throttle_value_max = msg.data
          rospy.loginfo("Got max throttle: %f." % self.throttle_value_max)
  
    
    def throttle_value_min_callback(self, msg):
        if msg.data:
          self.throttle_value_min = msg.data
  

    def on_enter_autonomous_mode(self, event):
        self.timer = None
        rospy.loginfo("Entering autonomous mode!")
        self.lock.acquire()
        self.load_model()
        self.is_autonomous = True
        self.lock.release()


    def stop_all_callback(self, msg):
        if msg.data:
            rospy.loginfo("Terminating auntonomous mode!")
            self.lock.acquire()
            self.is_autonomous = False
            self.lock.release()


    def calc_throttle(self, mean_steering_value):
        delta = abs(mean_steering_value)
        min_t = self.throttle_value_min
        max_t = self.throttle_value_max
        delta_thres = 0.25

        if (delta < delta_thres):
            return max_t
        else:
            inter = (delta - delta_thres) / (1.0 - delta_thres)
            return min_t + (max_t - min_t) * (1.0 - inter)


    def front_camera_callback(self, msg):
        current_timestamp = time.time()
        current_latency = current_timestamp - self.last_timestamp
        current_latency = self.latency_filter(current_latency)
        self.last_timestamp = current_timestamp
        self.publishers["perception_latency"].publish(current_latency)
        self.publishers["perception_fps"].publish(1/max(current_latency, 0.0001))

        self.lock.acquire()
        if self.is_autonomous:
            assert(self.model is not None)
            img = self.bridge.imgmsg_to_cv2(msg)
            img = normalize_image(img)
            X = img.reshape((1,64,64,1))
            with self.graph.as_default():
                steering_value = float(self.model.predict(X, batch_size=1))
            if abs(steering_value) > 0.5:
                steering_value *= 2.0
                steering_value = max(-1.0, min(1.0, steering_value))
            mean_steering_value = self.steering_filter(steering_value)
            throttle = self.calc_throttle(mean_steering_value)
            self.publishers["steering"].publish(steering_value)
            self.publishers["throttle"].publish(throttle)
        self.lock.release()

if __name__ == "__main__":
    try:
        Perception()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start perception node.')
