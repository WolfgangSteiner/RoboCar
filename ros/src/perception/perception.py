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


class LowPassFilter(object):
    def __init__(self, filter_constant):
        self.last_value = 0.0
        self.a = filter_constant

    def __call__(self, value):
        new_value = value * self.a + self.last_value * (1.0 - self.a)
        self.last_value = new_value
        return new_value


class Perception(object):
    def __init__(self):
        self.model = None
        self.graph = None
        self.bridge = CvBridge()
        self.timer = None
        self.steering_filter = LowPassFilter(0.75)
        self.lock = Lock()

        rospy.init_node("perception", log_level=rospy.INFO)
        rospy.Subscriber("/front_camera/image_warped", Image, self.front_camera_callback,  queue_size = 1, buff_size=2**24)
        rospy.Subscriber("/autonomous_signal", Bool, self.autonomous_signal_callback)
        rospy.Subscriber("/stop_signal", Bool, self.stop_all_callback)

        self.publishers = {}
        self.publishers["steering"] = rospy.Publisher("/steering_value", Float32)
        self.publishers["throttle"] = rospy.Publisher("/throttle_value", Float32)
        
        rospy.loginfo("Done initializing perception node.")
        rospy.spin()


    def load_model(self):
        rospy.loginfo("Loading model.")
        model_file = "/home/wolfgang/.ros/model.h5"
	self.model = load_model(model_file)
        self.model._make_predict_function()
	self.graph = tf.get_default_graph()


    def autonomous_signal_callback(self, msg):
        if msg.data:
            self.timer = rospy.Timer(rospy.Duration(1.0), self.on_enter_autonomous_mode, oneshot=True)
        elif self.timer is not None:
            self.timer.shutdown()
            self.timer = None


    def on_enter_autonomous_mode(self, event):
        self.timer = None
        rospy.loginfo("Entering autonomous mode!")
        self.lock.acquire()
        self.load_model()
        self.lock.release()


    def stop_all_callback(self, msg):
        if msg.data:
            rospy.loginfo("Terminating auntonomous mode!")
            self.lock.acquire()
            self.model = None
            self.graph = None
            keras.backend.clear_session()
            self.lock.release()


    def front_camera_callback(self, msg):
        self.lock.acquire()
        if self.model is not None:
            img = self.bridge.imgmsg_to_cv2(msg)
            img = preprocess_image(img)
            img = normalize_image(img)
            X = img.reshape((1,64,64,1))
            with self.graph.as_default():
                steering_value = float(self.model.predict(X, batch_size=1))
            steering_value = self.steering_filter(steering_value)
            self.publishers["steering"].publish(steering_value)
            self.publishers["throttle"].publish(0.40)
        self.lock.release()

if __name__ == "__main__":
    try:
        Perception()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start perception node.')
