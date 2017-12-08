#!/usr/bin/python

# Start up ROS pieces.
#PKG = 'my_package'
import roslib; #roslib.load_manifest(PKG)
import rosbag
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from Utils import date_file_name
from glob import glob
import yaml
import argparse
from Common import preprocess_image

# Reading bag filename from command line or roslaunch parameter.
import os
import sys


parser = argparse.ArgumentParser()
parser.add_argument("src_dir", type=str)
parser.add_argument("dst_dir", type=str)
args = parser.parse_args()

bridge = CvBridge()


def write_metadata(image_path, steering_value, throttle_value):
    d = {'steering_value': steering_value, 'throttle_value': throttle_value}
    with open(image_path + ".yaml", 'w') as yaml_file:
        yaml.dump(d, yaml_file)


def extract_rosbag(bag_file, out_path):
    steering_value = 0.0 
    throttle_value = 0.0

    with rosbag.Bag(bag_file, 'r') as bag:
        for topic, msg, t in bag.read_messages():
            if topic == "/steering_value":
                steering_value = msg.data

            elif topic == "/throttle_value":
                throttle_value = msg.data

            # Legacy code for old rosbags:
            elif topic == "/steering_value_us":
                steering_value = (msg.data - 1500) / 350.0


            # Legacy code for old rosbags:
            elif topic == "/throttle_value_us":
                throttle_value = (msg.data - 1500) / 200.0


            elif topic == "/front_camera/image_warped":
                try:
                    cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
                except CvBridgeError, e:
                    print e
                    continue

                cv_image = preprocess_image(cv_image)
                image_path = os.path.join(out_path, "%06d" % msg.header.seq)
                print image_path
                cv2.imwrite(image_path + ".png", cv_image)
                write_metadata(image_path, steering_value, throttle_value) 


for bag_file in glob(os.path.join(args.src_dir, "*.bag")):
    out_path = os.path.join(args.dst_dir, os.path.splitext(os.path.split(bag_file)[1])[0])
   
    if not os.path.exists(out_path):
        print "Extracting %s ..." % bag_file 
        os.makedirs(out_path)
        extract_rosbag(bag_file, out_path)
    
    



