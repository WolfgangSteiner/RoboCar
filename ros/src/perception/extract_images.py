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


# Reading bag filename from command line or roslaunch parameter.
import os
import sys

rosbag_dir = "../../rosbags"
data_dir = "./data"

bridge = CvBridge()

def extract_rosbag(bag_file, out_path):
    current_steering_value = 1500

    with rosbag.Bag(bag_file, 'r') as bag:
        try:
            for topic, msg, t in bag.read_messages():
                if topic == "/steering_value_us":
                    current_steering_value = msg.data

                elif topic == "/front_camera/image_warped":
                    try:
                        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
                    except CvBridgeError, e:
                        print e
                    image_name = "%06d" % msg.header.seq
                    image_name += "_%d" % current_steering_value
                    image_name += ".png"
                    image_name = os.path.join(out_path, image_name)
                    print image_name
                    cv2.imwrite(image_name, cv_image)
        except:
            pass


for bag_file in glob(os.path.join(rosbag_dir, "*.bag")):
    out_path = os.path.join(data_dir, os.path.splitext(os.path.split(bag_file)[1])[0])
   
    if not os.path.exists(out_path):
        print "Extracting %s ..." % bag_file 
        os.makedirs(out_path)
        extract_rosbag(bag_file, out_path)
    
    



