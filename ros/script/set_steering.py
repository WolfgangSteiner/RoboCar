#! /usr/bin/python

import argparse
import os
import sys
import rospy
from std_msgs.msg import Float32

parser = argparse.ArgumentParser()
parser.add_argument("direction", type=str)
args = parser.parse_args()

if args.direction == "left":
    delta = -1.0
elif args.direction == "right":
    delta = 1.0
elif args.direction == "center":
    delta = 0.0
else:
    print("Invalid direction argument")
    exit()


def on_shutdown():
    pub.publish(0.0)
      
rospy.on_shutdown(on_shutdown)

if __name__=='__main__':
    rospy.init_node('steering_setter')
    pub=rospy.Publisher('/steering_value', Float32, queue_size=1)
    rate=rospy.Rate(10)
    
    while not rospy.is_shutdown():
        pub.publish(delta)
        rate.sleep()
        
    pub.publish(0.0) 
