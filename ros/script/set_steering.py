#! /usr/bin/python

import os
import sys
import rospy
from std_msgs.msg import Float32


if len(sys.argv) == 1:
    delta = 0.0
elif sys.argv[1] == "left":
    delta = -1.0
elif sys.argv[1] == "right":
    delta = 1.0
elif sys.argv[1] == "center":
    delta = 0.0
else:
    delta = float(sys.argv[1])

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
