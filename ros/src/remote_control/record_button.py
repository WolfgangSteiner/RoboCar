#! /usr/bin/python

import rospy
from std_msgs.msg import Int16, Float32, Bool
import RPi.GPIO as GPIO
import time


def on_shutdown():
    GPIO.cleanup()
    pub.publish(False)


if __name__ == "__main__":
    rospy.init_node('record_button')
    is_recording = False
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(18, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    pub = rospy.Publisher('/record_telemetry', Bool, queue_size=128)
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        button_state = not GPIO.input(18)
        if not button_state and is_recording:
            is_recording = False
            pub.publish(False)
        elif button_state:
            is_recording = True
            pub.publish(True)
        
        rate.sleep()

