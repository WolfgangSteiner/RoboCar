#! /usr/bin/python
import rospy
from std_msgs.msg import Float32, Bool
import RPi.GPIO as GPIO
import os

pi_blaster = os.open("/dev/pi-blaster", os.O_WRONLY)

def init_gpio():
    GPIO.setmode(GPIO.BOARD)
    for i in (11,12,15,16):
        GPIO.setup(i, GPIO.OUT)

        
def flip_pins(channel, value):
    GPIO.output(channel, value)
    GPIO.output(channel+1, not value)


def truncate_velocity(vel):
    vel = abs(vel)
    if vel < 0.1:
        vel = 0.0
    elif vel > 1.0:
        vel = 10
    
    return vel


def write_velocity(vel, pin):
    os.write(pi_blaster, "%d=%.3f\n" % (pin, truncate_velocity(vel)))


def set_velocity_right(vel):
    flip_pins(11, vel < 0)
    write_velocity(vel, 20)


def set_velocity_left(vel):
    flip_pins(15, vel > 0)
    write_velocity(vel, 21)


def set_velocities(vel_left, vel_right):
    set_velocity_left(vel_left)
    set_velocity_right(vel_right)


def set_throttle_steering(vel, steer):
    if abs(vel) < 0.1:
        left_vel = steer
        right_vel = -steer
    elif steer < 0:
        left_vel = vel * (1.0 - 0.75 * abs(steer))
        right_vel = vel 
    else:
        left_vel = vel 
        right_vel = vel * (1.0 - 0.75 * abs(steer))
                
    set_velocities(left_vel, right_vel)    


def on_exit():
    set_velocities(0.0, 0.0)


steering_value = 0.0
throttle_value = 0.0
stop_signal = False


def on_steering_value(msg):
    global steering_value
    steering_value = msg.data
    if not stop_signal:
        set_throttle_steering(throttle_value, steering_value)


def on_throttle_value(msg):
    global throttle_value
    throttle_value = msg.data
    if not stop_signal:
        set_throttle_steering(throttle_value, steering_value)


def on_stop_signal(msg):
    global stop_signal
    stop_signal = msg.data
    if stop_signal:
        set_velocities(0.0, 0.0)


import atexit
atexit.register(on_exit)
init_gpio()
set_velocities(0.0, 0.0)
rospy.init_node("pi_motor_controller")
rospy.Subscriber("/steering_value", Float32, on_steering_value)
rospy.Subscriber("/throttle_value", Float32, on_throttle_value)
rospy.Subscriber("/stop_signal", Bool, on_stop_signal)
rospy.spin()
