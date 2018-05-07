#include <ros.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Bool.h>
#include <Servo.h>

ros::NodeHandle node_handle;

Servo steering, throttle;
const int KMaxSteer = 350;
const int KMinThrottle = -400;
const int KMaxThrottle =  400;
bool stop_value = false;


int map_value(float in_value, int min_value, int max_value)
{
  int in_value_us;
  
  if (in_value < 0.0)
  {
    in_value_us = int(in_value * abs(min_value)) + 1500;
  }
  else
  {
    in_value_us = int(in_value * max_value) + 1500;
  }
  
  return constrain(in_value_us, 1500 + min_value, 1500 + max_value);
}


void steer_callback(const std_msgs::Float32& msg)
{
  if (!stop_value)
  {
    steering.writeMicroseconds(map_value(msg.data, -KMaxSteer, KMaxSteer));
  }
}


void throttle_callback(const std_msgs::Float32& msg)
{
  if (!stop_value)
  {
    throttle.writeMicroseconds(map_value(msg.data, KMinThrottle, KMaxThrottle));
  }
}


void stop_callback(const std_msgs::Bool& msg)
{
  stop_value = msg.data;
  if (stop_value)
  {
    steering.writeMicroseconds(1500);
    throttle.writeMicroseconds(1500);
  }
}

ros::Subscriber<std_msgs::Float32> steering_sub("/steering_value", &steer_callback);
ros::Subscriber<std_msgs::Float32> throttle_sub("/throttle_value", &throttle_callback);
ros::Subscriber<std_msgs::Bool> stop_sub("/stop_signal", &stop_callback);

void setup()
{
  steering.attach(3);
  steering.writeMicroseconds(1500);

  throttle.attach(5);
  throttle.writeMicroseconds(1500);
  
  node_handle.initNode();
  node_handle.subscribe(stop_sub);
  node_handle.subscribe(steering_sub);
  node_handle.subscribe(throttle_sub);
}


void loop()
{
  node_handle.spinOnce();
  delay(1);
}

