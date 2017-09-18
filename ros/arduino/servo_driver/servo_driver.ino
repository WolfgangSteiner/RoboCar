#include <ros.h>
#include <std_msgs/Int16.h>
#include <Servo.h>

ros::NodeHandle node_handle;

Servo steering, throttle;
const int KMaxSteer = 400;
const int KMinThrottle = -200;
const int KMaxThrottle = 200;


void steer_callback(const std_msgs::Int16& msg)
{
   const int in_value = msg.data;
   const int out_value = constrain(in_value, 1500 - KMaxSteer, 1500 + KMaxSteer);
   steering.writeMicroseconds(out_value);
}


void throttle_callback(const std_msgs::Int16& msg)
{
   const int in_value = msg.data;
   const int out_value = constrain(in_value, 1500 + KMinThrottle, 1500 + KMaxThrottle);
   throttle.writeMicroseconds(out_value);
}  


ros::Subscriber<std_msgs::Int16> steering_sub("/steering_value_us", &steer_callback);
ros::Subscriber<std_msgs::Int16> throttle_sub("/throttle_value_us", &throttle_callback);


void setup()
{
  steering.attach(3);
  steering.writeMicroseconds(1500);

  throttle.attach(5);
  throttle.writeMicroseconds(1500);
  
  node_handle.initNode();
  node_handle.subscribe(steering_sub);
  node_handle.subscribe(throttle_sub);
}


void loop()
{
  node_handle.spinOnce();
  delay(1);
}

