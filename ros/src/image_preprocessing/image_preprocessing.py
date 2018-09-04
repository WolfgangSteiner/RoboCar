#!/usr/bin/python

from cv_bridge import CvBridge
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image


class ImagePreprocessing(object):
    def __init__(self):
        self.bridge = CvBridge()
        self.warped_size = (900,900)
        self.calc_perspective_transform()
        rospy.init_node("image_preprocessing")
        rospy.Subscriber("/front_camera/image_rect", Image, self.front_camera_callback,  queue_size = 1, buff_size=2**24)
        self.publishers = {}
        self.add_image_publisher("processed")

        rospy.spin()


    def add_image_publisher(self, name):
        self.publishers[name] = rospy.Publisher("/front_camera/image_" + name, Image)


    def publish_image(self, name, img, flip=None, encoding="passthrough"):
        if flip == 'y':
            img = cv2.flip(img,0)

        self.publishers[name].publish(self.bridge.cv2_to_imgmsg(img, encoding))


    def calc_perspective_transform(self):
        c = self.warped_size[1] // 2
        wp = 280
        hp = 210
        dp = 300

        src_coords = np.array([[269,200],[386,200],[405,281],[248,281]], dtype=np.float32)
        #src_coords = np.array([[257,282],[399,282],[445,349],[209,349]], dtype=np.float32)
        dst_coords = np.array([[c-wp//2, dp+hp],[c+wp//2,dp+hp],[c+wp//2,dp],[c-wp//2,dp]], dtype=np.float32)

        self.M = cv2.getPerspectiveTransform(src_coords, dst_coords)


    def warp_perspective(self, img):
        return cv2.warpPerspective(img, self.M, self.warped_size)


    def front_camera_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg)
        img_warped = self.warp_perspective(img)
        img_warped_scaled = cv2.resize(img_warped, (64,64), interpolation=cv2.INTER_CUBIC)
        self.publish_image("processed", img_warped_scaled, flip='y')


if __name__ == "__main__":
    try:
        ImagePreprocessing()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start image preprocessing node.')
