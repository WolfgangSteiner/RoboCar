from cv_bridge import CvBridge
import cv2
import numpy as np
import rospy
import imageutils
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from image_thresholding import dir_grad, mag_grad, equalize_adapthist_channel
from LaneLine import LaneLine
from Common import preprocess_image
from keras.models import model_from_json
import json


class Perception(object):
    def __init__(self):
        self.model = None
        self.bridge = CvBridge()
        self.warped_size = (900,900)
        self.calc_perspective_transform()
        self.canny_mask = None
        rospy.init_node("percption")
        rospy.Subscriber("/front_camera/image_rect_color", Image, self.front_camera_callback,  queue_size = 1, buff_size=2**24)
        self.publishers = {}
        self.xm_per_px = 0.001
        self.ym_per_px = 0.001
        self.input_frame_size = (900, 900)
        self.left_lane_line = LaneLine()
        self.right_lane_line = LaneLine()
        self.x_anchor_left = 300
        self.x_anchor_right = 600

        # self.left_lane_line.initialize(
        #     self.input_frame_size,
        #     self.x_anchor_left,
        #     self.xm_per_px, self.ym_per_px)
        #
        # self.right_lane_line.initialize(
        #     self.input_frame_size,
        #     self.x_anchor_right,
        #     self.xm_per_px, self.ym_per_px)

        self.load_model()

        for topic in ("warped",):
            self.add_image_publisher(topic)

        self.publishers["steering_rel"] = rospy.Publisher("/steering_angle_rel", Float32)
        rospy.spin()


    def load_model(self):
        model_file = "model.json"
        with open(model_file, 'r') as jfile:
            self.model = model_from_json(json.loads(jfile.read()))

        self.model.compile("adam", "mse")
        weights_file = model_file.replace('json', 'h5')
        self.model.load_weights(weights_file)


    def add_image_publisher(self, name):
        self.publishers[name] = rospy.Publisher("/front_camera/image_" + name, Image)


    def publish_image(self, name, img, flip=None, encoding="passthrough"):
        if flip == 'y':
            img = imageutils.flip_y(img)

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


    def gray_threshold(self, img_gray):
        return cv2.inRange(img_gray, 192, 255)


    def yuv_threshold(self, img_yuv):
        y,u,v = imageutils.split_channels(img_yuv)
        y_thres = cv2.inRange(y, 128, 255)
        u_thres = cv2.inRange(v, 0, 128)
        return imageutils.AND(y_thres, u_thres)


    def canny_edge_detection(self, img):
        img_blur = cv2.blur(img, ksize=(3,3))
        return cv2.Canny(img_blur, 64, 160, 9)


    def mask_canny(self, img):
        if self.canny_mask is None:
            self.canny_mask = np.ones(img.shape, dtype=img.dtype) * 255
            # Perspective A:
            #x = np.array([0, 0, 232, 359, 359, 537, 537, 899, 899])
            #y = 735 - np.array([899, 549, 735, 735, 704, 704, 735, 528, 899])

            # Perspective B:
            x =       np.array([  0,   0,  95, 372, 372, 538, 538, 798, 899, 899])
            y = 899 - np.array([899, 737, 833, 833, 812, 812, 833, 833, 717, 899])

            poly_pts = np.stack((x,y), axis = -1)
            cv2.fillPoly(self.canny_mask, [poly_pts], 0)

        return imageutils.AND(img, self.canny_mask)


    def crop_warped_image(self, img):
        return img[899-735:,:]


    def annotate_lane_lines(self, img):
        self.left_lane_line.draw_lane_points(img)
        self.right_lane_line.draw_lane_points(img)
        #self.left_lane_line.draw_histogram(img)
        #self.right_lane_line.draw_histogram(img)
        img = self.left_lane_line.annotate_poly_fit(img)
        img = self.right_lane_line.annotate_poly_fit(img)
        return img


    def front_camera_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg)
        img_gray = imageutils.bgr2gray(img)
        img_warped_gray = self.warp_perspective(img_gray)
        img_warped_gray = equalize_adapthist_channel(img_warped_gray)
        img_warped_gray_scaled = cv2.resize(img_warped_gray, (256,256), interpolation=cv2.INTER_CUBIC)

        self.publish_image("warped", img_warped_gray_scaled, flip='y')

        X = preprocess_image(img_warped_gray_scaled).reshape((1,64,64,1))
        steering_angle_rel = float(self.model.predict(X, batch_size=1))
        print steering_angle_rel
        self.publishers["steering_rel"].publish(steering_angle_rel)


        # img_canny = self.canny_edge_detection(img_gray)
        # self.publish_image("canny", img_canny)

        #img_warped_gray = self.crop_warped_image(img_warped_gray)
        #img_warped_canny = self.canny_edge_detection(img_warped_gray)
        #img_warped_canny = self.mask_canny(img_warped_canny)
        #self.publish_image("warped_canny", img_warped_canny, flip='y')


        #img_warped_gray_scaled = cv2.resize(img_warped_gray, (64,64), interpolation=cv2.INTER_LANCZOS4)
        #self.publish_image("warped_scaled", img_warped_gray_scaled, flip='y')
        #img_warped_canny_scaled = self.canny_edge_detection(img_warped_gray_scaled)
        #self.publish_image("warped_canny_scaled", img_warped_canny_scaled)

        # img_warped_canny = imageutils.flip_y(img_warped_canny)
        # self.left_lane_line.fit_lane_line(img_warped_canny)
        # self.right_lane_line.fit_lane_line(img_warped_canny)
        #
        # annotated_img = imageutils.expand_mask(img_warped_canny * 255)
        # annotated_img = self.annotate_lane_lines(annotated_img)
        # self.publish_image("lane_lines", annotated_img, encoding="rgb8")
        # img_gray = imageutils.bgr2gray(img)
        # img_gray_threshold = self.gray_threshold(img_gray)
        # self.publish_image("gray_threshold", img_gray_threshold)
        #
        # img_yuv = imageutils.bgr2yuv(img)
        # img_yuv_threshold = self.yuv_threshold(img_yuv)
        # self.publish_image("yuv_threshold", img_yuv_threshold)
        #
        # y,u,v = imageutils.split_channels(img_yuv)
        # y_mag_grad = mag_grad(y, 24, 255, 3)
        # y_mag_grad = imageutils.AND(y_mag_grad, img_yuv_threshold) * 255
        # #y_mag_grad = imageutils.expand_mask(y_mag_grad)
        # self.publish_image("mag_grad", y_mag_grad)




if __name__ == "__main__":
    try:
        Perception()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start perception node.')
