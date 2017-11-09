import pickle
import numpy as np
import cv2
import glob


def steering_value_rel(steering_value_us):
    return (steering_value_us - 1500.) / 350.


def steering_angle_deg(steering_value_rel):
    return 20 * steering_value_rel


def load_data(dir):
    data = []
    for f in glob.glob(dir + "/*/*.png"):
        sv_us = int(f.split(".")[1].split("_")[-1])
        sv_rel = steering_value_rel(sv_us)
        steering_angle = steering_angle_deg(sv_rel)
        data.append((f, sv_rel))

    return data


def preprocess_image(img):
    img_scaled = cv2.resize(img, (64,64), interpolation=cv2.INTER_CUBIC)
    return img_scaled.astype('float32') / 255.0 - 0.5


if __name__ == "__main__":
    data = load_data("center_01")
    for d in data:
        print d
