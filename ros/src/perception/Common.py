import pickle
import numpy as np
import cv2
import glob
import yaml
from __future__ import print_function


def load_data(dir):
    data = []
    for png_file in glob.glob(dir + "/*/*.png"):
        yaml_file = png_file.replace(".png", ".yaml")
        with open(yaml_file) as yf:
            labels = yaml.load(yf)
        
        data.append((png_file, labels["steering_value"]))

    return data


def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img[160:,:]
    img = cv2.resize(img, (64,32), interpolation=cv2.INTER_CUBIC)
    return img


def normalize_image(img):
    return img.astype('float32') / 255.0 - 0.5


if __name__ == "__main__":
    data = load_data("center_01")
    for d in data:
        print(d)
