from __future__ import print_function
import pickle
import numpy as np
import cv2
import glob
import yaml
import pickle, gzip
import os

def load_data(dir):
    data = []
    for png_file in glob.glob(dir + "/**/*.png", recursive=True):
        yaml_file = png_file.replace(".png", ".yaml")
        with open(yaml_file) as yf:
            labels = yaml.load(yf)
        
        data.append((png_file, labels["steering_value"]))

    return data


def load_data_from_pickle_file(pickle_file):
    data = []
    print(f"{pickle_file} ... ", end='')
    with gzip.open(pickle_file, "rb") as f:
        while True:
            try:          
                data.append(pickle.load(f, encoding='latin-1'))
            except EOFError:
                break

    print(f"{len(data)} images")
    return data


def load_pickled_data(path):
    if path.endswith(".pgz"):
        return load_data_from_pickle_file(path)
    else:
        data = []
        pickle_files = glob.glob(os.path.join(path, "**/*.pgz"), recursive=True)
        pickle_files.sort()
        for pickle_file in pickle_files:
            data += (load_data_from_pickle_file(pickle_file))
        print(f"Total: {len(data)} images")
        return data


def preprocess_image(img):
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = img[160:,:]
    img = cv2.resize(img, (64,64), interpolation=cv2.INTER_CUBIC)
    return img


def resize_image(img, size):
    return cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)


def crop_image(img, crop_x, crop_y):
    h,w = img.shape
    x1,x2 = crop_x
    y1,y2 = crop_y
    return img[y1:h-y2,x1:w-x2]


def normalize_image(img):
    return img.astype('float32') / 255.0 - 0.5


if __name__ == "__main__":
    data = load_data("center_01")
    for d in data:
        print(d)
