import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFilter, ImageFont
import Common
import math, random
import os
import cv2

max_steering_angle = 20.0


def random_rotate(img, steering_angle):
    angle = np.random.uniform(-7.5, 7.5)
    return img.rotate(angle, resample=Image.BICUBIC, expand = 0), steering_angle - angle / max_steering_angle


def random_translate(img, steering_angle):
    delta_x = np.random.uniform(-40, 40)
    delta_y = np.random.uniform(-10, 10)
    img = img.transform(img.size, Image.AFFINE, (1.0, 0.0, delta_x, 0.0, 1.0, 0.0))
    angle = math.atan2(delta_x , 95.0) * 180 / math.pi / 25
    return img,steering_angle + angle


def random_mirror(img, steering_angle):
    if random.random() > 0.5:
        img = ImageOps.mirror(img)
        steering_angle *= -1

    return img, steering_angle


def reshape_img(img):
    shape = img.shape
    if len(shape) == 2:
        shape = (shape[0], shape[1], 1)

    return img.reshape(shape)


def scale_img(img, scale):
    if scale == 1:
        return img
    else:
        w,h = img.size
        return img.resize((w//scale,h//scale), Image.ANTIALIAS)


def DataGenerator(data, batch_size=32, augment_data=True, scale=1, crop_x=[0,0], crop_y=[0,0]):
    num_data = len(data)
    idx = 0

    while True:
        X = []
        y = []
        for i in range(0,batch_size):
            while True:
                img, angle, throttle = data[idx]
                idx = (idx + 1) % num_data
                if True or angle >= 0.01 or random.random() < 0.25:
                    break

            img = Image.fromarray(img)
            img, angle = random_mirror(img, angle)
            img = scale_img(img, scale)
            img = np.array(img)
            img = Common.resize_image(img, (64,64))
            img = Common.normalize_image(np.array(img))
            img = Common.crop_image(img, crop_x, crop_y)
            img = reshape_img(img)
            X.append(img)
            y.append(max(min(angle, 1.0), -1.0))

        yield np.array(X), np.array(y)


if __name__ == '__main__':
    num_cols = 8 
    num_rows = 8
    w = 64 * 4
    h = 32 * 4

    dirs = "data.6"
    data = []
    for d in dirs.split(','):
        data += Common.load_data(d)

    random.shuffle(data)

    overview_img = Image.new("RGBA",(w * num_cols, h * num_rows), (0,0,0))
    font = ImageFont.truetype("/usr/share/fonts/truetype/ttf-dejavu/DejaVuSansMono.ttf", 24)
    gen = DataGenerator(data, batch_size=num_rows * num_cols)
    X,y = gen.next()
    draw = ImageDraw.Draw(overview_img)

    for j in range(num_rows):
        for i in range(num_cols):
            idx = j*num_cols + i
            xi = i*w
            yi = j*h
            img = (X[idx] * 255.0 + 127.5).astype('uint8')
            if len(img.shape) == 3 and img.shape[2] == 1:
                img = img.reshape(img.shape[0:2])
            img = Image.fromarray(img)
            img = img.resize((w,h), Image.BILINEAR)
            angle = y[idx]
            overview_img.paste(img, (xi,yi))
            draw.text((xi,yi), "%.2f" % angle, fill=(0,255,0), font=font)

    #overview_img.show()
    overview_img.save("overview.png")
