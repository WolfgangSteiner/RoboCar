import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFilter
import Common
import math, random


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


def DataGenerator(data, batch_size=64, augment_data=True):
    num_data = len(data)
    idx = 0
    recovery_angle_a = 7.5
    recovery_angle_b = 15.0
    side_camera_angle = 0.5

    while True:
        X = []
        y = []
        for i in range(0,batch_size):
            while True:
                file_name, angle = data[idx]
                steering_angle = max_steering_angle * angle
                idx = (idx + 1) % num_data
                if True or angle >= 0.01 or random.random() < 0.25:
                    break

            img = Image.open(file_name)
            dir_name,file_name = file_name.split('/')

            if dir_name.startswith("recovery_left"):
                angle += recovery_angle_b / max_steering_angle
            elif dir_name.startswith("recovery_right"):
                angle -= recovery_angle_b / max_steering_angle

            if dir_name.startswith("lane_left"):
                angle += recovery_angle_a / max_steering_angle
            elif dir_name.startswith("lane_right"):
                angle -= recovery_angle_a / max_steering_angle

            if False and augment_data:
                 img, angle = random_rotate(img, angle)
#                img, angle = random_translate(img, angle)

            img, angle = random_mirror(img, angle)
            img_data = Common.preprocess_image(np.array(img)).reshape((64,64,1))
            X.append(img_data)
            y.append(max(min(angle, 1.0), -1.0))

        yield np.array(X), np.array(y)


if __name__ == '__main__':
    num_cols = 24
    num_rows = 16
    w = 64
    h = 64

    dirs = "center_01"#,recovery_left_01,recovery_right_01,lane_left_01,lane_right_01"
    #dirs = "center_01"
    data = []
    for d in dirs.split(','):
        data += Common.load_data(d)

    #random.shuffle(data)

    overview_img = Image.new("RGBA",(w * num_cols, h * num_rows), (0,0,0))
    gen = DataGenerator(data, batch_size=num_rows * num_cols)
    X,y = gen.next()
    draw = ImageDraw.Draw(overview_img)

    for j in range(num_rows):
        for i in range(num_cols):
            idx = j*num_cols + i
            xi = i*w
            yi = j*h
            img = Image.fromarray((X[idx] * 255.0 + 127.5).astype('uint8'))
            img.thumbnail((w,h), Image.ANTIALIAS)
            angle = y[idx]
            overview_img.paste(img, (xi,yi))
            draw.text((xi,yi), "%.2f" % angle, fill = (255,0,0))

    overview_img.show()
