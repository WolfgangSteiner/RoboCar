#!/usr/bin/python

import cv2
import Common
import sys
import os

robot = os.environ.get("ROBOT",  "bumblebee")
print(robot)

if len(sys.argv) == 1:
    path = f"/home/wolfgang/RoboCar/data.{robot}"
else:
    path = sys.argv[1]


data = Common.load_pickled_data(path)

print(f"{len(data)} images")

def plot_steering(img, delta):
    h,w,_ = img.shape
    x1, x2 = 16, w - 16
    dx = x2 - x1
    cx = x1 + dx // 2
    y1, y2 = h - 32, h - 16
    cv2.rectangle(img, (x1,y1), (x2,y2), (128,128,128))
    pos = int(0.5 * delta * dx) + cx
    cv2.rectangle(img, (pos - 2, y1), (pos + 2, y2), (0,255,0))



for img, steering, throttle in data:
    h,w = img.shape[:2]
    h*=4
    w*=4
    img = cv2.resize(img, (w, h))
    if not len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#    cv2.putText(img, "%.2f" % steering, (4, h - 4), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0))
    plot_steering(img, steering)
    cv2.imshow("image", img)
    cv2.waitKey(40)
