from Color import color
import numpy as np
import cv2

def put_text(img, text, pos, font=cv2.FONT_HERSHEY_PLAIN, color=color.green, scale=1.0):
    (tw,th),baseline = cv2.getTextSize(text, font, scale, 1)
    cv2.putText(img, text, (pos[0],pos[1]+2*th), font, scale, color)
    return tw,th


def draw_marker(img, pos, size=4, color=color.red):
    x = pos[0]
    y = pos[1]
    pts = []
    pts.append((x,y-size/2))
    pts.append((x+size/2,y))
    pts.append((x,y+size/2))
    pts.append((x-size/2,y))
    pts = np.array(pts,np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=color)


def draw_pixel(img, pos, color=color.white):
    img[pos[1],pos[0],:] = color


def draw_line(img, p1, p2, color, thickness=1):
    cv2.line(img, (p1[0],p1[1]), (p2[0],p2[1]), color=color, lineType=cv2.LINE_AA, thickness=thickness)


def draw_rectangle(img, pos, size, color, thickness=1):
    p1 = np.array(pos, np.int)
    p2 = p1 + np.array(size, np.int)
    cv2.rectangle(img, (p1[0],p1[1]), (p2[0],p2[1]), color, thickness=thickness)


def fill_rectangle(img, pos, size, color):
    draw_rectangle(img, pos, size, color, thickness=cv2.FILLED)


def bordered_rectangle(img, pos, size, fill_color, border_color, thickness=1):
    fill_rectangle(img, pos, size, fill_color)
    draw_rectangle(img, pos, size, border_color, thickness)


class TextRenderer(object):
    def __init__(self, img):
        self.img = img
        self.h, self.w = self.img.shape[0:2]
        self.x = 10
        self.y = 10
        self.spacing = 30
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.scale = 1.0
        self.color = color.white


    def put_line(self, text):
        put_text(self.img, text, (self.x,self.y), font=self.font, color=self.color, scale=self.scale)
        self.y += self.spacing


    def text_at(self,text,pos,horizontal_align="left", vertical_align="top", color=None, scale=None, font=None, margin=[0.0]):
        if scale ==None:
            scale = self.scale

        (tw,th), baseline = self.calc_text_size(text,scale=scale,font=font)

        pos = np.array(pos)
        offset = np.zeros(2, np.int)
        if horizontal_align == "center":
            offset[0] = -tw/2
        if vertical_align == "bottom":
            offset[1] = - 2*margin[1]
        elif vertical_align == "top":
            offset[1] = th
        elif vertical_align == "center":
            offset[1] = th // 2

        if color == None:
            color = self.color

        if font==None:
            font = self.font
        elif font=="small":
            font = cv2.FONT_HERSHEY_PLAIN

        pos += offset
        pos += np.array(margin,np.int)
        cv2.putText(self.img, text, (pos[0],pos[1]), font, scale, color)
        return (pos, pos + np.array((tw,th)))


    def calc_bounding_box(self,text,pos,horizontal_align="left", vertical_align="top", scale=None, font=None, margin=[0.0]):
        if scale ==None:
            scale = self.scale

        if font==None:
            font = self.font
        elif font=="small":
            font = cv2.FONT_HERSHEY_PLAIN

        (tw,th), baseline = self.calc_text_size(text,scale=scale,font=font)

        pos = np.array(pos)
        offset = np.zeros(2, np.int)
        if horizontal_align == "center":
            offset[0] = -tw/2
        if vertical_align == "bottom":
            offset[1] = - 2 * margin[1]
        elif vertical_align == "top":
            offset[1] = th
        elif vertical_align == "center":
            offset[1] = th // 2

        pos += offset
        #pos -= np.array(margin,np.int)
        p1 = pos.astype(np.int) - np.array((0,+th))
        p2 = p1 + np.array((tw,th), np.int) + 2 * np.array(margin, np.int)
        return (p1,p2)



    def calc_text_size(self, text, scale=None, font=None):
        if scale == None:
            scale = self.scale
        if font==None:
            font = self.font

        return np.array(cv2.getTextSize(text, font, scale, 1))
