import numpy as np
import cv2
from skimage.exposure import equalize_adapthist


def binarize_img(img, min_thres, max_thres):
    return cv2.inRange(img, min_thres, max_thres) / 255


def equalize_channel(*args):
    results = []
    for c in args:
        results.append(cv2.equalizeHist(c))

    if len(results) == 1:
        return results[0]
    else:
        return results


def equalize_adapthist_channel(*args, **kwargs):
    results = []
    kernel_size = kwargs.get("kernel_size", None)
    clip_limit = kwargs.get("clip_limit", 0.01)
    nbins=kwargs.get("nbins", 256)
    for c in args:
        h,w = c.shape[0:2]
        if kernel_size is not None:
            kernel_size = [max(1,h//kernel_size[0]), max(1,w//kernel_size[1])]

        c_eq = equalize_adapthist(c, kernel_size=kernel_size, clip_limit=clip_limit, nbins=nbins) * 255.0
        results.append(c_eq.astype(np.uint8))

    if len(results) == 1:
        return results[0]
    else:
        return results


def equalize_slice(img, ny=16):
    h,w = img.shape[0:2]
    y = 0
    dy = h // ny
    result = np.zeros_like(img)
    while y <= h - dy:
        slice = img[y:y+dx,:]
        slice_max = slice.max()
        slice_min = slice.min()
        divisor = max(1,slice_max-slice_min)
        result[y:y+dx,:] = (slice - slice_min) / divisor * 255.0
        #result[y:y+ny,:] = slice
        y += ny

    return (result).astype(np.uint8)


def equalize_grid(img, ny=16,nx=16):
    h,w = img.shape[0:2]
    dy = h // ny
    dx = w // nx
    y = 0
    result = np.zeros_like(img)
    while y <= h - dy:
        x = 0
        while x <= w - dx:
            tile = img[y:y+dy,x:x+dx]
            tile_max = tile.max()
            tile_min = tile.min()
            divisor = max(1,tile_max-tile_min)
            result[y:y+dy,x:x+dx] = (tile - tile_min) / divisor * 255.0
        #result[y:y+ny,:] = slice
            x+=dx
        y += dy

    return (result).astype(np.uint8)



def equalize_slice_channel(*args, **kwargs):
    results = []
    ny = kwargs.get("ny", 16)
    for c in args:
        c_eq = equalize_slice(c, ny=ny)
        results.append(c_eq)

    if len(results) == 1:
        return results[0]
    else:
        return results


def equalize_grid_channel(*args, **kwargs):
    ny = kwargs.get('ny', 16)
    nx = kwargs.get('nx', 16)
    results = []
    for c in args:
        c_eq = equalize_grid(c, ny=ny, nx=nx)
        results.append(c_eq)

    if len(results) == 1:
        return results[0]
    else:
        return results


def hls_mask(img, min_h, max_h, min_l, max_l, min_s, max_s):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    return cv2.inRange(hls, np.array([min_h,min_l,min_s]), np.array([max_h,max_l,max_s]))


def mask_hls(img, min_h, max_h, min_l, max_l, min_s, max_s):
    mask = hls_mask(img, min_h, max_h, min_l, max_l, min_s, max_s)
    return apply_mask(img, hls_mask)


def apply_mask(img, mask):
    return cv2.bitwise_and(img,img,mask=mask)


def equalize(img):
    h,l,s = split_hls(img)
    l = cv2.equalizeHist(l)
    return join_hls(h,l,s)


def normalize_img(img):
    return img/np.max(img)


def dilate(img, ksize=5):
    return cv2.dilate(img,np.ones((ksize,ksize)),iterations = 1)


def grad_x(img, min_thres, max_thres, ksize=3, ch=1):
    sobel = cv2.Sobel(img[:,:,ch], cv2.CV_64F, 1, 0, ksize=ksize)
    sobel = np.absolute(sobel)
    sobel = np.uint8(255*sobel/np.max(sobel))
    return binarize_img(sobel, min_thres, max_thres)


def grad_y(img, min_thres, max_thres, ksize=3, ch=1):
    sobel = cv2.Sobel(img[:,:,ch], cv2.CV_64F, 0, 1, ksize=ksize)
    sobel = np.absolute(sobel)
    sobel = np.uint8(255*sobel/np.max(sobel))
    return binarize_img(sobel, min_thres, max_thres)


def mag_grad(img, min_thres, max_thres, ksize=3, ch=1):
    img_ch = img[:,:,ch] if len(img.shape) == 3 else img
    sobelx = cv2.Sobel(img_ch, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(img_ch, cv2.CV_64F, 0, 1, ksize=ksize)
    mag = np.sqrt(np.square(sobelx) + np.square(sobely))
    scaled_mag = np.uint8(255*mag/np.max(mag))
    return binarize_img(scaled_mag, min_thres, max_thres)


def dir_grad(img, min_thres, max_thres, ksize=3, ch=1):
    img_ch = img[:,:,ch] if len(img.shape) == 3 else img
    sobelx = cv2.Sobel(img_ch, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(img_ch, cv2.CV_64F, 0, 1, ksize=ksize)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    dir = np.arctan2(sobely,sobelx)
    return binarize_img(dir, min_thres * np.pi, max_thres * np.pi)


def and_images(img_a, img_b):
    return np.logical_and(img_a, img_b).astype(np.uint8)


def enhance_white_yellow(img, min_l=116, min_s=80):
    yello = hls_mask(img, 13, 24, min_l, 255, min_s, 255)
    white = hls_mask(img, 0, 180, 192, 255, 0, 255)
    mask = cv2.bitwise_or(yello, white)
    return apply_mask(img, mask)


def abs_diff_channels(a,b):
    return np.abs(a.astype(np.int32) - b.astype(np.int32)).astype(np.uint8)
