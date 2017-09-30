import numpy as np
from Drawing import *
from lanemath import *
from collections import deque

# Define a class to receive the characteristics of each line detection
class LaneLine(object):
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients averaged over the last n iterations in meters
        self.best_fit_in_meters = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #polynomial coefficients for the most rec_ent fit in meters
        self.current_fit_in_meters = [np.array([False])]
        #polynomial coefficients for the last good fit in meters
        self.last_good_fit_in_meters = None

        self.fit_queue = deque()

        #radius of curvature of the line in some units
        self.current_radius_in_meters = None
        self.radius = LowpassFilter(0.1)

        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #relative difference in fit coefficients between last and new fits
        self.rel_diffs = np.array([0,0,0], dtype='float')
        #absolute difference in fit coefficients between last and new fits
        self.abs_diffs = np.array([0,0,0], dtype='float')


        #x values for detected line points
        self.lane_points = None

        self.histogram = None

        self.peaks = None

        # conversion factors for poly coefficients from pixels to meters
        self.pconv = np.ones(4)

        self.sliding_window_coords = []

        self.frames_without_good_fit = 0


    def initialize(self, frame_size, x_anchor, xm_per_px, ym_per_px):
        self.frame_size = frame_size
        self.fit_margin = self.frame_size[1] // 8
        self.x_anchor = x_anchor
        self.pconv[0] = xm_per_px
        self.pconv[1] = xm_per_px / ym_per_px
        self.pconv[2] = xm_per_px / (ym_per_px**2)
        self.pconv[2] = xm_per_px / (ym_per_px**3)
        self.xm_per_px = xm_per_px
        self.ym_per_px = ym_per_px


    def has_good_fit(self):
        return self.best_fit is not None and len(self.best_fit) == 3


    def interpolate_line_points(self, h):
        if self.best_fit is None:
            return None
        y = np.arange(0,h+1,h/16)
        x = poly.polyval(y,self.best_fit)
        return np.stack((x,h - y), axis=1).astype(np.int32)



    def fit_lane_line(self,img):
        h,w = img.shape
        self.peaks = []
        self.sliding_window_coords = []
        self.detected = False
        self.current_fit = None
        self.current_fit_in_meters = None

        if self.has_good_fit():
            nonzero = img.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            x_poly = poly.polyval(h - nonzeroy, self.best_fit)
            left_edge = x_poly - self.fit_margin // 2
            right_edge = x_poly + self.fit_margin // 2

            ids = ((nonzerox >= left_edge) & (nonzerox <= right_edge)).nonzero()[0]
            lane_x = nonzerox[ids]
            lane_y = h - nonzeroy[ids]
            if len(lane_x) < 6:
                self.current_fit = None
            else:
                self.current_fit = fit_cubic(lane_x, lane_y)
                self.current_fit_in_meters = self.current_fit * self.pconv
                self.process_current_fit()

            self.lane_points = np.stack((lane_x,lane_y),axis=1)
            self.detected = self.is_current_fit_good()
            self.peaks = []
            self.histogram = None

        if not self.detected:
            # compute histogram of window centered at x_anchor with half image height
            x = self.x_anchor
            x1 = int(x - w/8)
            x2 = int(x + w/8)
            histogram = np.sum(img[int(img.shape[0]/2):,x1:x2], axis=0).astype(np.int32)
            start_x = find_peak(histogram)

            # prepare histogram for visualization
            if start_x != None:
                start_x += x1
                assert(start_x <= 900)
                x_coords = np.arange(x1,x2)
                y_coords = h - 1 - histogram
                self.histogram = np.stack((x_coords,y_coords),axis=1).astype(np.int32)
                self.peaks = [np.array((start_x,y_coords[start_x-x1]), np.int32)]

            # start sliding window line detection at the peak of the histogram
            if start_x != None:
                self.detect_with_sliding_window(img, start_x)
                if self.current_fit is not None:
                    self.process_current_fit()

            self.detected = True #self.is_current_fit_good()

        if not self.detected:
            self.current_fit = None
            self.current_fit_in_meters = None

        self.update_polynomial()


    def detect_with_sliding_window(self, img, start_x):
        if not start_x is None:
            lane_x,lane_y = self.perform_sliding_window(img, start_x, self.best_fit)
            self.lane_points = np.stack((lane_x,lane_y),axis=1)

            if len(self.lane_points) > 3:
                self.current_fit = fit_cubic(lane_x, lane_y)
                self.current_fit_in_meters = self.current_fit * self.pconv
        else:
            self.lane_points = []
            self.current_fit = None


    # sliding window lane line detection
    def perform_sliding_window(self, img, start_x, best_fit):
        h,w = img.shape

        delta_y = h // 32
        delta_x = w // 4

        lane_x = []
        lane_y = []

        y = h
        x = start_x
        dy =  delta_y // 16
        dx = 0.0
        ddx = 0.0
        last_ddx = None
        i = 0

        while y > 0:
            y1 = y - delta_y
            y2 = y
            x1 = int(max(0, x - delta_x / 2))
            x2 = int(min(w, x + delta_x / 2))
            window = img[y1:y2,x1:x2]
            nonzero = window.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            if len(nonzerox > 5):
                i += 1
                new_delta_x_calc = nonzerox.mean() - delta_x // 2
                new_delta_x_calc = clip_to_range(new_delta_x_calc, -delta_x, delta_x)
                x += new_delta_x_calc

            lane_x.append(nonzerox + x1)
            lane_y.append(nonzeroy + y1)
            self.sliding_window_coords.append(((x1,y1),(x2,y2)))

            y -= delta_y

        return np.concatenate(lane_x), h - np.concatenate(lane_y)


    def process_current_fit(self):
        assert self.current_fit is not None
        self.current_fit_in_meters = self.current_fit * self.pconv
        self.current_radius_in_meters = calc_radius(self.current_fit_in_meters, 15)
        self.radius.process(self.current_radius_in_meters)

        if self.last_good_fit_in_meters is not None:
            self.rel_diffs = rel_change(self.current_fit_in_meters, self.last_good_fit_in_meters)
            self.abs_diffs = abs_change(self.current_fit_in_meters, self.last_good_fit_in_meters)


    def calc_distance_from_center(self):
        if self.has_good_fit():
            d_in_px = self.best_fit[0] - self.frame_size[1] / 2
            d_in_m = d_in_px * self.pconv[0]
            return d_in_m
        else:
            return None


    def fit_in_meters(fit):
        return fit * self.pconv


    def is_current_fit_good(self):
        if self.current_fit is None:
            return False
        else:
            return True

        # elif self.last_good_fit_in_meters is None:
        #     return True
        # else:
        #     if self.abs_diffs[0] > 0.2 * 2:
        #         return False
        #
        #     if self.rel_diffs[1] > 0.15 * 2:
        #         return False
        #
        #     if self.rel_diffs[2] > 0.15 * 2:
        #         return False
        #
        # return True


    def update_polynomial(self):
        if self.current_fit is None:
            self.frames_without_good_fit += 1
            return

        elif not self.is_current_fit_good():
            self.frames_without_good_fit += 1
            return

        elif self.best_fit is None:
            self.best_fit = self.current_fit

        else:
            self.do_update_polynomial(self.current_fit)

        self.frames_without_good_fit = 0
        self.best_fit_in_meters = self.best_fit * self.pconv
        self.last_good_fit_in_meters = self.current_fit * self.pconv


    # low-pass filtering of the current best fit
    def do_update_polynomial(self, p):
        a = 0.25
        b = 1.0 - a
        self.best_fit = a * p + b * self.best_fit


    def draw_histogram(self, img):
        if not self.histogram is None:
            h,w = img.shape[0:2]
            cv2.polylines(img, [self.histogram], isClosed=False, color=color.light_green)
            for p in self.peaks:
                draw_marker(img, p)


    def draw_lane_points(self,img):
        h,w = img.shape[0:2]
        if self.lane_points is not None:
            for (x,y) in self.lane_points:
                draw_pixel(img, (x,h-y-1), color=color.pink)


    def annotate_poly_fit(self, annotated_img):
        composite_img = np.zeros_like(annotated_img)

        h,w = annotated_img.shape[0:2]
        y = np.arange(0,h+0.1,1)

        if len(self.sliding_window_coords):
            for p1,p2 in self.sliding_window_coords:
                cv2.rectangle(composite_img, p1, p2, color=color.orange)


        elif self.best_fit is not None:
            x = poly.polyval(y, self.best_fit)
            left_edge = x - self.fit_margin // 2
            right_edge = x + self.fit_margin // 2
            coords_left = np.transpose(np.stack((left_edge,h-y), axis=0))
            coords_right = np.transpose(np.stack((right_edge,h-y), axis=0))
            coords = np.concatenate((coords_left, np.flipud(coords_right))).astype(np.int32)
            cv2.fillPoly(composite_img, [coords], color=color.light_blue)

        if self.current_fit is not None:
            x = poly.polyval(y, self.current_fit)
            poly_coords = np.stack((x,h-y), axis=1).astype(np.int32)
            cv2.polylines(annotated_img, [poly_coords], isClosed=False, color=color.white, thickness=1, lineType=cv2.LINE_AA)

        return cv2.addWeighted(annotated_img, 1.0, composite_img, 0.3, 0)
