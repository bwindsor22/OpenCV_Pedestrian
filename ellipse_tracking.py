#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 20 17:42:52 2017
"""
import load_pedestrian as ld
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math as math
import matplotlib.cm as cm

"""
TODO: 
Non-Maximum suppression for ellipses
Add Kalman Filters
Add center of ellipse
Estimate tracks by finding nearest neighbors
Implement tracks tracking similar to load_pedestrian
"""

class points_tracker:
    def __init__(self, min_points):
        self.min_points = min_points
        self.min_idle_time = 30
        self.min_distance_from_last_point = 75

        self.archive_tracks = []
        self.active_tracks = []
        self.frame_idx = 0
        
    def update_tracks(self, shape_centers):
        self.add_centers_to_tracks(shape_centers)
        self.archive_old_tracks()
            
    def add_centers_to_tracks(self, shape_centers):
        if len(self.active_tracks) == 0:
            for cr in shape_centers:
                self.active_tracks.append([cr])
        else:
            track_ends = [tr[-1] for tr in self.active_tracks]
            for cr in shape_centers:
                min_index, min_len = self.find_closest_end(cr, track_ends)
                if min_len < self.min_distance_from_last_point: 
                    self.active_tracks[min_index].append(cr)
                else:
                    self.active_tracks.append([cr])

    def find_closest_end(self, cr, track_ends):        
        min_len = 99999
        min_index = 0
        for i, end in enumerate(track_ends):
            dist = self.dist_between_points(cr, end)
            if dist < min_len:
                min_len = dist
                min_index = i
        return(min_index, min_len)

    def dist_between_points(self, p0, p1):
        return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)
    
    def archive_old_tracks(self):
        for tr in self.active_tracks:
            if self.track_is_idling(tr):
                self.archive_tracks.append(tr)
                self.active_tracks.remove(tr)    

    def track_is_idling(self,tr):
        time_since_last_update = self.frame_idx - tr[-1][2]
        return time_since_last_update > self.min_idle_time
    
    def get_drawable_tracks(self):
        drawable_tracks = []
        for track in self.active_tracks:
            if len(track) >= self.min_points:
                drawable_tracks.append(track)
        for track in self.archive_tracks:
            if len(track) >= self.min_points:
                drawable_tracks.append(track)
        return(drawable_tracks)    

    def draw_active_tracks(self, frame_clr):
        drawable_tracks = self.get_drawable_tracks()
        for index, tr in enumerate(drawable_tracks):
            track_color = tuple(256 * x for x in cm.Spectral(index * 100 % 255)[0:3])
            cv2.polylines(frame_clr, [np.int32([tup[0:2] for tup in tr])], False, track_color)


class bkg_subtractor:
    def __init__(self, erode=5, dialate=8, kernel_size=3):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size,kernel_size))
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        self.erode = 3
        self.dialate = 3

    def process_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = self.fgbg.apply(frame)
        frame = self.erode_dialate(frame) 
        return( frame )       
   
    def erode_dialate(self, fgmask):
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel) 
        fgmask = cv2.erode(fgmask, self.kernel,iterations = self.erode)
        fgmask = cv2.dilate(fgmask, self.kernel, iterations = self.dialate)
        return(fgmask)

def run_settings(rng):
    if rng == "all":
        return(0, 4000)
    if rng == "person_horizontal":
        return(1200, 1500)
    if rng == "person_vertical":
        return(700,900)
    if rng == "two_horizontal":
        return(2433, 2600)
    if rng == "clear_one_horizontal":
        return(2650, 2876)
    if rng == "multiple people":
        return(3742, 4000)
    if rng == "mixed_pair":
        return(250, 370)
    if rng == "non-maxima one case": 
        # less than 200 area gives overlapping ellipses
        # need to implent algorithm to detect overlap
        # simpler algorithm: 2D rotated rectangle collision
        return(2650, 2701)
    return(0, 500)

def get_contours(frame, area=0):
    """
    Returns contours from a num with filtered area
    """
    _, contours, hierarchy = cv2.findContours(frame, 1, 2)
    
    return contours

def get_ellipses(frame_bw, frame_clr, Pt, area=0):
    """
    Draws ellipses on contours. 
    Uses area a lower bound for what to remove.
    """

    contours = get_contours(frame_bw, area)
    ellipses = []
    shape_centers = []
    for contour in contours:
        if cv2.contourArea(contour) <= area:
            continue
        try:
            ellipse = cv2.fitEllipse(contour)
            ellipses.append(ellipse)
        except Exception:
            continue
    
    for ellipse in ellipses:
        cv2.ellipse(frame_clr, ellipse, (255, 255, 255), 2)
        
        center = (int(ellipse[0][0]), int(ellipse[0][1]) )
        cv2.circle(frame_clr, center, 3, (0, 0, 255), thickness=3)
        shape_centers.append( (center[0], center[1], Pt.frame_idx) )
    
    Pt.update_tracks(shape_centers)
    Pt.draw_active_tracks(frame_clr)
    
    return frame_clr, ellipses

out_dir, cam = ld.load_dataset("oculus")
start_frame, end_frame = run_settings("all")


Bkg = bkg_subtractor()
Pt = points_tracker(5)
cam.set(1, start_frame)
for frame_idx in range(start_frame, end_frame):
    Pt.frame_idx = frame_idx
    
    ret, frame = cam.read()
    if not ret:
        print("end of reel")
        break

    print(frame_idx)
    
    frame = frame[220:720, 0:800]
    
    frame_bw = Bkg.process_frame(frame)
    
    frame_clr = frame.copy()

    
    """
    plt.figure()
    plt.title('after bckground sub')
    plt.imshow(frame_bw, cmap="gray")
    plt.show()
    """

    frame_clr, ellipses = get_ellipses(frame_bw, frame_clr, Pt, area=500)
    
    plt.figure()
    plt.title('after bckground sub')
    plt.imshow(frame_clr, cmap="gray")
    plt.show()
