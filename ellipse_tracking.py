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
from filterpy.kalman import KalmanFilter

"""
TODO:
Separate multi-person ellipses

"""

class Track:
    def __init__(self, cr, process_noise=0.001):
        self.kf = self.intialize_kalman_filter(cr, process_noise)
        self.estimated_points = [cr]
        self.measured_points = [cr]
        
    def intialize_kalman_filter(self, cr, process_noise):
        f1 = KalmanFilter(dim_x=4, dim_z=2)
        dt = 1.# time step
        f1.F = np.array ([[1, dt, 0,  0],
                        [0,  1, 0,  0],
                        [0,  0, 1, dt],
                        [0,  0, 0,  1]])
        f1.u = 0 # no info about rotors turning
        f1.H = np.array ([[1, 0, 0, 0], #measurement to state.
                          [0, 0, 1, 0]])
        f1.R = np.array([[5,0], #  variance in measurments
                         [0, 5]])
        f1.Q = np.eye(4) * process_noise # process noise
        f1.x = np.array([[cr[0],0,cr[1],0]]).T # initial position
        f1.P = np.eye(4) * 500 #covariance matrix
        return(f1)

    def update(self, cr, frame_idx):
        self.kf.predict()
        z = np.array([[cr[0]],[cr[1]]])
        self.kf.update(z)
        est_loc = self.kf.x
        self.estimated_points.append( (est_loc[0,0], est_loc[2,0], frame_idx ) )
        self.measured_points.append( (cr[0], cr[1], frame_idx) )

        
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
                track = Track(cr)
                self.active_tracks.append(track)
        else:
            track_ends = [tr.estimated_points[-1] for tr in self.active_tracks]
            for cr in shape_centers:
                min_index, min_len = self.find_closest_end(cr, track_ends)
                if min_len < self.min_distance_from_last_point: 
                    self.active_tracks[min_index].update(cr, self.frame_idx)
                else:
                    track = Track(cr)
                    self.active_tracks.append(track)

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
        time_since_last_update = self.frame_idx - tr.estimated_points[-1][2]
        return time_since_last_update > self.min_idle_time
    
    def get_drawable_tracks(self):
        drawable_tracks = []
        for track in self.active_tracks:
            if len(track.estimated_points) >= self.min_points:
                drawable_tracks.append(track.estimated_points)
        for track in self.archive_tracks:
            if len(track.estimated_points) >= self.min_points:
                drawable_tracks.append(track.estimated_points)
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
        self.erode = erode
        self.dialate = dialate

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

run_name = "all"
out_dir, cam = ld.load_dataset("oculus")
start_frame, end_frame = run_settings(run_name)

#pictures = ld.load_picture_dataset("oculus_end")

Bkg = bkg_subtractor()
Pt = points_tracker(5)
cam.set(1, start_frame)
for frame_idx in range(start_frame, end_frame):
    Pt.frame_idx = frame_idx
    
    #frame = cv2.imread(pictures[frame_idx])
    
    ret, frame = cam.read()
    if not ret:
        print("end of reel")
        break
    
    print(frame_idx)
    
    
    frame_bw = Bkg.process_frame(frame)
    
    if frame_idx < 50:
        continue
    
    
    frame_clr = frame.copy()
    
    
    plt.figure()
    plt.title(run_name)
    plt.imshow(frame_bw, cmap="gray")
    plt.show()
    

    frame_clr, ellipses = get_ellipses(frame_bw, frame_clr, Pt, area=500)
    
    
    plt.figure()
    plt.title(run_name)
    plt.imshow(frame_clr, cmap="gray")
    plt.show()
    
    