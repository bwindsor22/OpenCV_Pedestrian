#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 19:08:32 2017

@author: brad
"""
# three things we're looking for:
# 1 - heat map of common locations
# 2 - count of people per frames
# 3 - flow map of traffic
# TODO
# differentiate between new tracks and old tracks
# More intelligently deal with subtracks
# Collect settings into one place
# eliminate the notion of a tuple


#%%
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.cluster.hierarchy as hac
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from dtw import dtw
from time import clock
import load_pedestrian as ld



#%% Load Settings

def load_dataset( dataset, base_dir ):
    if dataset == "hallway":
        out_dir = base_dir + 'outputFiles/hallway/'
        video_file = 'hallway_cropped_long.mp4'
    if dataset == "oculus":
        out_dir = base_dir + 'outputFiles/Oculus/'
        video_file = 'pedestrian_datasets_oculus/Oculus.mp4'
    if dataset == "cafe":
        out_dir = base_dir + 'outputFiles/cafe_cropped_long_center_tracks/'
        video_file = 'cafe_cropped_center_long.mp4'
    cam = cv2.VideoCapture(datasets + video_file)        
    return(out_dir, video_file)

base_dir = '/home/brad/pythonFiles/opencv_pedestrian_2/'
datasets = '/home/brad/pythonFiles/datasets/'
out_dir, video_file = load_dataset("oculus", base_dir)

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
track_len = 10
detect_interval = 25

write_every, write_frames, erode, dialate, min_l, min_dist = 50, 5, 5, 8, 100, 100
crop_frame = 1
run_name = 'time_difference'+ str(erode) + '_' + str(dialate) + '_' + str(min_l)



#%% track points in video

def anorm2(a):
    return (a*a).sum(-1)
def draw_str(dst, target, s):
    x, y = target
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)

def dist_between_points(p0, p1):
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

def track_is_idling(tr):
    diff = 0
    prev_tup = (-1, -1, -1)
    for tup in tr[-30:-1]:
        if prev_tup == (-1,-1,-1):
            prev_tup = tup
            continue
        diff = diff + dist_between_points(tup[0:2], prev_tup[0:2])
    return diff < 30
    
def dist_from_beginning(tr):
    p0 = tr[0][0:2]
    p1 = tr[-1][0:2]
    return dist_between_points(p0, p1)


def get_drawable_tracks(tracks, min_points, min_length):
    drawable_tracks = []
    for track in tracks:
        if len(track) >= min_points and dist_from_beginning(track) > min_length:
            drawable_tracks.append(track)
    return(drawable_tracks)

out_dir, cam = ld.load_dataset("oculus")

frame_idx = 0        
active_tracks = []
archive_tracks = []
fgbg = cv2.createBackgroundSubtractorMOG2()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

while (frame_idx <= write_frames * write_every ):
    print(frame_idx)
    ret, frame = cam.read()
    if crop_frame == 1:
        frame = frame[0:1080, 0:1200]
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    fgmask = fgbg.apply(frame)
    fgmask = erode_dialate(fgmask, erode, dialate)

    
    vis = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR).copy()
    
    #Caclulate Optical Flow and add to existing traks
    if len(active_tracks) > 0:
        img0, img1 = prev_gray, frame_gray
        p0 = np.float32([tr[-1][0:2] for tr in active_tracks]).reshape(-1, 1, 2) #most_recent_points
        p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params) #points_new_location
        p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params) #reconstructed points
        
        distance_old_to_reconstructed_old = abs(p0-p0r).reshape(-1, 2).max(-1) 
        limiter = 1
        new_points_found = distance_old_to_reconstructed_old < limiter
        next_iteration_active = [] 
        for tr, (x, y), new_point_found in zip(active_tracks, p1.reshape(-1, 2), new_points_found):
            if new_point_found:
                tr.append((x, y, frame_idx))
                next_iteration_active.append(tr)
            else:
                if track_is_idling(tr):
                    archive_tracks.append(tr)
                else:
                    next_iteration_active.append(tr) 
                    
        active_tracks = next_iteration_active
        
        
    #Add new features to track
    if ( np.sum(fgmask) > 0 and frame_idx % detect_interval == 0 ):
        p = cv2.goodFeaturesToTrack(frame_gray, mask = fgmask, **feature_params)
        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                active_tracks.append([(x, y, frame_idx)])
           
    #write if needed
    if (frame_idx % write_every == 0 and frame_idx != 0):        
        drawable_tracks = get_drawable_tracks(archive_tracks + active_tracks, min_l, min_dist)

        print("size of archive_tracks:" + str(len(archive_tracks)))
        print("size of active_tracks:" + str(len(active_tracks)))
        print("size of drawable_tracks: " + str(len(drawable_tracks)))

        draw_str(vis, (20, 20), 'active track count: %d' % len(active_tracks))
        draw_str(vis, (20, 40), 'archive track count: %d' % len(archive_tracks))
        draw_str(vis, (20, 60), 'drawable track count: %d' % len(drawable_tracks))
    
        if p is not None:
            for (x, y) in p.reshape(-1,2):
                cv2.circle(vis, (x, y), 3, (255, 0, 0), -1)
                cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
        
        if drawable_tracks is not None:
            track_num = 0
            for tr in drawable_tracks:
                track_color = tuple([256*x for x in cm.Paired(track_num % 255)[0:3]])
                cv2.polylines(vis, [np.int32([tup[0:2] for tup in tr])], False, track_color)
                draw_str(vis, np.int32((tr[-1][0],tr[-1][1]+track_num)),'track: %d' % track_num)
                draw_str(vis, np.int32((tr[0][0],tr[0][1]+track_num)),'track: %d' % track_num)
                track_num = track_num + 1
                
        
        cv2.imwrite(out_dir + run_name + '_drawable_' + str(frame_idx) + '.png', vis)
        cv2.imwrite(out_dir + run_name + "_frame_"+ str(limiter) + '_' + str(frame_idx) + '.png', frame)
        
    frame_idx += 1
    prev_gray = frame_gray


#%% Use Dynamic Time Warping to find track similarity
# http://www.cs.ucr.edu/~eamonn/vldb05.pdf

def simple_norm(x, y):
     return(math.pow(x*x + y*y, 0.5))
 
def gen_dtw_fig(dist, cost, acc, path, plot_num):
     plt.subplot(2,1,plot_num)
     plt.imshow(acc.T, origin='lower', cmap=cm.gray, interpolation='nearest')
     plt.plot(path[0], path[1], 'w')
     plt.xlim((-0.5, acc.shape[0]-0.5))
     plt.ylim((-0.5, acc.shape[1]-0.5))
     plt.annotate("minimum distance: " + str(dist), xy=(2, 1), xytext=(3, 4),
                 arrowprops=dict(facecolor='black', shrink=0.05))
def apply_dtw_1d(x0, x1):
     #diff = np.average(x1) - np.average(x0)
     #x1 = x1 - diff   
     dist, cost, acc, path = dtw(x0, x1, dist=lambda x0, x1: np.linalg.norm(x0 - x1, ord=1))
     
     return(dist, cost, acc, path)

    
 
def apply_dtw(track0, track1, gen_figs):
     x0 = track0[:,0].reshape(-1,1)
     x1 = track1[:,0].reshape(-1,1)
     
     distX, costX, accX, pathX = apply_dtw_1d(x0, x1)
     
     y0 = track0[:,1].reshape(-1,1)
     y1 = track1[:,1].reshape(-1,1)
     
     distY, costY, accY, pathY = apply_dtw_1d(y0, y1)
     
     
     if(gen_figs == 1):
         fig, ax = plt.subplots(nrows=2)
         gen_dtw_fig(distX, costX, accX, pathX, 1)
         gen_dtw_fig(distY, costY, accY, pathY, 2)

     total_cost = simple_norm(distX, distY)
     total_cost = simple_norm(total_cost, math.pow( abs( len(track0) - len(track1) ), 0.5) )
     
     return(total_cost)

track_arrays = []
for track in drawable_tracks:
    track_arrays.append(np.array(track))
 
num_tracks = len(track_arrays) 
costs = np.empty((num_tracks,num_tracks))
for i in np.linspace(0,num_tracks - 1, num_tracks,dtype='int32'):
    for j in np.linspace(0,num_tracks - 1, num_tracks,dtype='int32'):
        if i >= j:
            costs[i,j] = 0
            continue
        track0 = track_arrays[i]
        track1 = track_arrays[j]        
        cost = apply_dtw(track0, track1, 0)

        time_cost = abs(np.average(track0[:,2]) - np.average(track1[:,2]))
        cost = cost + time_cost
        
        print("i: " + str(i) + " j: " + str(j) + " cost: " + str(cost) + " time cost: " + str(time_cost))
        costs[i,j] = cost

plt.imshow(costs, cmap='hot', interpolation='nearest')
plt.show()     

#%% Perform hirearchial grouping on tracks
data = costs + np.transpose(costs)
data_squareform = squareform(data)

z = hac.linkage(data_squareform, 'ward')

fig = plt.figure(figsize=(10, 10))

plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
plt.grid(True)

hac.dendrogram(
    z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()


   
#%% Save results
save_dir = base_dir + "saves/"
np.save( save_dir + run_name,costs)
np.save( save_dir + run_name + "_tracks", track_arrays )


