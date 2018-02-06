#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 12:11:37 2017
Helper class for loading datasets, images and otherwise
"""
import glob
import cv2


datasets = '/datasets/'

base_dir = '/opencv_pedestrian_2/'




video_datasets = dict(hallway = dict(
                                    out_dir = base_dir + 'outputFiles/hallway/',
                                    video_file = datasets + 'hallway_cropped_long.mp4'
                                     ),
                      cafe = dict(
                                  out_dir = base_dir + 'outputFiles/cafe_cropped_long_center_tracks/',
                                  video_file = datasets + 'cafe_cropped_center_long.mp4'
                                  )
                      )

picture_datasets = dict(mall = dict(
                                    path = datasets + '/mall/frames/',
                                    extension = '*.jpg'
                                    ),
                        )
   
class ImageDataset:
    def __init__(self, name, span_range):
        dataset = picture_datasets[name]
        self.img_files = glob.glob(dataset['path'] + dataset['extension'])
        self.img_files.sort()
        
        self.start_frame = 0
        self.end_frame = len(self.img_files) - 1
    
    def get_next_frame(self, frame_idx):        
        if frame_idx <= self.end_frame:
            return( True, cv2.imread(self.img_files[frame_idx]))
        return( False, None)

        
class VideoDataset:
    def __init__(self, name, frame_range):
        self.dataset = video_datasets[name]
        self.cam = cv2.VideoCapture(self.dataset['video_file'])
        
        if 'frame_ranges' in self.dataset.keys() \
        and frame_range in self.dataset['frame_ranges'].keys():
            self.start_frame, self.end_frame = self.dataset['frame_ranges'][frame_range]
        else:
            self.start_frame = 0
            self.end_frame = 500
    
        self.cam.set(1, self.start_frame)
    
    def get_next_frame(self, frame_idx):
        if frame_idx <= self.end_frame:
            return( self.cam.read() )
        return(False, None)

def get_dataset(name, frame_range='all'):
    if name in picture_datasets.keys():
        return( ImageDataset(name, frame_range) )

    elif name in video_datasets.keys():
        return( VideoDataset(name, frame_range) )
    
    return "Not Found"
        
