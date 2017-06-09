#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 20 17:54:51 2017
"""
import glob
import cv2

#location of datasets folder checked out from:
#https://bwindsor22@bitbucket.org/bwindsor22/pedestrian_datasets.git 
datasets = '/home/brad/pythonFiles/datasets/'

#location of folder checked out from:
# https://bwindsor22@bitbucket.org/bwindsor22/opencv_pedestrian_2.git
base_dir = '/home/brad/pythonFiles/opencv_pedestrian_2/'
 

def load_dataset( dataset):
    if dataset == "hallway":
        out_dir = base_dir + 'outputFiles/hallway/'
        video_file = 'hallway_cropped_long.mp4'
    if dataset == "oculus":
        #frame = frame[220:720, 0:800] #somtimes use 
        out_dir = base_dir + 'outputFiles/Oculus/'
        video_file = 'pedestrian_datasets_oculus/Oculus.mp4'
    if dataset == "cafe":
        out_dir = base_dir + 'outputFiles/cafe_cropped_long_center_tracks/'
        video_file = 'cafe_cropped_center_long.mp4'
    cam = cv2.VideoCapture(datasets + video_file)        
    return(out_dir, cam)

def load_picture_dataset(dataset):
    if dataset == "mall":
        mall_path = datasets + '/mall/frames/'
        mall_img_files = glob.glob(mall_path + '*.jpg')
        mall_img_files.sort()
        return(mall_img_files)
    if dataset == "oculus_end":
        oculus_path = datasets + 'pedestrian_datasets_oculus/Oculus Edge/'
        oculus_img_files = glob.glob(oculus_path + '*.JPG')
        oculus_img_files.sort()
        return(oculus_img_files)
    return(null)