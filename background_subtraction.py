#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 18:54:58 2017

@author: brad
"""
import cv2
import numpy as np
import glob
out_dir = '/home/brad/pythonFiles/datasets/mall_background/'
mall_path ="/home/brad/pythonFiles/datasets/mall/frames/"
mall_img_files = glob.glob(mall_path + '*.jpg')
mall_img_files.sort()

def erode_dialate(fgmask, erode, dialate):
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel) 
    fgmask = cv2.erode(fgmask, kernel,iterations = erode)
    fgmask = cv2.dilate(fgmask, kernel, iterations = dialate)
    return(fgmask)

fgbg = cv2.createBackgroundSubtractorMOG2()

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

i = 0
for image_file in mall_img_files:
    i = i + 1
    if i > 200:
        break
    image_file_split = image_file.split('/')
    image_name = image_file_split[len(image_file_split) - 1 ]   

    frame = cv2.imread(image_file)
    
    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)    
    
    fgmask = erode_dialate(fgmask, 2, 2)
    cv2.imwrite(out_dir + image_name , fgmask)
    
print('done')
    
