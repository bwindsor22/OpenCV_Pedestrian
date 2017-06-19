#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 14:34:24 2017

modified from TheSalarKhan github repo
allows for longer term memory but much shorter vs MOGbkgsubtract.py
"""

import cv2
import numpy as np

class frame_diff_bkgsubtract:
	# When constructing background subtractor, we
	# take in two arguments:
	# 1) alpha: The background learning factor, its value should
	# be between 0 and 1. The higher the value, the more quickly
	# your program learns the changes in the background. Therefore, 
	# for a static background use a lower value, like 0.001. But if 
	# your background has moving trees and stuff, use a higher value,
	# maybe start with 0.01.
    def __init__(self, long_term_alpha, thresh, firstFrame=None, bkg=None, erode=3, dialate=6, kernel_size=3):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size,kernel_size))
        self.erode = erode
        self.dialate = dialate
        self.thresh = thresh
        self.frame_count = 0

        
        self.long_term_alpha = long_term_alpha
        self.alpha = long_term_alpha
        
        if bkg is None:
            self.create_new_bkg = True
            self.backGroundModel = self.denoise(firstFrame)
        else:
            self.create_new_bkg = False
            self.backGroundModel = bkg
            

    def process_frame(self, frame):
        self.frame_count += 1
        foreground = self.getForeground(frame)
        ret, mask = cv2.threshold(foreground, self.thresh, 255, cv2.THRESH_BINARY)
        
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        mask = self.erode_dialate(mask)
        return mask
        
    def getForeground(self,frame):
        # use a higher alpha to adaptively learn, as needed
        if self.create_new_bkg:
            if self.frame_count > 1000:
                self.create_new_bkg = False
                self.alpha = self.long_term_alpha
            elif self.frame_count == 200:
                self.alpha = 0.005
            elif self.frame_count == 100:
                self.alpha = 0.01
            else:
                self.alpha = 0.05

		# apply the background averaging formula:
		# NEW_BACKGROUND = CURRENT_FRAME * ALPHA + OLD_BACKGROUND * (1 - APLHA)        
        self.backGroundModel =  frame * self.alpha + self.backGroundModel * (1 - self.alpha)

		# after the previous operation, the dtype of
		# self.backGroundModel will be changed to a float type
		# therefore we do not pass it to cv2.absdiff directly,
		# instead we acquire a copy of it in the uint8 dtype
		# and pass that to absdiff.
        
        return cv2.absdiff(self.backGroundModel.astype(np.uint8),frame)

    def erode_dialate(self, frame):
        frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, self.kernel) 
        frame = cv2.erode(frame, self.kernel,iterations = self.erode)
        frame = cv2.dilate(frame, self.kernel, iterations = self.dialate)
        return(frame)

    def denoise(self, frame):
        frame = cv2.medianBlur(frame,5)
        frame = cv2.GaussianBlur(frame,(5,5),0)
    
        return frame
