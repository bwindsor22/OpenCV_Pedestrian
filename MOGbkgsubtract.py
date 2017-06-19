#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 14:29:58 2017

Basic background subtractor
Issue is python cv2 has limited parameters (no alpha) so limited ability to track
people standing still
"""

import cv2

class mog_bkg_subtractor:
    def __init__(self, erode=5, dialate=8, kernel_size=3):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size,kernel_size))
        self.erode = erode
        self.dialate = dialate
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=1000)

    def process_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = self.fgbg.apply(frame)
        frame = self.erode_dialate(frame) 
        return( frame )       
   
    def erode_dialate(self, frame):
        frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, self.kernel) 
        frame = cv2.erode(frame, self.kernel,iterations = self.erode)
        frame = cv2.dilate(frame, self.kernel, iterations = self.dialate)
        return(frame)
