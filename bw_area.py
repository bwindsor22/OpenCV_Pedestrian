# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#Misc 
import os

#Matlab imports
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import cv2

#Constants for data locations
DATA = "data"
BKGND = "bkgnd"
IMG_BASE = "seq_00{0}0.jpg"

def get_contours(img_num, area=0):
    """
    Returns contours from a num with filtered area
    """
    bkgnd_path = os.path.join(DATA, BKGND, IMG_BASE.format(img_num))
 
    bimg = cv2.imread(bkgnd_path)

    gray_bimg = cv2.cvtColor(bimg, cv2.COLOR_BGR2GRAY)
    _, contours, hierarchy = cv2.findContours(gray_bimg, 1, 2)
    return contours

def draw_ellipses(img_num, area=0):
    """
    Draws ellipses on contours. 
    Uses area a lower bound for what to pass through.
    """
    img_path = os.path.join(DATA, IMG_BASE.format(img_num))
    img = cv2.imread(img_path)
    contours = get_contours(img_num, area)
    for contour in contours:
        try:
            ellipse = cv2.fitEllipse(contour)
            cv2.ellipse(img, ellipse, (255, 255, 255), 2)
        except Exception:
            continue
    return img

def show_image(img_num, area=0):
    """
    Circles contours and displays image.
    """
    img = draw_ellipses(img_num, area)
    plt.imshow(img, cmap="gray")
    
plt.figure()
contours = get_contours(200, 0)
areas = [int(cv2.contourArea(contour)) for contour in contours]
hist, bins = np.histogram(areas, bins = 1000)
hist = hist[3:]
bins = bins[3:]
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)