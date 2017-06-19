#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 12:31:31 2017

@author: brad
"""

import numpy as np
import matplotlib.pyplot as plt
import load_pedestrian as ld



#%%

from frame_diff_bkgsubtract import frame_diff_bkgsubtract
base_dir = '/home/brad/pythonFiles/opencv_pedestrian_2/'


Dataset = ld.get_dataset('oculus','person_horizontal')
ret, frame = Dataset.get_next_frame(0)
#Bkg = BackGroundSubtractor(0.001,30,firstFrame=frame)
noise_bkg = np.load(base_dir + 'noisebkg.npy')
Bkg = frame_diff_bkgsubtract(0.001,30,bkg=noise_bkg)

frame_idx = 0
while(True):
    frame_idx += 1
    print(frame_idx)
	# Read a frame from the camera
    ret, frame = Dataset.get_next_frame(frame_idx)
    if not ret:
        print("end of reel")
        break    

		# Show the filtered image
    plt.figure()
    plt.title('input')
    plt.imshow(frame)
    plt.show()
    
    
    
    mask = Bkg.process_frame(frame)
    
    plt.figure()
    plt.title('mask')
    plt.imshow(mask, cmap='gray')
    plt.show()


