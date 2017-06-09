#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 17:47:00 2017

@author: brad
"""
import matplotlib.pyplot as plt
import numpy.random as random
import copy

class PosSensor1(object):
    def __init__(self, pos = [0,0], vel = (0,0), noise_scale = 1.):
        self.vel = vel
        self.noise_scale = noise_scale
        self.pos = copy.deepcopy(pos)
        self.times_read = 0
    def read(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.times_read += 1
        if self.times_read > 30:
            self.vel= (- 5, self.vel[1])
        return [self.pos[0] + random.randn() * self.noise_scale,
                self.pos[1] + random.randn() * self.noise_scale]
"""
pos = [4,3]
s = PosSensor1(pos, (2,1), 1)
for i in range (50):
    pos = s.read()
    plt.scatter(pos[0], pos[1])
plt.show()
"""

from filterpy.kalman import KalmanFilter
import numpy as np
f1 = KalmanFilter(dim_x=4, dim_z=2)
dt = 1.# time step

f1.F = np.array ([[1, dt, 0,  0],
                [0,  1, 0,  0],
                [0,  0, 1, dt],
                [0,  0, 0,  1]])
f1.u = 0 # no info about rotors turning
f1.H = np.array ([[1, 0, 0, 0], #measurement to state.
                  [0, 0, 1, 0]])
f1.R = np.array([[5,0], # initial variance in measurments
                 [0, 5]])
f1.Q = np.eye(4) * 0.001 # process noise
f1.x = np.array([[0,0,0,0]]).T # initial position
f1.P = np.eye(4) * 500 #covariance matrix

# initialize storage and other variables for the run
count = 60
xs, ys = [],[]
pxs, pys = [],[]
s = PosSensor1 ([0,0], (2,1), 3.)
for i in range(count):
    pos = s.read()
    z = np.array([[x],[y]])
    f1.predict ()
    f1.update (z)
    xs.append (f1.x[0,0])
    ys.append(f1.x[2,0])
    pxs.append(pos[0])
    pys.append(pos[1])
    
p1, = plt.plot (xs, ys, 'r--')
p2, = plt.plot (pxs, pys)
