# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 13:42:57 2020

@author: thera
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

filename = 'bs_calibration'
xlabel = 'Put Price'
ylabel = 'Implied Volatility'
title = 'Black-Scholes: Approx. Inverted Pricing Function'



npzfile = np.load(filename+ '.npz')

inputs = npzfile['inputs']
targets = npzfile['targets']
predictions = npzfile['predictions']
num_epochs = predictions.shape[1]
num_frames = num_epochs + 8

fig = plt.figure()
ax = plt.axes()
line1, = ax.plot([], [], lw=2)
plt.plot(inputs, targets, 'k-')

ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)   

def init():
    line1.set_data([], [])
    return

def animate(i):
    i = min(i, num_epochs-1)
    line1.set_data(inputs, predictions[:, i])
    ax.set_title(title + f': Epoch {i+1}')

anim = FuncAnimation(fig, animate, frames=num_frames, init_func=init, interval=150)
anim.save(filename + '.gif', writer='imagemagick')