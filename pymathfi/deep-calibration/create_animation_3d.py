# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 13:42:57 2020

@author: thera
"""
# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits import mplot3d
from matplotlib import cm

# %% Setup
filename = 'cev_calibration'
xlabel = r'$\sigma$'
ylabel = r'$\alpha$'
zlabel = 'Put Option Price'
title = 'CEV: Calibrated Pricing Function'

# %% Create Animation
npzfile = np.load(filename+ '.npz')

inputs = npzfile['inputs']
targets = npzfile['targets']
predictions = npzfile['predictions']
num_epochs = predictions.shape[1]
num_frames = num_epochs + 8

X = np.reshape(targets[:, 0], (15, 15))
Y = np.reshape(targets[:, 1], (15, 15))
Z = np.reshape(predictions[:, 0], (15, 15))

#Z = abs(np.reshape(predictions[:, 0], (15, 15)) - np.reshape(inputs, (15, 15)))
#fig = plt.figure()
#Z = np.reshape(inputs, (15, 15))

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)
ax.set_zlabel(zlabel)   
plot_handle = [ax.plot_surface(X, Y, Z, cmap=cm.plasma)]

def animate(i):
    ax.set_title(title + f': Epoch {min(i+1, num_epochs)}')
    i = min(i, num_epochs - 2)
    plot_handle[0].remove()
#    Z = abs(np.reshape(predictions[:, i + 1], (15, 15)) - np.reshape(inputs, (15, 15)))
    Z = np.reshape(predictions[:, i + 1], (15, 15))
    plot_handle[0] = ax.plot_surface(X, Y, Z, cmap=cm.plasma)
    

anim = FuncAnimation(fig, animate, frames=num_frames, interval=150)
#anim.save(filename + '.gif', writer='imagemagick')





















