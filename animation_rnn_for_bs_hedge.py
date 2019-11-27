# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 09:55:57 2019

@author: thera
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

filename = 'rnn_bs_hedge'
npzfile = np.load(filename+ '.npz')

grid_S0 = npzfile['grid_S0']
grid_S1 = npzfile['grid_S1']
BS_delta_0 = npzfile['BS_delta_0']
BS_delta_1 = npzfile['BS_delta_1']
RNN_delta_0 = npzfile['RNN_delta_0']
RNN_delta_1 = npzfile['RNN_delta_1']
num_epochs = RNN_delta_0.shape[1]

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    return
def animate(i):
    i = min(i, num_epochs-1)
    ax1.set_title('RNN Delta at $t_1$: Epoch %d' % (i + 1))
    line1.set_data(grid_S0, RNN_delta_0[:, i])
    
    ax2.set_title('RNN Delta at $t_2$: Epoch %d' % (i + 1))
    line2.set_data(grid_S1, RNN_delta_1[:,i])
    return


num_frames = num_epochs + 5

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

ax1.set_xlim([np.min(grid_S0), np.max(grid_S0)])
ax1.set_ylim([0, 1])
ax1.set_xlabel('$S_0$')
ax1.set_ylabel('Delta')     
line1, = ax1.plot([], [], lw=2)
ax1.plot(grid_S0, BS_delta_0, 'k-')

ax2.set_xlim([np.min(grid_S0), np.max(grid_S0)])
ax2.set_ylim([0, 1])
ax2.set_xlabel('$S_1$')
ax2.set_ylabel('Delta')     
line2, = ax2.plot([], [], lw=2)
ax2.plot(grid_S1, BS_delta_1, 'k-')


anim = FuncAnimation(fig, animate, frames=num_frames, init_func=init, interval=150)
anim.save('rnn_bs_hedge.gif', writer='imagemagick')