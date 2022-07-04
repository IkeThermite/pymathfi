# -*- coding: utf-8 -*-
"""
Created on Tue May  3 12:58:42 2022

@author: ralphr
"""
import numpy as np
from scipy.stats import norm

def GBM_paths(n_samples, S0, r, sigma, timeline):
    '''
    dS_t = rS_t dt + \sigma S_t dW_t
    timeline = [[0, t1, t2, ..., T]]
    '''
    dt = np.tile(np.diff(timeline), (n_samples, 1))
    Z = norm.rvs(size=(n_samples, dt.shape[1]))  
    Wt = np.sqrt(dt)*Z
    St = S0*np.exp(np.cumsum((r - 0.5*sigma**2)*dt + sigma*Wt, axis=1))
    St = np.hstack((S0*np.ones((n_samples, 1)), St))
    return St

S0 = 100
r = 0.3
sigma = 0.2
n_samples = 10
timeline = [[0, 0.5, 1]]

St = GBM_paths(n_samples, S0, r, sigma, timeline)
discount_factors = np.exp(-r*np.array(timeline))

# timeline = np.tile(timeline, (n_samples, 1))
  
    