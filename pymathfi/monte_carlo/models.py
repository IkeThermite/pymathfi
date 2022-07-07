# -*- coding: utf-8 -*-
"""
Models for use with Monte Carlo simulation.
"""
import numpy as np
from scipy.stats import norm


class MCModel():
    def simulate(self, num_samples, timeline):
        raise NotImplementedError
    
    
class MCBlackScholesModel(MCModel):
    """
    SDE: dS_t = rS_t dt + \sigma S_t dW_t
    """
    def __init__(self, S0, r, sigma):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
    
    def simulate(self, n_samples, timeline):
        """
        timeline = np.array([[t0, t1, t2, ..., T]])
        Assumes that S0 corresponds to t = 0 <= t_0
        Returns:
            paths, a n_samples x len(timeline) matrix
            discount_factors, a n_samples x len(timeline) matrix
        """
        timeline_starts_at_zero = np.abs(timeline[0, 0] - 0) <= 1e-8
        if timeline_starts_at_zero:
            dt = np.diff(timeline)
        else:
            dt = np.diff(timeline, prepend=0)
        
        dt = np.tile(dt, (n_samples, 1))
        Z = norm.rvs(size=(n_samples, dt.shape[1]))  
        Wt = np.sqrt(dt)*Z
        paths = self.S0*np.exp(np.cumsum((self.r - 0.5*self.sigma**2)*dt +
                                 self.sigma*Wt, axis=1))
        paths = np.hstack((np.ones((n_samples, 1))*self.S0, paths))
        discount_factors = np.exp(-self.r*np.tile(timeline, (n_samples, 1)))
        
        if not timeline_starts_at_zero:
            paths = paths[:, 1:]
                
        return paths, discount_factors

