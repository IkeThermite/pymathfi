# -*- coding: utf-8 -*-
"""
Created on Tue May  3 13:29:17 2022

@author: ralphr
"""
import numpy as np

class PayoffSampleGenerator():
    """
    Input: model, product
    Output: samples of product.payoff at product.maturity using the model
    Currently only handles payoffs that occur at a single date, maturity
    """
    def __init__(self, model, product, discounted_flag=True):
        self.model = model
        self.product = product
        self.discounted_flag = discounted_flag
        self.timeline = [[0, product.maturity]]
    
    def generate(self, n_samples):
        paths, discount_factors = self.model.simulate(n_samples, self.timeline)
        if (self.discounted_flag):
            payoffs = discount_factors * self.product.payoff(paths)
        else:
            payoffs = self.product.payoff(paths)
        return payoffs[:, -1]


def montecarlo_estimate(sample_generator, n_samples, target_accuracy=None, 
                        confidence_level=0.05, max_batch_size=1e6):
    """
    Pr(True Value - 3*mc_std < mc_est < True Value + 3*mc_std) = 99.73 %
    """
    samples = sample_generator.generate(n_samples)
    
    sample_mean = np.mean(samples)
    sample_variance = np.var(samples, ddof=1)
    
    mc_std = np.sqrt(sample_variance)/np.sqrt(n_samples)
    mc_est = sample_mean
    return mc_est, mc_std

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        