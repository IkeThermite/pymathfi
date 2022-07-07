# -*- coding: utf-8 -*-
"""
Monte Carlo estimators.
"""
import numpy as np

class CrudeMonteCarloPricer():
    def __init__(self, product, model):
        self.product = product
        self.model = model
    
    def price(self, n_samples, target_accuracy=None):
        """
        Pr(True Value - 3*mc_stdev < mc_estimate < True Value + 3*mc_stdev) = 99.73 %
        """
        underlying, discount_factors = self.model.simulate(
            n_samples, self.product.timeline)
        
        samples = (discount_factors*self.product.payoff(underlying))
        
        sample_mean = np.mean(samples)
        sample_variance = np.var(samples, ddof=1)
        
        mc_estimate = sample_mean
        mc_stdev = np.sqrt(sample_variance)/np.sqrt(n_samples)
        
        return mc_estimate, mc_stdev