# -*- coding: utf-8 -*-
"""
Created on Thu May  5 08:08:38 2022

@author: ralphr
"""
import numpy as np

class Product():
    def __init__(self, maturity):
        self.maturity = maturity
    
    def payoff(self, underlying):
        raise NotImplementedError()

        
class Identity(Product):
    def payoff(self, underlying):
        return underlying


class CallOption(Product):
    """
    Vanilla European Call option.
    """
    def __init__(self, strike, maturity):
        super().__init__(maturity)
        self.strike = strike
    
    def payoff(self, underlying):
        # Reminder: np.maximum is element-wise
        return np.maximum(underlying - self.strike, 0)
    
class PutOption(Product):
    """
    Vanilla European Put option.
    """
    def __init__(self, strike, maturity):
        super().__init__(maturity)
        self.strike = strike
        
    def payoff(self, underlying):
        # Reminder: np.maximum is element-wise
        return np.maximum(self.strike - underlying, 0)
    
