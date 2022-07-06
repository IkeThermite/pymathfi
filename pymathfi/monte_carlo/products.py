# -*- coding: utf-8 -*-
"""
Products for use with Monte Carlo simulation.
"""
import numpy as np


class MCProduct():
    def __init__(self, maturity):
        self.maturity = maturity
    
    def payoff(self, underlying):
        raise NotImplementedError()

        
class MCIdentity(MCProduct):
    def payoff(self, underlying):
        return underlying


class MCEuropeanCallOption(MCProduct):
    """
    Vanilla European Call option.
    """
    def __init__(self, strike, maturity):
        super().__init__(maturity)
        self.strike = strike
    
    def payoff(self, underlying):
        # Reminder: np.maximum is element-wise
        return np.maximum(underlying - self.strike, 0)
    
    
class MCEuropeanPutOption(MCProduct):
    """
    Vanilla European Put option.
    """
    def __init__(self, strike, maturity):
        super().__init__(maturity)
        self.strike = strike
        
    def payoff(self, underlying):
        # Reminder: np.maximum is element-wise
        return np.maximum(self.strike - underlying, 0)