# -*- coding: utf-8 -*-
"""
Created on Tue May  3 14:20:10 2022

@author: ralphr
"""
import numpy as np

import analytical as formula
from pytest import approx

class TestAnalyticalFormulae:
    def test_black_scholes_put(self):
        S0 = 100
        T = 2
        r = 0.05
        sigma = 0.2        
        strikes = np.array([180, 100, 20])
        target_prices = [63.487654789203376, 6.610521528574566, 
            0.000000001412420]
        
        prices = formula.black_scholes_put(S0, strikes, T, r, sigma)
        assert prices == approx(target_prices)
    
    def test_black_scholes_call(self):
        S0 = 100
        T = 2
        r = 0.05
        sigma = 0.2        
        strikes = np.array([180, 100, 20])
        target_prices = [0.616919542730657, 16.126779724978633, 
            81.90325164069322]
        
        prices = formula.black_scholes_call(S0, strikes, T, r, sigma)
        assert prices == approx(target_prices)

# class TestModels:

# class TestMonteCarlo:
    
    
                