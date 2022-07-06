# -*- coding: utf-8 -*-
"""
Test analytical formulae against results from other implementations.
"""

import numpy as np
from pytest import approx
import pymathfi.analytical as analytical


class TestAnalyticalFormulae:
    def test_black_scholes_put(self):
        S0 = 100
        T = 2
        r = 0.05
        sigma = 0.2        
        strikes = np.array([180, 100, 20])
        target_prices = [63.487654789203376, 6.610521528574566, 
            0.000000001412420]
        
        prices = analytical.black_scholes_put(S0, strikes, T, r, sigma)
        assert prices == approx(target_prices)
    
    def test_black_scholes_call(self):
        S0 = 100
        T = 2
        r = 0.05
        sigma = 0.2        
        strikes = np.array([180, 100, 20])
        target_prices = [0.616919542730657, 16.126779724978633, 
            81.90325164069322]
        
        prices = analytical.black_scholes_call(S0, strikes, T, r, sigma)
        assert prices == approx(target_prices)
    
    def test_CEV_put(self):
        S0 = 100
        T = 2
        r = 0.05
        strikes = np.array([60, 100, 140])
        alphas = np.array([0.2, 0.5, 0.9])
        sig_CEV = np.array([8, 2, 0.4])
        target_prices = [
            [0.395102341625255, 6.682237364858608, 29.220025067034527], 
            [0.254691434287102, 6.619770490971426, 29.553505240830873], 
            [0.526050894195097, 9.252524142765573, 32.526199550777335]]
        for i in range(len(sig_CEV)):
            for j in range(len(alphas)):
                price = analytical.CEV_put(S0, strikes[j], T, r, sig_CEV[i],
                                           alphas[i])
                assert price == approx(target_prices[i][j])
    
    def test_CEV_call(self):
        S0 = 100
        T = 2
        r = 0.05
        strikes = np.array([60, 100, 140])
        alphas = np.array([0.2, 0.5, 0.9])
        sig_CEV = np.array([8, 2, 0.4])
        target_prices = [
            [46.104857259467686, 16.198495561262661, 2.542786542000208], 
            [45.964446352129528, 16.136028687375472, 2.876266715796540], 
            [46.235805812037526, 18.768782339169633, 5.848961025742998]]
        for i in range(len(sig_CEV)):
            for j in range(len(alphas)):
                price = analytical.CEV_call(S0, strikes[j], T, r, sig_CEV[i],
                                           alphas[i])
                assert price == approx(target_prices[i][j])