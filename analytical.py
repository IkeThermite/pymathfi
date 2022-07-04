# -*- coding: utf-8 -*-
"""
Collection of analytical formulae.
"""
import numpy as np
import scipy.stats as stats
from scipy.stats import norm

def lognormal_rv(mu, sigma_squared):
    """
    X ~ logN(mu, sigma_squared)
    """
    return stats.lognorm(np.sqrt(sigma_squared), loc=0, scale=np.exp(mu))


def gbm_rv(S0, t, mu, sigma):
    """
    SDE: dS_t = mu S_t dt + \sigma S_t dW_t
    Returns S_t, a scipy.stats random variable    
    """
    return lognormal_rv(np.ln(S0) + (mu - 0.5*sigma**2)*t, sigma**2 * t)


def d1(S0, K, T, r, sigma):
    return (np.log(S0/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))


def d2(S0, K, T, r, sigma):
    return d1(S0, K, T, r, sigma) - sigma*np.sqrt(T)


def black_scholes_put(S0, K, T, r, sigma):
    return (norm.cdf(-d2(S0, K, T, r, sigma))*K*np.exp(-r*T) -
            norm.cdf(-d1(S0, K, T, r, sigma))*S0)


def black_scholes_call(S0, K, T, r, sigma):
    return (norm.cdf(d1(S0, K, T, r, sigma))*S0 -
            norm.cdf(d2(S0, K, T, r, sigma))*K*np.exp(-r*T))


def black_scholes_put_delta(S0, K, T, r, sigma):
    return -norm.cdf(-d1(S0, K, T, r, sigma))


def black_scholes_call_delta(S0, K, T, r, sigma):
    return norm.cdf(d1(S0, K, T, r, sigma))


    
        