# -*- coding: utf-8 -*-
"""
Collection of analytical formulae.
"""
import numpy as np
import scipy.stats as stats
from scipy.stats import norm

#%%  Generating Lognormal Random Variables
def lognormal_rv(mu, sigma_squared):
    """
    returns X ~ logN(mu, sigma_squared), a scipy.stats random variable
    """
    return stats.lognorm(np.sqrt(sigma_squared), loc=0, scale=np.exp(mu))


def gbm_rv(S0, t, mu, sigma):
    """
    SDE: dS_t = mu S_t dt + sigma S_t dW_t
    Returns S_t, a scipy.stats random variable    
    """
    return lognormal_rv(np.log(S0) + (mu - 0.5*sigma**2)*t, sigma**2 * t)

#%% Black-Scholes Analytical Formulae
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

#%% CEV Analytical Formulae
def CEV_put(S0, K, T, r, sigma, alpha):
    """
    SDE: dS_t = r S_t dt + sigma S_t^alpha dW_t
    From NMF II, Lecture 2, Local Volatility
    """
    kappa = 2 * r / (sigma ** 2 * (1 - alpha)
        * (np.exp(2 * r * (1 - alpha) * T) - 1))
    x = kappa * S0 ** (2 * (1 - alpha)) * np.exp(2 * r * (1 - alpha) * T)
    y = kappa * K ** (2 * (1 - alpha))
    z = 2 + 1 / (1 - alpha)
    return -S0 * stats.ncx2.cdf(y, z, x) + (
            K * np.exp(-r * T) * (1 - stats.ncx2.cdf(x, z - 2, y)))


def CEV_call(S0, K, T, r, sigma, alpha):
    """
    SDE: dS_t = r S_t dt + sigma S_t^alpha dW_t
    From NMF II, Lecture 2, Local Volatility
    """
    kappa = 2 * r / (sigma ** 2 * (1 - alpha)
        * (np.exp(2 * r * (1 - alpha) * T) - 1))
    x = kappa * S0 ** (2 * (1 - alpha)) * np.exp(2 * r * (1 - alpha) * T)
    y = kappa * K ** (2 * (1 - alpha))
    z = 2 + 1 / (1 - alpha)
    return S0 * (1 - stats.ncx2.cdf(y, z, x)) - (
            K * np.exp(-r * T) * stats.ncx2.cdf(x, z - 2, y))

#%%
if (__name__ == '__main__'):
    print(f'Executing {__file__} as a standalone script.')