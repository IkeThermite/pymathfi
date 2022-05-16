import numpy as np
import scipy.stats as stats

def d1(S0, K, T, r, sigma):
    return (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def d2(S0, K, T, r, sigma):
    return d1(S0, K, T, r, sigma) - sigma * np.sqrt(T)


def price_put_BS(S0, K, T, r, sigma):
    return (stats.norm.cdf(-d2(S0, K, T, r, sigma)) * K * np.exp(-r * T) -
            stats.norm.cdf(-d1(S0, K, T, r, sigma)) * S0)


def price_call_BS(S0, K, T, r, sigma):
    return (stats.norm.cdf(d1(S0, K, T, r, sigma)) * S0 -
            stats.norm.cdf(d2(S0, K, T, r, sigma)) * K * np.exp(-r * T))

def delta_put_BS(S0, K, T, r, sigma):
    return -stats.norm.cdf(-d1(S0, K, T, r, sigma))

def delta_call_BS(S0, K, T, r, sigma):
    return stats.norm.cdf(d1(S0, K, T, r, sigma))

def price_put_CEV(S0, K, T, r, sigma, alpha):
    kappa = 2 * r / (sigma ** 2 * (1 - alpha)
        * (np.exp(2 * r * (1 - alpha) * T) - 1))
    x = kappa * S0 ** (2 * (1 - alpha)) * np.exp(2 * r * (1 - alpha) * T)
    y = kappa * K ** (2 * (1 - alpha))
    z = 2 + 1 / (1 - alpha)
    return -S0 * stats.ncx2.cdf(y, z, x) + (
            K * np.exp(-r * T) * (1 - stats.ncx2.cdf(x, z - 2, y)))

def price_call_CEV(S0, K, T, r, sigma, alpha):
    kappa = 2 * r / (sigma ** 2 * (1 - alpha)
        * (np.exp(2 * r * (1 - alpha) * T) - 1))
    x = kappa * S0 ** (2 * (1 - alpha)) * np.exp(2 * r * (1 - alpha) * T)
    y = kappa * K ** (2 * (1 - alpha))
    z = 2 + 1 / (1 - alpha)
    return S0 * (1 - stats.ncx2.cdf(y, z, x)) - (
            K * np.exp(-r * T) * stats.ncx2.cdf(x, z - 2, y))


if (__name__ == '__main__'):
    print(f'Executing {__file__} as a standalone script')
