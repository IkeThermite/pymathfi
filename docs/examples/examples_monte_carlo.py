# -*- coding: utf-8 -*-
"""
Demonstrates the use of the Monte Carlo code.
"""
import numpy as np
import matplotlib.pyplot as plt
import pymathfi.monte_carlo as mc


def run():
    example_gbm_paths()
    return

# %% Generate Geometric Brownian Motion Paths
def example_gbm_paths():
    print("EXAMPLE: GBM Paths using monte_carlo.models.MCBlackScholesModel")
    S0 = 100
    r = 0.2
    sigma = 0.2
    n_samples = 50
    T = 1
    timeline = np.array([np.linspace(0, T, 100)])
    
    model = mc.models.MCBlackScholesModel(S0, r, sigma)
    paths, discount_factors = model.simulate(n_samples, timeline)
    
    fig, ax = plt.subplots()
    ax.plot(np.tile(timeline, (n_samples, 1)).T, paths.T)
    ax.set_xlabel("Time")
    ax.set_ylabel("Underlying")
    ax.set_title("Geometric Brownian Motion")
    
    fig, ax = plt.subplots()
    ax.plot(np.tile(timeline, (n_samples, 1)).T, discount_factors.T)
    ax.set_xlabel("Time")
    ax.set_ylabel("Discount Factors")
    ax.set_title("Constant Continously Compounded Interest Rate")
    return