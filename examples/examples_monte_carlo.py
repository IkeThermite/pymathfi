# -*- coding: utf-8 -*-
"""
Demonstrates the use of the Monte Carlo code.
"""
import numpy as np
import matplotlib.pyplot as plt
import pymathfi.monte_carlo as mc
import pymathfi.analytical as analytical

def run():
    print(f"{__name__}: Setting random seed to 0.")
    np.random.seed(seed=0)       
    example_gbm_paths()
    example_black_scholes_call()
    example_black_scholes_put()
    return


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
    
    gbm_rvs = analytical.gbm_rv(S0, timeline[0, 1:], r, sigma)
    mean_path = np.hstack((S0, gbm_rvs.mean()))
    
    fig, ax = plt.subplots()
    ax.plot(np.tile(timeline, (n_samples, 1)).T, paths.T)
    ax.plot(timeline[0, :], mean_path, 'k')
    ax.set_xlabel("Time")
    ax.set_ylabel("Underlying")
    ax.set_title("Geometric Brownian Motion")
    
    fig, ax = plt.subplots()
    ax.plot(np.tile(timeline, (n_samples, 1)).T, discount_factors.T)
    ax.set_xlabel("Time")
    ax.set_ylabel("Discount Factors")
    ax.set_title("Constant Continously Compounded Interest Rate")
    return


def example_black_scholes_call():
    print("EXAMPLE: Black-Scholes European Call using monte_carlo.solvers.CrudeMonteCarloPricer")
    S0 = 100
    r = 0.2
    sigma = 0.2
    n_samples = 500000
    K = 100
    T = 1
    
    model = mc.models.MCBlackScholesModel(S0, r, sigma)
    product = mc.products.MCEuropeanCallOption(K, T)
    pricer = mc.solvers.CrudeMonteCarloPricer(product, model)
    
    crude_mc_price, crude_mc_stdev = pricer.price(n_samples)
    analytical_price = analytical.black_scholes_call(S0, K, T, r, sigma)
   
    print("Black-Scholes European Call:")
    print(f"Price: {analytical_price: .4f}, MC Price: {crude_mc_price: .4f}, MC StDev: {crude_mc_stdev: .4f}")
    

def example_black_scholes_put():
    print("EXAMPLE: Black-Scholes European Put using monte_carlo.solvers.CrudeMonteCarloPricer")
    S0 = 100
    r = 0.2
    sigma = 0.2
    n_samples = 500000
    K = 100
    T = 1
    
    model = mc.models.MCBlackScholesModel(S0, r, sigma)
    product = mc.products.MCEuropeanPutOption(K, T)
    pricer = mc.solvers.CrudeMonteCarloPricer(product, model)
    
    crude_mc_price, crude_mc_stdev = pricer.price(n_samples)
    analytical_price = analytical.black_scholes_put(S0, K, T, r, sigma)
   
    print("Black-Scholes European put:")
    print(f"Price: {analytical_price: .4f}, MC Price: {crude_mc_price: .4f}, MC StDev: {crude_mc_stdev: .4f}")
    
    