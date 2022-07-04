# -*- coding: utf-8 -*-
"""
Created on Thu May  5 08:24:15 2022

@author: ralphr
"""
import numpy as np
import matplotlib.pyplot as plt
import models
import products
import montecarlo
import analytical

np.random.seed(seed=0)

# %% Generate Geometric Brownian Motions Paths
S0 = 100
r = 0.2
sigma = 0.2
n_samples = 50
T = 1
timeline = np.array([np.linspace(0, T, 100)])

model = models.GeometricBrownianMotion(S0, r, sigma)
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

# %% Monte Carlo Estimate for Vanilla Call and Put on Geometric Brownian Motion
S0 = 100
r = 0.2
sigma = 0.2
n_samples = 500000
K = 100
T = 1

model = models.GeometricBrownianMotion(S0, r, sigma)
product = products.CallOption(K, T)
payoff_sample_generator = montecarlo.PayoffSampleGenerator(model, product)
mc_call_price, call_std = montecarlo.montecarlo_estimate(payoff_sample_generator, n_samples)
call_price = analytical.black_scholes_call(S0, K, T, r, sigma)

product = products.PutOption(K, T)
payoff_sample_generator = montecarlo.PayoffSampleGenerator(model, product)
mc_put_price, put_std = montecarlo.montecarlo_estimate(payoff_sample_generator, n_samples)
put_price = analytical.black_scholes_put(S0, K, T, r, sigma)

print("######################################")
print("Monte Carlo Estimates")
print("Vanilla European Call")
print(f"Price: {call_price: .4f}, MC Price: {mc_call_price: .4f}, MC StDev: {call_std: .4f}")
print("Vanilla European Put")
print(f"Price: {put_price: .4f}, MC Price: {mc_put_price: .4f}, MC StDev: {put_std: .4f}")
print("######################################")

# %% Linear Regression to Model Conditional Expectations
S0 = 100
r = 0.2
sigma = 0.2
n_samples = 50
T = 1
timeline = np.array([[0, 0.5, 1]])

model = models.GeometricBrownianMotion(S0, r, sigma)
paths, discount_factors = model.simulate(n_samples, timeline)

fig, ax = plt.subplots()
ax.plot(paths[:,1], paths[:,2], 'o')
ax.set_xlabel("$S_t$")
ax.set_ylabel("$S_{t+1}$")
ax.set_title("Geometric Brownian Motion")

# fig, ax = plt.subplots()
# ax.plot(np.tile(timeline, (n_samples, 1)).T, discount_factors.T)
# ax.set_xlabel("Time")
# ax.set_ylabel("Discount Factors")
# ax.set_title("Constant Continously Compounded Interest Rate")


















