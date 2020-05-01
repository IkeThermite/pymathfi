import analytical
import finite_difference as fd
import numpy as np
import monte_carlo.monte_carlo as mc

# %% OO MC Implementation Test
np.random.seed(0)
sig = 0.2
r = 0.06
S0 = 100
K = 100
num_samples = 100000
T = 1

model = mc.MCGeometricBrownianMotion(sig, r, S0)
product = mc.MCEuropeanCallOption(K, T, r)

pricer = mc.MCPricer(product, model)
MC_price, MC_std = pricer.price(0, num_samples)

analytical_price = analytical.pricing.price_call_BS(S0, K, T, r, sig)
print(MC_price)
print(analytical_price)



# %% Finite Difference Tests
# # Error is very largely determined by where the grid is centered around the moneyness
# # The more itm, the less error.
# # ? How many zeros are there in the initial condition.

# BS_volatility = 0.4
# short_rate = 0.06
# time_steps = 40
# stock_steps = 80
# stock_min = 0
# stock_max = 150
# strike = 50
# maturity = 1

# exponent = 0.9
# CEV_volatility = BS_volatility * strike ** (exponent - 1)

# model = fd.models.FDCEVModel(exponent, CEV_volatility, short_rate, 
#         (stock_min, stock_max), stock_steps, time_steps)

# product = fd.products.FDEuropeanPutOption(strike, maturity, short_rate)

# solver = fd.solvers.FD1DThetaMethod(model, product)
# U, S = solver.solve()

# lower_bound = 45
# upper_bound = 55
# logical_index = np.logical_and(S >= lower_bound, S <= upper_bound)
# stock = S[logical_index]
# analytical_price = analytical.pricing.price_put_CEV(stock, strike, maturity, 
#         short_rate, CEV_volatility, exponent)

# print(U[logical_index])
# print(analytical_price)