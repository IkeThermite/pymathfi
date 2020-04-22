import analytical
import finite_difference as fd
import numpy as np
# import finite_difference

# # Error is very largely determined by where the grid is centered around the moneyness
# # The more itm, the less error.
# # ? How many zeros are there in the initial condition.
# volatility = 0.4
# short_rate = 0.06
# time_steps = 40
# stock_steps = 80
# stock_min = 0
# stock_max = 150
# strike = 50
# maturity = 1

# model = fd.models.FDBlackScholesModel(volatility, short_rate, 
#         (stock_min, stock_max), stock_steps, time_steps)

# product = fd.products.FDEuropeanCallOption(strike, maturity, short_rate)

# solver = fd.solvers.FD1DThetaMethod(model, product)
# U, S = solver.solve()

# lower_bound = 90
# upper_bound = 110
# logical_index = np.logical_and(S >= lower_bound, S <= upper_bound)
# stock = S[logical_index]
# analytical_price = analytical.pricing.price_call_BS(stock, strike, maturity, 
#         short_rate, volatility)


# print(U[logical_index])
# print(analytical_price)