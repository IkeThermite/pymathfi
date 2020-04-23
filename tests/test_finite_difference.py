import unittest
import numpy as np
import finite_difference as fd
import analytical

class Test1DThetaMethod(unittest.TestCase):
    def test_price_call_BS(self):

        volatility = 0.4
        short_rate = 0.06
        time_steps = 40
        stock_steps = 80
        stock_min = 0
        stock_max = 150
        strike = 50
        maturity = 1
        
        model = fd.models.FDBlackScholesModel(volatility, short_rate, 
        (stock_min, stock_max), stock_steps, time_steps)

        product = fd.products.FDEuropeanCallOption(strike, maturity, short_rate)

        solver = fd.solvers.FD1DThetaMethod(model, product)
        U, S = solver.solve()

        lower_bound = 45
        upper_bound = 55
        logical_index = np.logical_and(S >= lower_bound, S <= upper_bound)
        stock = S[logical_index]
        analytical_price = analytical.pricing.price_call_BS(stock, strike, maturity, 
                short_rate, volatility)
        fd_price = U[logical_index]

        for i in range(len(stock)):
            self.assertAlmostEqual(fd_price[i], analytical_price[i],
                    places=2)

    def test_price_put_BS(self):

        volatility = 0.4
        short_rate = 0.06
        time_steps = 40
        stock_steps = 80
        stock_min = 0
        stock_max = 150
        strike = 50
        maturity = 1
        
        model = fd.models.FDBlackScholesModel(volatility, short_rate, 
        (stock_min, stock_max), stock_steps, time_steps)

        product = fd.products.FDEuropeanPutOption(strike, maturity, short_rate)

        solver = fd.solvers.FD1DThetaMethod(model, product)
        U, S = solver.solve()

        lower_bound = 45
        upper_bound = 55
        logical_index = np.logical_and(S >= lower_bound, S <= upper_bound)
        stock = S[logical_index]
        analytical_price = analytical.pricing.price_put_BS(stock, strike, maturity, 
                short_rate, volatility)
        fd_price = U[logical_index]

        for i in range(len(stock)):
            self.assertAlmostEqual(fd_price[i], analytical_price[i],
                    places=2)
