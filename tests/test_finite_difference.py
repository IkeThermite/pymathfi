import unittest
import numpy as np
import finite_difference as fd
import analytical

class Test1DThetaMethod(unittest.TestCase):
    def test_vanilla_BS(self):
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

        call_option = fd.products.FDEuropeanCallOption(strike, maturity, 
                short_rate)
        put_option = fd.products.FDEuropeanPutOption(strike, maturity, 
                short_rate)
        
        # TODO: The S that comes back is independent of the product
        call_solver = fd.solvers.FD1DThetaMethod(model, call_option)
        U_call, S = call_solver.solve()
        put_solver = fd.solvers.FD1DThetaMethod(model, put_option)
        U_put, S = put_solver.solve()

        lower_bound = 45
        upper_bound = 55
        logical_index = np.logical_and(S >= lower_bound, S <= upper_bound)
        stock = S[logical_index]
        analytical_call_price = analytical.pricing.price_call_BS(stock, strike, maturity, 
                short_rate, volatility)
        analytical_put_price = analytical.pricing.price_put_BS(stock, strike, maturity, 
                short_rate, volatility)
        fd_call_price = U_call[logical_index]
        fd_put_price = U_put[logical_index]

        for i in range(len(stock)):
            self.assertAlmostEqual(fd_call_price[i], analytical_call_price[i],
                    places=2)
            self.assertAlmostEqual(fd_put_price[i], analytical_put_price[i],
                    places=2)

    def test_vanilla_CEV(self):
        BS_volatility = 0.4
        short_rate = 0.06
        time_steps = 40
        stock_steps = 80
        stock_min = 0
        stock_max = 150
        strike = 50
        maturity = 1
        exponent = 0.9
        CEV_volatility = BS_volatility * strike ** (exponent - 1)
        
        model = fd.models.FDCEVModel(exponent, CEV_volatility, short_rate, 
        (stock_min, stock_max), stock_steps, time_steps)

        call_option = fd.products.FDEuropeanCallOption(strike, maturity, short_rate)
        put_option = fd.products.FDEuropeanPutOption(strike, maturity, 
                short_rate)

        # TODO: The S that comes back is independent of the product
        call_solver = fd.solvers.FD1DThetaMethod(model, call_option)
        U_call, S = call_solver.solve()
        put_solver = fd.solvers.FD1DThetaMethod(model, put_option)
        U_put, S = put_solver.solve()

        lower_bound = 45
        upper_bound = 55
        logical_index = np.logical_and(S >= lower_bound, S <= upper_bound)
        stock = S[logical_index]
        analytical_call_price = analytical.pricing.price_call_CEV(stock, strike, maturity, 
                short_rate, CEV_volatility, exponent)
        analytical_put_price = analytical.pricing.price_put_CEV(stock, strike, maturity, 
                short_rate, CEV_volatility, exponent)
        fd_call_price = U_call[logical_index]
        fd_put_price = U_put[logical_index]

        for i in range(len(stock)):
            self.assertAlmostEqual(fd_call_price[i], analytical_call_price[i],
                    places=2)
            self.assertAlmostEqual(fd_put_price[i], analytical_put_price[i],
                    places=2)

    