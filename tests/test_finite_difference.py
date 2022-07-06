# -*- coding: utf-8 -*-
"""
Test finite difference methods against analytical formulae.
"""
import numpy as np
from pytest import approx
import pymathfi.analytical as analytical
import pymathfi.finite_difference as fd


class Test1DThetaMethod:
    def test_black_scholes_put(self):
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
        
        put_option = fd.products.FDEuropeanPutOption(strike, maturity, 
                short_rate)
        
        # TODO: The S that comes back is independent of the product
        put_solver = fd.solvers.FD1DThetaMethod(model, put_option)
        U_put, S = put_solver.solve()
        
        lower_bound = 45
        upper_bound = 55
        logical_index = np.logical_and(S >= lower_bound, S <= upper_bound)
        stock = S[logical_index]
        analytical_put_price = analytical.black_scholes_put(stock, strike, maturity, 
                short_rate, volatility)
        fd_put_price = U_put[logical_index]
        
        for i in range(len(stock)):
            assert fd_put_price[i] == approx(analytical_put_price[i], rel=1e-3)


    def test_black_scholes_call(self):
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
        
        # TODO: The S that comes back is independent of the product
        call_solver = fd.solvers.FD1DThetaMethod(model, call_option)
        U_call, S = call_solver.solve()
        
        lower_bound = 45
        upper_bound = 55
        logical_index = np.logical_and(S >= lower_bound, S <= upper_bound)
        stock = S[logical_index]
        analytical_call_price = analytical.black_scholes_call(stock, strike, maturity, 
                short_rate, volatility)
        fd_call_price = U_call[logical_index]
        
        for i in range(len(stock)):
            assert fd_call_price[i] == approx(analytical_call_price[i], rel=1e-3)
        
        
    def test_CEV_put(self):
        BS_volatility = 0.4
        short_rate = 0.06
        time_steps = 40
        stock_steps = 80
        stock_min = 0
        stock_max = 150
        strike = 50
        maturity = 1
        exponent = 0.8
        CEV_volatility = BS_volatility * strike ** (1 - exponent)
        
        model = fd.models.FDCEVModel(exponent, CEV_volatility, short_rate, 
        (stock_min, stock_max), stock_steps, time_steps)

        put_option = fd.products.FDEuropeanPutOption(strike, maturity, 
                short_rate)

        # TODO: The S that comes back is independent of the product
        put_solver = fd.solvers.FD1DThetaMethod(model, put_option)
        U_put, S = put_solver.solve()

        lower_bound = 45
        upper_bound = 55
        logical_index = np.logical_and(S >= lower_bound, S <= upper_bound)
        stock = S[logical_index]
        analytical_put_price = analytical.CEV_put(stock, strike, maturity, 
                short_rate, CEV_volatility, exponent)
        fd_put_price = U_put[logical_index]

        for i in range(len(stock)):
            assert fd_put_price[i] == approx(analytical_put_price[i], rel=1e-3)
            
    
    def test_CEV_call(self):
        BS_volatility = 0.4
        short_rate = 0.06
        time_steps = 40
        stock_steps = 80
        stock_min = 0
        stock_max = 150
        strike = 50
        maturity = 1
        exponent = 0.8
        CEV_volatility = BS_volatility * strike ** (1 - exponent)
        
        model = fd.models.FDCEVModel(exponent, CEV_volatility, short_rate, 
        (stock_min, stock_max), stock_steps, time_steps)

        call_option = fd.products.FDEuropeanCallOption(strike, maturity, short_rate)

        # TODO: The S that comes back is independent of the product
        call_solver = fd.solvers.FD1DThetaMethod(model, call_option)
        U_call, S = call_solver.solve()

        lower_bound = 45
        upper_bound = 55
        logical_index = np.logical_and(S >= lower_bound, S <= upper_bound)
        stock = S[logical_index]
        analytical_call_price = analytical.CEV_call(stock, strike, maturity, 
                short_rate, CEV_volatility, exponent)
        fd_call_price = U_call[logical_index]

        for i in range(len(stock)):
            assert fd_call_price[i] == approx(analytical_call_price[i], rel=1e-3)



