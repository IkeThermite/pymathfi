"""
Products for use with finite difference methods.
"""
import numpy as np

class FDProduct():
    def __init__(self, maturity):
        self.maturity = maturity

    def lower_boundary(self, stock_min, time):
        raise NotImplementedError
    
    def upper_boundary(self, stock_max, time):
        raise NotImplementedError

    def terminal_condition(self, stock, time):
        raise NotImplementedError


class FDEuropeanCallOption(FDProduct):
    def __init__(self, strike, maturity, short_rate):
        super().__init__(maturity)
        self.strike = strike
        self.short_rate = short_rate
        # TODO: It's a problem that the short rate must be here and in model
    
    def lower_boundary(self, stock_min, time):
        return 0 * time

    def upper_boundary(self, stock_max, time):
        return stock_max - self.strike * np.exp(-self.short_rate 
                * (self.maturity - time))
    
    def terminal_condition(self, stock, time):
        return np.maximum(stock - self.strike, 0)


class FDEuropeanPutOption(FDProduct):
    def __init__(self, strike, maturity, short_rate):
        super().__init__(maturity)
        self.strike = strike
        self.short_rate = short_rate
        # TODO: It's a problem that the short rate must be here and in model
    
    def lower_boundary(self, stock_min, time):
        return  -stock_min + self.strike * np.exp(-self.short_rate 
                * (self.maturity - time))

    def upper_boundary(self, stock_max, time):
        return 0 * time
    
    def terminal_condition(self, stock, time):
        return np.maximum(self.strike - stock, 0)