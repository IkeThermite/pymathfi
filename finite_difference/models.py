import numpy as np

class FDBlackScholesModel():
    
    def __init__(self, volatility, short_rate, stock_bounds, stock_steps, 
            time_steps):
        self.volatility = volatility
        self.short_rate = short_rate
        self.stock_bounds = stock_bounds
        self.stock_steps = stock_steps
        self.time_steps = time_steps
    
    def local_volatility(self, stock, time):
        # Inherit appropriate vector size
        ones = np.ones(np.asarray(stock).shape) * np.ones(np.asarray(time).shape)
        return self.volatility * ones

class FDCEVModel():

    def __init__(self, exponent, volatility, short_rate, stock_bounds, 
            stock_steps, time_steps):
        self.exponent = exponent
        self.volatility = volatility
        self.short_rate = short_rate
        self.stock_bounds = stock_bounds
        self.stock_steps = stock_steps
        self.time_steps = time_steps
    
    def local_volatility(self, stock, time):
        # Inherit appropriate vector size
        ones = np.ones(np.asarray(stock).shape) * np.ones(np.asarray(time).shape)
        return self.volatility * stock ** (self.exponent - 1) * ones