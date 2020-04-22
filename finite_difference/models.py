
class FDBlackScholesModel():
    
    def __init__(self, volatility, short_rate, stock_bounds, stock_steps, time_steps):
        self.volatility = volatility
        self.short_rate = short_rate
        self.stock_bounds = stock_bounds
        self.stock_steps = stock_steps
        self.time_steps = time_steps