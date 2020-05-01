import numpy as np

class MCEuropeanCallOption():
    def __init__(self, strike, maturity, short_rate):
        self.strike = strike
        self.maturity = maturity
        self.short_rate = short_rate
        self.timeline = [maturity]
    
    def discounted_payoff(self, stock, current_time):
        # stock.shape is (num_samples, len(self.timeline))
        return (np.exp(-self.short_rate * (self.maturity - current_time)) * 
                np.maximum(stock - self.strike, 0))


class MCGeometricBrownianMotion():
    def __init__(self, volatility, short_rate, initial_stock_value):
        self.volatility = volatility
        self.short_rate = short_rate
        self.S0 = initial_stock_value
    
    def simulate(self, num_samples, timeline):
        # Convert to array with non-zero columns
        # https://stackoverflow.com/questions/36009907/numpy-reshape-1d-to-2d-array-with-1-column
        timeline = np.asarray(timeline).reshape(1, -1)
        timeline = np.tile(timeline, (num_samples, 1))
        delta_t = np.diff(np.hstack((np.zeros((timeline.shape[0], 1)), timeline)))
        Z = np.random.normal(size=timeline.shape)
        simulated_asset = self.S0 * np.exp(np.cumsum((self.short_rate - 
                0.5 * self.volatility ** 2) * delta_t + self.volatility * 
                np.sqrt(delta_t) * Z, axis=1))
        return simulated_asset


class  MCPricer():
    def __init__(self, product, model):
        self.model = model
        self.product = product
    
    def price(self, current_time, num_samples):
        paths = self.model.simulate(num_samples, self.product.timeline)
        samples = self.product.discounted_payoff(paths, current_time)
        mc_price = np.mean(samples)
        mc_std = np.std(samples) / np.sqrt(num_samples)
        return mc_price, mc_std