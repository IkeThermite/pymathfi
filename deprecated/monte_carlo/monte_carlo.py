import numpy as np
import scipy.stats as stats

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
        Z = stats.norm.rvs(size=timeline.shape)
        simulated_asset = self.S0 * np.exp(np.cumsum((self.short_rate - 
                0.5 * self.volatility ** 2) * delta_t + self.volatility * 
                np.sqrt(delta_t) * Z, axis=1))
        return simulated_asset


class  MCPricer():
    def __init__(self, product, model):
        self.model = model
        self.product = product
        self.sum = 0
        self.squared_sum = 0
        self.N = 0
        self.CONFIDENCE_LEVEL = 0.05
        self.MAX_BATCH_SIZE = 1e6
    
    def price(self, current_time, num_samples, target_accuracy=None):
        paths = self.model.simulate(num_samples, self.product.timeline)
        samples = self.product.discounted_payoff(paths, current_time)
        
        self.sum += np.sum(samples)
        self.squared_sum += np.sum(samples ** 2) 
        self.N += num_samples

        sample_mean = self.sum / self.N
        sample_var = self.squared_sum / self.N - sample_mean ** 2
        mc_std = np.sqrt(sample_var / (self.N - 1))
        mc_price = sample_mean
        accuracy = stats.norm.ppf(1 - self.CONFIDENCE_LEVEL) * mc_std
       
        if target_accuracy is not None:
            next_batch_size = 1
            print(f"MCPricer.price: Increasing simulations for target accuracy.")
            while ((accuracy > target_accuracy) and (next_batch_size > 0)):
                batch_size_estimate = (accuracy / target_accuracy) ** 2 * self.N - self.N          
                next_batch_size = int(np.minimum(np.floor(batch_size_estimate), self.MAX_BATCH_SIZE))

                print(f"Accuracy: {accuracy:8.6f}, Next Batch Size: {next_batch_size}")
                mc_price, mc_std, accuracy = self.price(current_time, next_batch_size)
            print(f"MCPricer.price: Final Accuracy: {accuracy:8.6f}")
        
        return mc_price, mc_std, accuracy
        
        # mc_price = np.mean(samples)
        # mc_var = np.var(samples) / num_samples
        # mc_std = np.sqrt(mc_var)

        # return mc_price, mc_std