import numpy as np

class FD1DThetaMethod():

    def __init__(self, volatility, short_rate, time_steps, space_steps, stock_min, 
            stock_max, strike, maturity):
        self.volatility = volatility
        self.short_rate = short_rate
        self.time_steps = time_steps
        self.space_steps = space_steps
        self.stock_min = stock_min
        self.stock_max = stock_max

        self.strike = strike
        self.maturity = maturity

    def lower_boundary(self, stock_min, time):
        # European call option lower boundary
        return 0

    def upper_boundary(self, stock_max, time):
        # European call option upper boundary
        return stock_max - (np.exp(- self.short_rate * (self.maturity - time)) 
            * self.strike)
    
    def initial_condition(self, stock):
        # European call option payoff
        return np.max(stock - self.strike, 0)

    def solve(self):
        N = self.space_steps
        M = self.time_steps
        T = self.maturity
        Smin = self.stock_min
        Smax = self.stock_max
        dS = (Smax - Smin)/N
        dtau = T/M
        
        I = np.eye(N - 1)
        T1 = np.diag(-1 * np.ones((N - 2)), -1) + np.diag(np.ones((N - 2)), 1)
        T2 = (np.diag(np.ones((N - 2)), -1) + np.diag(np.ones((N - 2)), 1) +
                np.diag(-2 * np.ones(N - 1)))
        D1 = np.diag(Smin/dS + np.arange(1, N - 1))
        D2 = D1 ** 2



if (__name__ == '__main__'):
    print(f'Executing {__file__} as a standalone script')