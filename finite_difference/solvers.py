import numpy as np

class FD1DThetaMethod():

    def __init__(self, model, product):
        self.model = model
        self.product = product
    
    # Reverse time
    # TODO: Could potentially pass the interest rate to the boundary here
    def lower_boundary(self, stock_min, tau):
        return self.product.lower_boundary(stock_min, 
                self.product.maturity - tau)

    def upper_boundary(self, stock_max, tau):
        return self.product.upper_boundary(stock_max, 
                self.product.maturity - tau)
    
    def initial_condition(self, stock, tau):
        return self.product.terminal_condition(stock, 
                self.product.maturity - tau)
    
    def solve(self):
        theta = 0.5
        sig = self.model.volatility
        r = self.model.short_rate
        N = self.model.stock_steps
        M = self.model.time_steps
        Smin = self.model.stock_bounds[0]
        Smax = self.model.stock_bounds[1]
        T = self.product.maturity

        dS = (Smax - Smin) / N
        dtau = T / M
        S = Smin + np.arange(0, N + 1) * dS
        tau = np.arange(0, M + 1) * dtau

        I = np.eye(N - 1)
        T1 = np.diag(-1 * np.ones((N - 2)), -1) + np.diag(np.ones((N - 2)), 1)
        T2 = (np.diag(np.ones((N - 2)), -1) + np.diag(np.ones((N - 2)), 1) +
                np.diag(-2 * np.ones(N - 1)))
        D1 = np.diag(Smin / dS + np.arange(1, N))
        D2 = D1 ** 2
        F = ((1 - r * dtau) * I + 0.5 * r * dtau * D1 @ T1 
                + 0.5 * sig ** 2 * dtau * D2 @ T2)
        G = 2 * I - F
        bl = 0.5 * dtau * (Smin / dS + 1) * (sig ** 2 * (Smin / dS + 1) 
                - r) * self.lower_boundary(Smin, tau)
        bu = 0.5 * dtau * (Smax / dS - 1) * (sig ** 2 * (Smax / dS - 1) 
                + r) * self.upper_boundary(Smax, tau)
        B = np.vstack((bl, np.zeros((N - 3, M + 1)), bu))

        U = np.zeros((N - 1, M + 1))
        U[:, 0] = self.initial_condition(S[1:-1], 0)
        for i in range(1, M + 1):
            A = theta * G + (1 - theta) * I
            b = (((1 - theta) * F + theta * I) @ U[:, i - 1] 
                    + (1 - theta) * B[:, i - 1] + theta * B[:, i])
            U[:, i] = np.linalg.solve(A, b)
        
        return U[:, -1], S[1:-1]
        

if (__name__ == '__main__'):
    print(f'Executing {__file__} as a standalone script')