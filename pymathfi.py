import analytical
import numpy as np
# import finite_difference

# S0 = 100
# K = 100
# T = 1/50
# r = 0.05
# sigma = 0.2

# price = analytical.pricing.price_call_BS(S0, K, T, r, sigma)
# print(price)

#%% Implementing Theta Method
volatility = 0.2
short_rate = 0.1
time_steps = 5
space_steps = 5
stock_min = 50
stock_max = 150

strike = 100
maturity = 1

def lower_boundary(stock_min, tau):
        # European call option lower boundary
        return 0 * tau

def upper_boundary(stock_max, tau):
    # European call option upper boundary
    return stock_max - np.exp(- short_rate * tau) * strike
    
def initial_condition(stock):
    # European call option payoff
    return np.maximum(stock - strike, 0)


theta = 0.5
sig = volatility
r = short_rate
N = space_steps
M = time_steps
T = maturity
Smin = stock_min
Smax = stock_max
dS = (Smax - Smin)/N
dtau = T/M
S = Smin + np.arange(0, N + 1) * dS
tau = np.arange(0, M + 1) * dtau

I = np.eye(N - 1)
T1 = np.diag(-1 * np.ones((N - 2)), -1) + np.diag(np.ones((N - 2)), 1)
T2 = (np.diag(np.ones((N - 2)), -1) + np.diag(np.ones((N - 2)), 1) +
        np.diag(-2 * np.ones(N - 1)))
D1 = np.diag(Smin/dS + np.arange(1, N))
D2 = D1 ** 2
F = (1 - r * dtau) * I + 0.5 * r * dtau * D1 @ T1 + 0.5 * sig ** 2 * dtau * D2 @ T2
G = 2 * I - F
bl = 0.5 * dtau * (Smin / dS + 1) * (sig ** 2 * (Smin / dS + 1) - r) * lower_boundary(Smin, tau)
bu = 0.5 * dtau * (Smax / dS - 1) * (sig ** 2 * (Smax / dS - 1) + r) * upper_boundary(Smax, tau)
B = np.vstack((bl, np.zeros((N - 3, M + 1)), bu))

U = np.zeros((N - 1, M + 1))
U[:, 0] = initial_condition(S[1:-1])
for i in range(1, M):
    A = theta * G + (1 - theta) * I
    b = ((1 - theta) * F + theta * I) @ U[:, i - 1] + (1 - theta) * B[:, i - 1] + theta * B[:, i]
    U[:, i] = np.linalg.solve(A, b)
            
