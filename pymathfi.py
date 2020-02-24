import analytical

S0 = 100
K = 100
T = 1/50
r = 0.05
sigma = 0.2

price = analytical.price_call_BS(S0, K, T, r, sigma)