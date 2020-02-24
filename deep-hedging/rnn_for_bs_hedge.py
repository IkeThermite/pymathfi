# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 12:10:19 2019

@author: thera
"""
# %% Imports and Seeds
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
#import matplotlib.pyplot as plt

np.random.seed(0)
torch.manual_seed(0)

# %% Training Parameters
num_samples = 50000
batch_size = 200
num_epochs = 20
learning_rate = 0.01
num_neurons = 16


# %% Functions
def d1(S0, K, T, r, sigma):
    return (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

def d2(S0, K, T, r, sigma):
    return d1(S0, K, T, r, sigma) - sigma * np.sqrt(T)

def price_put_BS(S0, K, T, r, sigma):
    return (stats.norm.cdf(-d2(S0, K, T, r, sigma)) * K * np.exp(-r * T) - 
                           stats.norm.cdf(-d1(S0, K, T, r, sigma)) * S0)
def price_call_BS(S0, K, T, r, sigma):
    return (stats.norm.cdf(d1(S0, K, T, r, sigma)) * S0 - 
            stats.norm.cdf(d2(S0, K, T, r, sigma)) * K * np.exp(-r * T))

def delta_put_BS(S0, K, T, r, sigma):
    return -stats.norm.cdf(-d1(S0, K, T, r, sigma))

def delta_call_BS(S0, K, T, r, sigma):
    return stats.norm.cdf(d1(S0, K, T, r, sigma));

# %% Parameters
filename = 'rnn_bs_hedge'
test_model_construction = False
S0 = 100
K = 100
T = 1/50
r = 0
sigma = 0.2
S0_lower_bound = 90;
S0_upper_bound = 110;
make_gif = True
num_timesteps = 2
dt = T / num_timesteps

uniform_samples = np.random.rand(num_samples, 1)
normal_samples_1 = np.random.randn(num_samples, 1)
normal_samples_2 = np.random.randn(num_samples, 1)
W0_values = uniform_samples - 0.5;
delta_W1 = normal_samples_1 * np.sqrt(dt)
delta_W2 = normal_samples_2 * np.sqrt(dt)

S0_values = (S0_upper_bound - S0_lower_bound) * uniform_samples + S0_lower_bound
S1_values = S0_values * np.exp((r - 0.5 * sigma **2) * dt 
                               + sigma * delta_W1)
S2_values = S1_values * np.exp((r - 0.5 * sigma **2) * dt 
                               + sigma * delta_W2)
call_values = price_call_BS(S0_values, K, T, r, sigma)

# %% Create Data Loaders
training_set = torch.utils.data.TensorDataset(torch.Tensor(np.hstack([W0_values, delta_W1])),
        torch.Tensor(S0_values),                                      
        torch.Tensor(S1_values),
        torch.Tensor(S2_values),
        torch.Tensor(call_values))
training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size,
                                              shuffle=True)

# %% Construct RNN
class Net(nn.Module):
    def __init__(self, num_neurons):
        super(Net, self).__init__()
        self.lin1 = nn.Linear(2, num_neurons)
        self.sig1 = nn.Sigmoid()
        self.bn1 = nn.BatchNorm1d(num_neurons)
        self.lin2 = nn.Linear(num_neurons, 1)
        self.sig2 = nn.Sigmoid()
        

    def forward(self, X):
        batch_size, steps, _ = X.shape
        output = []
        self.y = torch.rand(batch_size, 1)
        
        for i in range(steps):
            self.y = torch.cat((X[:,i,:], self.y), dim=1)
            self.y = self.bn1(self.sig1(self.lin1(self.y)))
            self.y = self.sig2(self.lin2(self.y));
            output.append(self.y)
        
        return output, self.y

net = Net(num_neurons)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

# %% Test Model Construction
if test_model_construction:
    
    
    print('Layers in Net:\n', net)
    params = list(net.parameters())
    print('Parameters per layer:')
    for i in range(len(params)):
        print(params[i].shape)
    
    data_iter = iter(training_loader)
    X, S0, S1, S2, C0 = data_iter.next()
    X.unsqueeze_(-1)    
    
    optimizer.zero_grad()
    outputs, h_end = net(X)
    loss = criterion(outputs[1] * (S2 - S1) +
                     outputs[0] * (S1 - S0) + C0, 
                     torch.max(S2 - K, torch.zeros(batch_size, 1)))
    
    loss.backward()
    for param in net.parameters():
        print(param.grad.data.sum())
    optimizer.step()

# %% Train Recurrent Neural Network
num_iterations = (num_samples / batch_size) - 1

grid_size = 50
grid_uniform = np.linspace(0, 1, num=grid_size+2)[1:-1]
grid_W0 = grid_uniform - 0.5
grid_dW1 = stats.norm.ppf(grid_uniform) * np.sqrt(dt)
grid_S0 = (S0_upper_bound - S0_lower_bound) * grid_uniform + S0_lower_bound
grid_S1 = grid_S0 * np.exp((r - 0.5 * sigma **2) * dt + sigma * grid_dW1)
grid_X = torch.Tensor(np.hstack([grid_W0.reshape(grid_size,1), grid_dW1.reshape(grid_size,1)]))
grid_X.unsqueeze_(-1)

BS_delta_0 = delta_call_BS(grid_S0, K, T, r, sigma)
BS_delta_1 = delta_call_BS(grid_S1, K, T-dt, r, sigma)

RNN_delta_0 = np.zeros((grid_size, num_epochs))
RNN_delta_1 = np.zeros((grid_size, num_epochs))

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(training_loader):
        X, S0, S1, S2, C0 = data
        X.unsqueeze_(-1)
        
        optimizer.zero_grad()
        outputs, _ = net(X)
        loss = criterion(outputs[1] * (S2 - S1) +
                         outputs[0] * (S1 - S0) + C0, 
                         torch.max(S2 - K, torch.zeros(batch_size, 1)))
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * batch_size
    
    print('[%d] loss: %.6f' % (epoch + 1, running_loss))
    predictions, _ = net(grid_X)
    RNN_delta_0[:, epoch] = predictions[0].detach().numpy().flatten()
    RNN_delta_1[:, epoch] = predictions[1].detach().numpy().flatten()

torch.save(net, 'rnn.pth')
np.savez(filename, 
         grid_S0=grid_S0,
         grid_S1=grid_S1,
         BS_delta_0=BS_delta_0,
         BS_delta_1=BS_delta_1,
         RNN_delta_0=RNN_delta_0,
         RNN_delta_1=RNN_delta_1)

#grid_X.unsqueeze_(-1)
#outputs, _ = net(grid_X)
#
#plt.figure()
#plt.subplot(1, 2, 1)
#plt.plot(grid_S0, outputs[0].detach().numpy().flatten(), 'b-')
#plt.plot(grid_S0, BS_delta_0, 'k-')
#
#plt.subplot(1, 2, 2)
#plt.plot(grid_S1, outputs[1].detach().numpy().flatten(), 'b-')
#plt.plot(grid_S1, BS_delta_1, 'k-')

































