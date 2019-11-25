# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:54:30 2019

@author: thera
"""
# %% Imports and Seeds
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

np.random.seed(0)
torch.manual_seed(0)

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
filename = 'bs_delta_1'
S0 = 100
K = 100
T = 1/50
r = 0.05
sigma = 0.2
num_samples = 1000;
num_epochs  = 15;
batch_size = 4;
S0_lower_bound = 90;
S0_upper_bound = 110;

uniform_samples = np.random.rand(num_samples, 1)
S0_values = (S0_upper_bound - S0_lower_bound) * uniform_samples + S0_lower_bound
delta_values = delta_call_BS(S0_values, K, T, r, sigma)

# %% Construct Neural Net
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.lin1 = nn.Linear(1, 16)
        self.lin2 = nn.Linear(16, 1)
        self.sigmoid1 = nn.Sigmoid();
        
    def forward(self, S0):
        out = self.lin1(S0)
        out = self.lin2(out)
        out = self.sigmoid1(out)
        return out

net = Net()

# %% Create Data Loaders
training_set = torch.utils.data.TensorDataset(torch.Tensor(uniform_samples), 
        torch.Tensor(delta_values))
training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size,
                                              shuffle=True)

# %% Define Loss Function and Optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.1)

# %% Test Net Construction
#print('Layers in Net:\n', net)
#params = list(net.parameters())
#print('Parameters per layer:')
#for i in range(len(params)):
#    print(params[i].shape)
#
#data_iter = iter(training_loader)
#inputs, targets = data_iter.next()
#
#optimizer.zero_grad()
#outputs = net(inputs)
#loss = criterion(outputs, targets)
#
#loss.backward()
#for param in net.parameters():
#    print(param.grad.data.sum())
#optimizer.step()


# %% Train network
plt.figure(1)
test_set = torch.Tensor(np.linspace(0, 1).reshape((50,1)))
target_set =  delta_call_BS((S0_upper_bound - S0_lower_bound) * test_set 
                            + S0_lower_bound, K, T, r, sigma)
output_set = np.zeros((50, num_epochs));

num_iterations = (num_samples / batch_size) - 1
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(training_loader, 0):
        inputs, targets = data;
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward + Backward + Optimize
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
       
        # Print statistics
        running_loss += loss.item()
    
    print('[%d] loss: %.6f' % (epoch + 1, running_loss))
    predictions = net(test_set)
    output_set[:, epoch] = predictions.detach().numpy().flatten();
    plt.plot(test_set.detach().numpy(), predictions.detach().numpy())
    
    
# %% Figures

#predictions = net(test_set)

plt.figure(1)
#plt.plot(uniform_samples, delta_values, 'k.')
plt.plot(test_set.detach().numpy(), predictions.detach().numpy())
plt.title('Analytical Black-Scholes Put Delta')
plt.xlabel('S_0')

np.savez(filename, test_set=test_set, output_set=output_set, target_set=target_set)














