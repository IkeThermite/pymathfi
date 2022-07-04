# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 10:44:54 2019
https://medium.com/dair-ai/building-rnns-is-fun-with-pytorch-and-google-colab-3903ea9a3a79
@author: thera
"""
import torch
import torch.nn as nn

n_inputs = 3
n_neurons = 1
batch_size = 4
rnn = nn.RNNCell(n_inputs, n_neurons)

# Batch size is 4, sequence size / num timesteps is 2
X_batch = torch.tensor([[[0,1,2], [3,4,5], 
                         [6,7,8], [9,0,1]],
                        [[9,8,7], [0,0,0], 
                         [6,5,4], [3,2,1]]
                       ], dtype = torch.float) # X0 and X1
print(X_batch.size())
# Goes: time-steps x batch_size x num_inputs per timestep
# Output size should be: batch_size x n_neurons at each step
y = torch.zeros(batch_size, n_neurons)
output = []

# for each time step 
for i in range(2):
    y = rnn(X_batch[i], y) #same as model.forward
    output.append(y)
    
print(output)

# torch.RNNCell accepts a tensor as input and outputs the next hidden state
# %% Push this architecture into a graph
class CleanBasicRNN(nn.Module):
    def __init__(self, batch_size, n_inputs, n_neurons):
        super(CleanBasicRNN, self).__init__()
        
        self.rnn = nn.RNNCell(n_inputs, n_neurons)
        self.hy = torch.zeros(batch_size, n_neurons) # initialize hidden state
        
    def forward(self, X):
        output = []
        
        # for each time step
        for i in range(2):
            self.hy = self.rnn(X[i], self.hy)
            output.append(self.hy)
        
        return output, self.hy
        
n_inputs = 3
n_neurons = 1
batch_size = 4

X_batch = torch.tensor([[[0,1,2], [3,4,5], 
                         [6,7,8], [9,0,1]],
                        [[9,8,7], [0,0,0], 
                         [6,5,4], [3,2,1]]
                       ], dtype = torch.float) # X0 and X1
model = CleanBasicRNN(batch_size, n_inputs, n_neurons)
output_val, states_val = model(X_batch)
print(output_val) # contains all output for all timesteps
print(states_val) # contains values for final state or final timestep, i.e., t=1