# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 10:44:54 2019
https://medium.com/dair-ai/building-rnns-is-fun-with-pytorch-and-google-colab-3903ea9a3a79
@author: thera
"""
import torch
import torch.nn as nn

class SingleRNN(nn.Module):
    def __init__(self, n_inputs, n_neurons):
        super(SingleRNN, self).__init__()
        
        self.Wx= torch.randn(n_inputs, n_neurons) # 4 x 1
        self.Wy = torch.randn(n_neurons, n_neurons) # 1 x 1
        
        self.b = torch.zeros(1, n_neurons) # 1 x 4
        
    def forward(self, X0, X1):
        self.Y0 = torch.tanh(torch.mm(X0, self.Wx) + self.b) # 4 x 1
        self.Y1 = torch.tanh(torch.mm(self.Y0, self.Wy) + 
                             torch.mm(X1, self.Wx) + self.b) # 4 x 1
        
        return self.Y0, self.Y1
    
n_inputs = 4
n_neurons = 1

X0_batch = torch.tensor([[0, 1, 2, 0], [3, 4, 5, 0],
                         [6, 7, 8, 0], [9, 0, 1, 0]],
                        dtype=torch.float) #t=0 -> 4 x 4
X1_batch = torch.tensor([[9, 8, 7, 0], [0, 0, 0, 0],
                         [6, 5, 4, 0], [3, 2, 1, 0]],
                        dtype=torch.float) #t=1 -> 4 x 4
model = SingleRNN(n_inputs, n_neurons)
Y0_val, Y1_val = model(X0_batch, X1_batch)
