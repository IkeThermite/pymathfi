# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 11:09:51 2019
https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_recurrent_neuralnetwork/
@author: thera
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets

# %% Loading MNIST Dataset

train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)
test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())
print("Training Data Set Size: {0}".format(train_dataset.train_data.size()))
print("Number of Labels: {0}".format(train_dataset.train_labels.size()))
print("Test Data Set Size: {0}".format(test_dataset.test_data.size()))
print("Number of Labels: {0}".format(test_dataset.test_labels.size()))

# %% Creating Iterables
batch_size = 100
n_iters = 3000 # Total number of iterations
num_epochs = n_iters // (len(train_dataset) // batch_size) # Divided into epochs

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# %% Create Model Class - Model A: 1 Hidden Layer
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()
        # Hidden dimension
        self.hidden_dim = hidden_dim
        
        # Number of hidden layers
        self.layer_dim = layer_dim
        
        # Building RNN
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, input_dim)
        # batch_dim = number of samples per batch
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        # (layer_dim, batch_size, hidden_dim)
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        
        # We need to detach the hidden state to prevent exploding / vanishing gradients
        # This is part of truncated backpropagation through time (BPTT)
        out, hn = self.rnn(x, h0.detach())
        
        # Index hideen state of last time step
        # out.size() --> 100, 28, 10
        # out.[:, -1, :] --> 100, 10 --> just want the last time step hidden states!
        out = self.fc(out[:, -1, :])
        # out.size() --> 100, 10
        return out
    
# %% Instantiate Model Class
input_dim = 28
hidden_dim = 100
layer_dim = 1
output_dim = 10
model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim)

# %% Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# %% Check Model Construction and Paramters
print('Layers in Net:\n', model)
params = list(model.parameters())
print('Parameters per layer:')
for i in range(len(params)):
    print(params[i].shape)

# %% Train Model
seq_dim = 28 # Number of steps to unroll

iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        model.train()
        # Load images as tensors with gradient accumulation abilities
        images = images.view(-1, seq_dim, input_dim)

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()
        
        # Forward pass to get output / logits
        # outputs.size() --> 100, 10
        outputs = model(images)
        
        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)
        
        # Getting gradients w.r.t. parameters
        loss.backward()
        
        # Updated parameters
        optimizer.step()
        
        iter += 1;
        
        if iter % 500 == 0:
            model.eval()
            # Calculate accuracy
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in test_loader:
                # Load images as tensors with gradient accumulation abilities
                images = images.view(-1, seq_dim, input_dim)
                
                # Forward pass only to get logits / output
                outputs = model(images)
        
                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)
                
                # Total number of labels
                total += labels.size(0)
                
                # Total correct predictions
                correct += (predicted == labels).sum()
            
            accuracy = 100 * correct / total
            
            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        