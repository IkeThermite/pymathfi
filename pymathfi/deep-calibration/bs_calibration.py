# Imports and Seeds
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
np.random.seed(0)
torch.manual_seed(0)

# Pricing Functions
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

# Model
class Net(nn.Module):
    def __init__(self, num_neurons):
        super(Net, self).__init__()
        self.lin1 = nn.Linear(1, num_neurons)
        self.softplus1 = nn.Softplus()
        self.bn2 = nn.BatchNorm1d(num_neurons, affine=False)
        self.lin2 = nn.Linear(num_neurons, 1)
        self.softplus2 = nn.Softplus()
        
    def forward(self, out):
        out = self.softplus1(self.lin1(out))
        out = self.bn2(out)
        out = self.softplus2(self.lin2(out))
        return out

# Generate Training Data and Data Loaders
training_samples = 100000
training_batch_size = 200
sigma_upper_bound = 1.5
sigma_lower_bound = 0.05
sigma = np.reshape(np.linspace(sigma_lower_bound, 
                               sigma_upper_bound, training_samples), 
        (training_samples, 1))

put_prices = np.zeros((training_samples, 1))

S0 = 100
K = 100
r = 0.1;
T = 5;

for i in range(training_samples):
    put_prices[i] = price_put_BS(S0, K, T, r, sigma[i])
     
    
training_set = torch.utils.data.TensorDataset(torch.Tensor(put_prices),
                                              torch.Tensor(sigma))
training_loader = torch.utils.data.DataLoader(training_set, 
                                              batch_size=training_batch_size,
                                              shuffle=True)

#plt.plot(put_prices, sigma)
# Initialize Net
num_neurons = 16
net = Net(num_neurons)

# Define Loss Function and Optimizer, Train Network
num_epochs = 20
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate)

test_samples = 501
test_targets = np.reshape(np.linspace(sigma_lower_bound, 
                               sigma_upper_bound, test_samples), (test_samples, 1))
test_inputs = np.zeros((test_samples, 1))
for i in range(test_samples):
    test_inputs[i] = price_put_BS(S0, K, T, r, test_targets[i])

test_predictions = np.zeros((test_samples, num_epochs))
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
    
    print(f'[{epoch + 1}] Loss: {running_loss}')    
    predictions = net(torch.tensor(test_inputs, dtype=torch.float32))
    test_predictions[:, epoch] = predictions.detach().numpy().flatten()

filename = 'bs_calibration'
torch.save(net, filename + '.pth')
np.savez(filename, inputs=test_inputs, targets=test_targets, predictions=test_predictions)
 
## Create animation
#def init():
#    line1.set_data([], [])
#def animate(i):
#    i = min(i, num_epochs)
#    line1.set_data(test_inputs, test_predictions[:, i])
#    return
#
#num_frames = num_epochs + 8
#fig = plt.figure()
#ax = plt.axes()
#line1, = ax.plot([], [], lw=2)
#plt.plot(test_inputs, test_targets, 'k-')
#anim = FuncAnimation(fig, animate, frames=num_frames, init_func=init, interval=100)

# Test Net Construction
#print('Layers in Net:\n', net)
#params = list(net.parameters())
#print('Parameters per layer:')
#for i in range(len(params)):
#    print(params[i].shape)
#    
#data_iter = iter(training_loader)
#training_prices, training_sigmas = data_iter.next()
#
#optimizer.zero_grad()
#output_sigmas = net(training_prices)
#loss = criterion(output_sigmas, training_sigmas)
#
#loss.backward()
#for param in net.parameters():
#    print(param.grad.data.sum())
#optimizer.step()
