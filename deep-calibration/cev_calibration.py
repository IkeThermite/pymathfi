# Imports and Seeds
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.optim as optim

np.random.seed(0)
torch.manual_seed(0)

# %% Pricing Functions
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

def price_put_CEV(S0, K, T, r, sigma, alpha):
    kappa = 2 * r / (sigma ** 2 * (1 - alpha)
        * (np.exp(2 * 3 * (1 - alpha) * T) - 1))
    x = kappa * S0 ** (2 * (1 - alpha)) * np.exp(2 * r * (1 - alpha) * T)
    y = kappa * K ** (2 * (1 - alpha))
    z = 2 + 1 / (1 - alpha)
    return -S0 * stats.ncx2.cdf(y, z, x) + (
            K * np.exp(-r * T) * (1 - stats.ncx2.cdf(x, z - 2, y)))

# %% Model
class Net(nn.Module):
    def __init__(self, num_neurons):
        super(Net, self).__init__()
        self.lin1 = nn.Linear(1, num_neurons)
        self.softplus1 = nn.Softplus()
        self.bn2 = nn.BatchNorm1d(num_neurons, affine=False)
        self.lin2 = nn.Linear(num_neurons, 2) # must return alpha and sigma
        self.softplus2 = nn.Softplus()
        
    def forward(self, out):
        out = self.softplus1(self.lin1(out))
        out = self.bn2(out)
        out = self.softplus2(self.lin2(out))
        return out

# %% Generate Training Data and Data Loaders
training_dim1_samples = 250
training_dim2_samples = 250
training_samples = training_dim1_samples * training_dim2_samples
training_batch_size = 200
sigma_upper_bound = 2
sigma_lower_bound = 0.3
sigma = np.linspace(sigma_lower_bound, sigma_upper_bound, training_dim1_samples)
alpha_upper_bound = 0.8
alpha_lower_bound = 0.3
alpha = np.linspace(alpha_lower_bound, alpha_upper_bound, training_dim2_samples)
X, Y = np.meshgrid(sigma, alpha)

put_prices = np.zeros((training_samples, 1))
sigma = np.reshape(X.flatten(), put_prices.shape) 
alpha = np.reshape(Y.flatten(), put_prices.shape) 

S0 = 100
K = 100
r = 0.05;
T = 1;

for i in range(training_samples):
    put_prices[i] = price_put_CEV(S0, K, T, r, sigma[i], alpha[i])

#Z = np.reshape(put_prices, X.shape)
#fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax.plot_surface(X, Y, Z, cmap=cm.plasma)

training_set = torch.utils.data.TensorDataset(torch.Tensor(put_prices),
                                              torch.Tensor(np.hstack((sigma, alpha))))
training_loader = torch.utils.data.DataLoader(training_set, 
                                              batch_size=training_batch_size,
                                              shuffle=True)
# %% Generate Testing (out of sample) Data
test_dim1_samples = 15
test_dim2_samples = 15
test_samples = test_dim1_samples * test_dim2_samples
test_dim1 = np.linspace(sigma_lower_bound, sigma_upper_bound, test_dim1_samples)
test_dim2 = np.linspace(alpha_lower_bound, alpha_upper_bound, test_dim2_samples)
test_X, test_Y = np.meshgrid(test_dim1, test_dim2)
test_inputs = np.zeros((test_samples, 1))
test_dim1 = np.reshape(test_X.flatten(), test_inputs.shape)
test_dim2 = np.reshape(test_Y.flatten(), test_inputs.shape)

for i in range(test_samples):
    test_inputs[i] = price_put_CEV(S0, K, T, r, test_dim1[i], test_dim2[i])

test_targets = np.hstack((test_dim1, test_dim2))

# %% Initialize Net
num_neurons = 16
net = Net(num_neurons)

# %% Define Loss Function and Optimizer
learning_rate = 0.005
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate)


# %% Train Network
num_epochs = 20
test_predictions = np.zeros((test_samples, 2, num_epochs))

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(training_loader, 0):
        inputs, targets = data
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
    test_predictions[:, :, epoch] = predictions.detach().numpy()

# %% Visually Examine Pricing Performance
test_prices = np.zeros((test_samples, num_epochs))
for epoch in range(num_epochs):
    for i in range(test_samples):
        test_prices[i, epoch] =  price_put_CEV(S0, K, T, r, test_predictions[i, 0, epoch], test_predictions[i, 1, epoch])

test_predictions = test_prices

filename = 'cev_calibration'
torch.save(net, filename + '.pth')
np.savez(filename, inputs=test_inputs, targets=test_targets, predictions=test_predictions)

# Test Net Construction
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
#
#loss = criterion(outputs, targets)
#
#loss.backward()
#for param in net.parameters():
#    print(param.grad.data.sum())
#optimizer.step()
