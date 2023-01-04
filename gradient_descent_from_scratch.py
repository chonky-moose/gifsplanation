#%%
# Reference:
# https://machinelearningmastery.com/gradient-descent-optimization-from-scratch/

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
torch.set_printoptions(sci_mode=False)
import torchvision
import torchvision.transforms as T

from mnist_classifier import Mnist_CNN
from mnist_vae import VariationalAutoencoder
# %%
def example_function(x):
    return x ** 2.0

# range for input
r_min, r_max = -1.0, 1.0
# sample input range uniformly at 0.1 increments
inputs = np.arange(r_min, r_max+0.1, 0.1)
# compute outputs
results = example_function(inputs)
# show inputs and outputs on line plot
plt.plot(inputs, results)

#%%
def gradient_descent(fn, derivative, bounds, n_iter, step_size):
    xs, ys = [], []
    # generate an initial point
    x = bounds[:, 0] + np.random.rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    for i in range(n_iter):
        gradient = derivative(x)
        x = x - step_size * gradient
        y = fn(x)
        xs.append(x)
        ys.append(y)
        print(f'step {i}: f({x[0]:.4f}) = {y[0]:.5f}')
    return xs, ys

#%%
# derivative of example function
def derivative(x):
    return x * 2.0

# define the range for input
bounds = np.asarray([[-1.0, 1.0]])
# number of iterals
n_iter = 30
# maximum step size
step_size = 0.9

xs, ys = gradient_descent(example_function, derivative,
                          bounds, n_iter, step_size)
# %%
# Plot the function (blue) and plot the descent (red)
inputs = np.arange(bounds[0,0], bounds[0,1]+0.1, 0.1)
results = example_function(inputs)
plt.plot(inputs, results)
plt.plot(xs, ys, '.-', color='red')


# %%
#############################################################################
#############################################################################
#############################################################################
#############################################################################

#%%
# Gradient Ascent
def fn(x):
    return -x**2
r_min, r_max = -1.0, 1.0
inputs = np.arange(r_min, r_max+0.1, 0.1)
results = fn(inputs)
plt.plot(inputs, results)
# %%
def gradient_ascent(fn, inputs, n_iter, step_size):    
    xs, ys = [], []
    # choose a random initial point
    ind = np.random.choice(np.arange(len(inputs))) 
    x = inputs[ind]
    x.retain_grad()
    y = fn(x)
    for i in range(n_iter):
        y.backward()
        gradient = x.grad
        x = x + step_size * gradient
        x.retain_grad()
        y = fn(x)
        xs.append(x.item())
        ys.append(y.item())
        print(f'step {i}: f({x:.4f}) = {y:.5f}')
    return xs, ys
# %%
xs, ys = gradient_ascent(fn, inputs, 30, 0.9)
# %%
# Plot the function (blue) and plot the descent (red)
results = fn(inputs)
plt.plot(inputs.detach().numpy(), results.detach().numpy())
plt.plot(xs, ys, '.-', color='red')


################################################################################
################################################################################
################################################################################
################################################################################
# %%
# Apply gradient descent/ascent on a complex function (e.g. a neural network)
cnn = Mnist_CNN((1,28,28), 10)
cnn.load_state_dict(torch.load(r'./mnist_cnn.pt'))

tf = T.Compose([
    T.ToTensor(),
    T.Normalize((0.1307,), (0.3081,))
])
trainset = torchvision.datasets.MNIST(r'../DATASETS/mnist', train=True,
                                    download=True, transform=tf)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                            batch_size=1,
                                            shuffle=True)
x, label = next(iter(train_loader))
x.requires_grad = True
x.retain_grad()
tmp = x
print(x.shape, label)
#%%
fig, ax = plt.subplots(1,1, figsize=(3,3))
ax.imshow(x[0][0].detach().numpy())

prediction_for = 1
y = cnn(x)[0][prediction_for]
print('y', y)
y.backward()
print('x.grad', x.grad.shape)

#%%
n_iters = 100
step_size = 0.1
fig, ax = plt.subplots(10, 10, figsize=(20,20))
for i in range(n_iters):
    y = cnn(x)[0][prediction_for]
    print(y)
    y.backward()
    gradient = x.grad
    x = x + step_size * gradient
    x.retain_grad()
    ax[i//10, i%10].imshow(x[0][0].detach().numpy(), cmap='gray')
# %%
