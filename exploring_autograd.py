#%%
# References
# https://stackoverflow.com/questions/54754153/autograd-grad-for-tensor-in-pytorch

#%%
# What does it mean to take a gradient?
# Let's explore torch.autograd.grad
import numpy as np
import matplotlib.pyplot as plt
import torch
# %%
# Dummy data
x = torch.ones(2,2, requires_grad=True)
print("x", x)
gt = torch.ones_like(x)*16 - 0.5 # ground-truths
print('ground truth', gt)

# Do some computations
y = (x+2) ** 2
print('y', y)

# dydx = torch.autograd.grad(outputs=y, inputs=x) # RuntimeError: grad can be implicitly created only for scalar outputs
dydx = torch.autograd.functional.jacobian(func=lambda var: (var+2)**2, inputs=x)
print('dydx', dydx.shape)
print(dydx)

# Compute loss

loss = torch.nn.MSELoss()(y, gt)
print('Loss', loss)

# Now compute gradients
dloss_dx = torch.autograd.grad(outputs=loss, inputs=x)
print('dloss/dx:', dloss_dx)


#%%
# Reference:
# Video+3+-+Autograd.ipynb
# https://www.youtube.com/watch?v=M0fX15_-xrY&t=79s
# %%
x = torch.linspace(0, 2*np.pi, steps=35, requires_grad=True)
print(x)
f = lambda x: x**2
print(f)

y = f(x)
print(y)

out = y.sum()
print(out)
# %%
out.backward()
# %%
plt.plot(x.detach(), y.detach())

#%%
plt.plot(x.detach(), x.grad.detach())

#%%