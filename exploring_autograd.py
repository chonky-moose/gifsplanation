
# References
# https://stackoverflow.com/questions/54754153/autograd-grad-for-tensor-in-pytorch

#%%
# What does it mean to take a gradient?
# Let's explore torch.autograd.grad
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

dydx = torch.autograd.functional.jacobian(func=lambda var: (var+2)**2, inputs=x)
print('dydx', dydx.shape)
print(dydx)

# Compute loss
loss = torch.nn.MSELoss()(y, gt)
print('Loss', loss)

# Now compute gradients
dloss_dx = torch.autograd.grad(outputs=loss, inputs=x)
print('dloss/dx:', dloss_dx)
# %%
