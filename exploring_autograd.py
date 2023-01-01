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
v = x + 2
y = v ** 2
print('y', y)
dydx = torch.autograd.functional.jacobian(func=y, inputs=x)
print('dydx', dydx)
# Compute loss
loss = torch.nn.MSELoss()(y, gt)
print('Loss', loss)

# Now compute gradients
dloss_dx = torch.autograd.grad(outputs=loss, inputs=x)
print('dloss/dx:', dloss_dx)