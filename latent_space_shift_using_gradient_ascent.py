#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import plotly.express as px

import torch
import torchvision
import torchvision.transforms as T

from mnist_vae import VariationalAutoencoder
from mnist_classifier import Mnist_CNN
from latent_space_visualization import encode_input, visualize_shifts, visualize_scatter
# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vae = VariationalAutoencoder(latent_dims=2)
vae.load_state_dict(torch.load(r'./mnist_vae_z2.pt'))
vae = vae.to(device)
vae.eval()

cnn = Mnist_CNN((1,28,28), 10)
cnn.load_state_dict(torch.load(r'./mnist_cnn.pt'))
cnn = cnn.to(device)
cnn.eval()
# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_dir = r'../DATASETS'
tf = T.Compose([
    T.ToTensor(),
    T.Normalize((0.1307,),(0.3081,))
])
train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True,
                                           transform=tf)
test_dataset = torchvision.datasets.MNIST(data_dir, train=False, download=True,
                                          transform=tf)

# %%
z_df = encode_input(vae, test_dataset)
z_df
#%%
px.scatter(z_df, x='z0', y='z1', color=z_df.label.astype(str))


#%%
n_iters = 10
step_size = 0.5

starting_digit, target_class = 1, 0
labels = train_dataset.targets.numpy()
labels_idx = {i : np.where(labels==i)[0][0] for i in range(10)}
x, _ = train_dataset[labels_idx[starting_digit]]
x = x.unsqueeze(0) # first x
x.requires_grad = True
x.retain_grad()
z = vae.encode(x.to(device)) # first z
z.retain_grad()

zs = [z.detach().cpu().numpy()]
fig, ax = plt.subplots(2, 5, figsize=(10, 4))
for i in range(n_iters):
    Dz = vae.decode(z)
    y = cnn(Dz.to(device))[0][target_class]
    y.backward(retain_graph=True)
    
    z = z + step_size*z.grad # update z with gradient
    z.retain_grad()
    zs.append(z.detach().cpu().numpy())
    
    ax[i//5, i%5].imshow(Dz[0][0].detach().cpu().numpy(), cmap='gray')
# %%
# transform zs into a DataFrame
rows = [list(z[0])+["Shifted"] for z in zs]
shifted_df = pd.DataFrame(rows, columns=['z0', 'z1', 'label'])

df = pd.concat([z_df, shifted_df])
df['dot_size'] = [50 if x=='Shifted' else 10 for x in df['label']]
df

#%%
px.scatter(df, x='z0', y='z1', color=df.label.astype(str), size='dot_size',
            opacity=0.5)
# %%
