#%%
# Modified but heavily copied from
# https://medium.com/dataseries/convolutional-autoencoder-in-pytorch-on-mnist-dataset-d65145c132ac

#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
# %%
data_dir = r'../DATASETS/mnist'
tf = T.Compose([
    T.ToTensor(),
    T.Normalize((0.1307,), (0.3081,))
])
train_dataset = torchvision.datasets.MNIST(data_dir, train=True,
                                           download=True, transform=tf)
test_dataset = torchvision.datasets.MNIST(data_dir, train=False,
                                          download=True, transform=tf)

batch_size = 128
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
class Encoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1,8,3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=0), 
            nn.ReLU()
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.encoder_lin = nn.Sequential(
            nn.Linear(3*3*32, 128),
            nn.ReLU(),
            nn.Linear(128, encoded_space_dim)
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x
# %%
class Decoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3*3*32),
            nn.ReLU()
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32,3,3))
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x

# %%
####################################################################
# Autoencoder class similar to torchxrayvision's _ResNetAE class
# ie have encode, decode as methods and
# and return a dictionary of latent variable 'z' and decoded output 'out'
class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder, latent_space_dim):
        super().__init__()
        self.encoder = encoder(latent_space_dim)
        self.decoder = decoder(latent_space_dim)
        
    def encode(self, x):
        z = self.encoder(x)
        return z

    def decode(self, z):
        decoded = self.decoder(z)
        return decoded
        
    def forward(self, x):
        ret = {}
        ret["z"] = z = self.encoder(x)
        ret['out'] = self.decoder(z)
        return ret
# %%
# Train an instance of the Autoencoder class


def train_epoch(epoch, ae, device, dataloader, loss_fn, optimizer):
    ae.train()
    train_loss = []
    for image_batch, _ in tqdm(dataloader):
        image_batch = image_batch.to(device)
        decoded = ae(image_batch)['out']
        loss = loss_fn(decoded, image_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss.append(loss.detach().cpu().numpy())
    print(f'Epoch: {epoch}, Train loss: {np.mean(train_loss)}')
    return np.mean(train_loss)

def test_epoch(epoch, ae, device, dataloader, loss_fn):
    ae.eval()
    with torch.no_grad():
        conc_out = []
        conc_label = []
        for image_batch, _ in tqdm(dataloader):
            image_batch = image_batch.to(device)
            decoded = ae(image_batch)['out']
            conc_out.append(decoded.cpu())
            conc_label.append(image_batch.cpu())
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label)
        
        val_loss = loss_fn(conc_out, conc_label)

    print(f'Epoch: {epoch}, Validation loss: {val_loss.item()}')    
    return val_loss.data

def plot_ae_outputs(ae, n=10):
    plt.figure(figsize=(16, 4.5))
    targets = test_dataset.targets.numpy()
    t_idx = {i : np.where(targets==i)[0][0] for i in range(n)}
    for i in range(n):
        ax = plt.subplot(2, n, i+1)
        img = test_dataset[t_idx[i]][0].unsqueeze(0).to(device)
        
        ae.eval()
        # Get the reconstructed image
        with torch.no_grad():
            rec_img = ae(img)['out']
        
        # Plot original image
        plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n//2:
            ax.set_title('Original images')
        
        ax = plt.subplot(2, n, i+1+n)
        plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n//2:
            ax.set_title('Reconstructed images')
    plt.show()

# %%
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = torch.nn.MSELoss()
    lr = 0.001
    d = 2 # latent space dimensions
    ae = Autoencoder(Encoder, Decoder, d).to(device)
    optim = torch.optim.Adam(ae.parameters(), lr=lr)
    
    num_epochs = 20
    losses = {'train loss':[], 'val loss':[]}
    for epoch in range(1, num_epochs+1):
        train_loss = train_epoch(epoch, ae, device, train_loader, loss_fn, optim)
        val_loss = test_epoch(epoch, ae, device, test_loader, loss_fn)
        losses['train loss'].append(train_loss)
        losses['val loss'].append(val_loss)
        plot_ae_outputs(ae,n=10)
# %%
# Save model
savepath = r'./mnist_ae_z2.pt'
torch.save(ae.state_dict(), savepath)