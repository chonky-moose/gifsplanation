#%%
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

from mnist_autoencoder import Autoencoder

data_dir = r'../DATASETS/'
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
class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
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
        self.encoder_lin1 = nn.Sequential(
            nn.Linear(3*3*32, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dims)
        )
        self.encoder_lin2 = nn.Sequential(
            nn.Linear(3*3*32, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dims)
        )
        
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
                
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        mu = self.encoder_lin1(x)
        sigma = torch.exp(self.encoder_lin2(x))
        z = mu + sigma * self.N.sample(mu.shape)
        
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dims, 128),
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
class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)
        
    def encode(self, x):
        z = self.encoder(x)
        return z
    
    def decode(self, z):
        decoded = self.decoder(z)
        return decoded
    
    def forward(self, x):
        ret = {}
        ret["z"] = z = self.encoder(x)
        ret["out"] = self.decoder(z)
        return ret


# %%
def vae_train_epoch(epoch, vae, device, dataloader, loss_fn, optimizer):
    vae = vae.to(device)
    vae.train()
    train_loss = []
    for image_batch, _ in tqdm(dataloader):
        image_batch = image_batch.to(device)
        decoded = vae(image_batch)['out']
        loss = ((image_batch - decoded)**2).sum() + vae.encoder.kl
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss.append(loss.detach().cpu().numpy())
    print(f'Epoch: {epoch}, Train loss: {np.mean(train_loss)}')
    return np.mean(train_loss)

# def test_epoch(epoch, vae, device, dataloader, loss_fn):
#     vae.eval()
#     with torch.no_grad():
#         conc_out = []
#         conc_label = []
#         for image_batch, _ in tqdm(dataloader):
#             image_batch = image_batch.to(device)
#             decoded = vae(image_batch)['out']
#             conc_out.append(decoded.cpu())
#             conc_label.append(image_batch.cpu())
#         conc_out = torch.cat(conc_out)
#         conc_label = torch.cat(conc_label)
        
#         val_loss = loss_fn(conc_out, conc_label) + vae.encoder.kl
        
#     print(f'Epoch: {epoch}, Validation loss: {val_loss.item()}')
#     return val_loss.data

def plot_vae_outputs(vae, n=10):
    vae.to(device)
    plt.figure(figsize=(16, 4.5))
    targets = test_dataset.targets.numpy()
    t_idx = {i : np.where(targets==i)[0][0] for i in range(n)}
    for i in range(n):
        ax = plt.subplot(2,n,i+1)
        img = test_dataset[t_idx[i]][0].unsqueeze(0).to(device)
        vae.eval()
        
        # Get the reconstructed image
        with torch.no_grad():
            rec_img = vae(img)['out']
            
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
        if i ==n//2:
            ax.set_title("Reconstructed images")
    plt.show()
# %%
if __name__ == '__main__':    
    # loss_fn = torch.nn.MSELoss()
    lr = 0.001
    d = 2 # latent space dimensions
    vae = VariationalAutoencoder(d).to(device)
    optim = torch.optim.Adam(vae.parameters(), lr=lr)
    
    num_epochs = 15
    losses = {'train loss':[], 'val loss':[]}
    for epoch in range(1, num_epochs+1):
        train_loss = vae_train_epoch(epoch, vae, device, train_loader,
                                     loss_fn=None, optimizer=optim)
        losses['train loss'].append(train_loss)
        plot_vae_outputs(vae, n=10)
# %%
# Save VAE
# savepath = r'./mnist_vae_z2.pt'
# torch.save(vae.state_dict(), savepath)