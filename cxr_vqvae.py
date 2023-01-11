#%%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from antixk_vae import VQVAE

import torch
import torchvision
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# %%
tf = T.Compose([
    T.Grayscale(),
    T.Resize((224, 224)),
    T.ToTensor(),
    # T.Normalize((0.4823,),(0.2363,))
])
train_path = r'C:\Users\lab402\Projects\DATASETS\ucsd_cxr\chest_xray\train'
train_dataset = ImageFolder(train_path, transform=tf)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

test_path = r'C:\Users\lab402\Projects\DATASETS\ucsd_cxr\chest_xray\test'
test_dataset = ImageFolder(test_path, transform=tf)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)


# %%
images, labels = next(iter(train_loader))
print(images.shape, labels)
#%%
vqvae = VQVAE(in_channels=1, embedding_dim=5000, num_embeddings=5000)
vqvae = vqvae.to(device)
recon_images, input_images, vq_loss = vqvae(images.to(device))
#%%
fig, ax = plt.subplots(2, 4, figsize=(8, 4))
for i, img in enumerate(recon_images):
    ax[i//4, i%4].imshow(img[0].detach().cpu().numpy(), cmap='gray')
#%%
print(input_images.shape)
print(vq_loss)

#%%
def vae_train_epoch(epoch, vae, device, dataloader, optimizer):
    vae = vae.to(device)
    vae.train()
    train_loss = []
    for image_batch, _ in tqdm(dataloader):
        image_batch = image_batch.to(device)
        recon_img, input_img, vq_loss = vae(image_batch) # reconstructed image (ie, z decoded to image)
        loss_dict = vae.loss_function(recon_img, input_img, vq_loss)
        loss = loss_dict['loss']
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss.append(loss.detach().cpu().numpy())
    print(f'Epoch: {epoch}, Train loss: {np.mean(train_loss)}')
    return np.mean(train_loss)
# %%
def plot_vae_outputs(vae, dataset, n=2):
    vae.to(device)
    plt.figure(figsize=(8, 5))
    targets = np.array(dataset.targets)
    t_idx = {i : np.where(targets==i)[0][0] for i in range(n)}
    for i in range(n):
        ax = plt.subplot(2,n,i+1)
        img = test_dataset[t_idx[i]][0].unsqueeze(0).to(device)
        
        # Get the reconstructed image
        vae.eval()
        with torch.no_grad():
            recon_img, _, _ = vae(img)
            
        # Plot original image
        plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
        
        ax = plt.subplot(2, n, i+1+n)
        plt.imshow(recon_img.cpu().squeeze().numpy(), cmap='gist_gray')
    plt.show()
# %%
if __name__ == '__main__':    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr = 3e-4
    dimensions = [5000]
    for dimension in dimensions:
        vqvae = VQVAE(in_channels=1, embedding_dim=dimension, num_embeddings=dimension).to(device)
        optim = torch.optim.Adam(vqvae.parameters(), lr=lr)
        
        num_epochs = 20
        current_best_loss = 99999
        losses = {'train loss':[], 'val loss':[]}
        for epoch in range(1, num_epochs+1):
            train_loss = vae_train_epoch(epoch, vqvae, device, train_loader, optim)
            losses['train loss'].append(train_loss)
            if train_loss <= current_best_loss:
                savepath = f'./vqvae_dim_px224_{str(dimension)}_epoch{epoch}.pt'
                torch.save(vqvae.state_dict(), savepath)
                current_best_loss = train_loss
            plot_vae_outputs(vqvae, test_dataset, n=2)
            

# %%
