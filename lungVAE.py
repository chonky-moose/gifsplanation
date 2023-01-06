#%%
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

import torchvision
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
from torch.distributions import Normal, Independent
from torch.distributions.kl import kl_divergence as KLD
import numpy as np
from torch.nn.functional import softplus, sigmoid, softmax
import pdb
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#%%
class convBlock(nn.Module):
	def __init__(self, inCh, nhid, nOp, pool=True,
					ker=3,padding=1,pooling=2):
		super(convBlock,self).__init__()

		self.enc1 = nn.Conv2d(inCh,nhid,kernel_size=ker,padding=1)
		self.enc2 = nn.Conv2d(nhid,nOp,kernel_size=ker,padding=1)
		self.bn = nn.BatchNorm2d(inCh)	

		if pool:
			self.scale = nn.AvgPool2d(kernel_size=pooling)
		else:
			self.scale = nn.Upsample(scale_factor=pooling)
		self.pool = pool
		self.act = nn.ReLU()

	def forward(self,x):
		x = self.scale(x)
		x = self.bn(x)
		x = self.act(self.enc1(x))
		x = self.act(self.enc2(x))
		return x

class uVAE(nn.Module):
	def __init__(self, nlatent,unet=False, 
					nhid=8, ker=3, inCh=1,h=640,w=512):
		super(uVAE, self).__init__()
		self.latent_space = nlatent
		self.unet = unet

		if not self.unet:
			### VAE Encoder with 4 downsampling operations
			self.enc11 = nn.Conv2d(inCh,nhid,kernel_size=ker,padding=1)
			self.enc12 = nn.Conv2d(nhid,nhid,kernel_size=ker,padding=1)

			self.enc2 = convBlock(nhid,2*nhid,2*nhid,pool=True)
			self.enc3 = convBlock(2*nhid,4*nhid,4*nhid,pool=True)
			self.enc4 = convBlock(4*nhid,8*nhid,8*nhid,pool=True)
			self.enc5 = convBlock(8*nhid,16*nhid,16*nhid,pool=True)

			self.bot11 = nn.Conv1d(16*nhid,1,kernel_size=1)
			self.bot12 = nn.Conv1d(int((h/16)*(w/16)),2*nlatent,kernel_size=1)

			### Decoder with 4 upsampling operations
			self.bot21 = nn.Conv1d(nlatent,int((h/64)*(w/64)),kernel_size=1)
			self.bot22 = nn.Conv1d(1,nhid,kernel_size=1)
			self.bot23 = nn.Conv1d(nhid,4*nhid,kernel_size=1)
			self.bot24 = nn.Conv1d(4*nhid,16*nhid,kernel_size=1)

		### U-net Encoder with 4 downsampling operations
		self.uEnc11 = nn.Conv2d(inCh,nhid,kernel_size=ker,padding=1)
		self.uEnc12 = nn.Conv2d(nhid,nhid,kernel_size=ker,padding=1)

		self.uEnc2 = convBlock(nhid,2*nhid,2*nhid,pool=True,pooling=4)
		self.uEnc3 = convBlock(2*nhid,4*nhid,4*nhid,pool=True,pooling=4)
		self.uEnc4 = convBlock(4*nhid,8*nhid,8*nhid,pool=True)
		self.uEnc5 = convBlock(8*nhid,16*nhid,16*nhid,pool=True)

		### Joint U-Net + VAE decoder 
		if not self.unet:
			self.dec5 = convBlock(32*nhid,8*nhid,8*nhid,pool=False)
		else:
			self.dec5 = convBlock(16*nhid,8*nhid,8*nhid,pool=False)

		self.dec4 = convBlock(16*nhid,4*nhid,4*nhid,pool=False)
		self.dec3 = convBlock(8*nhid,2*nhid,2*nhid,pool=False,pooling=4)
		self.dec2 = convBlock(4*nhid,nhid,nhid,pool=False,pooling=4)

		self.dec11 = nn.Conv2d(2*nhid,nhid,kernel_size=ker,padding=1)
		self.dec12 = nn.Conv2d(nhid,inCh,kernel_size=ker,padding=1)
		
		self.act = nn.ReLU()
		self.mu_0 = torch.zeros((1,nlatent)).to(device)
		self.sigma_0 = torch.ones((1,nlatent)).to(device)

		self.h = h
		self.w = w

	def vae_encoder(self,x):
		### VAE Encoder
		x = self.act(self.enc11(x))
		x = self.act(self.enc12(x))
		x = self.enc2(x)
		x = self.enc3(x)
		x = self.enc4(x)
		x = self.enc5(x)

		z = self.act(self.bot11(x.view(x.shape[0],x.shape[1],-1)))
		z = self.bot12(z.permute(0,2,1))

		return z.squeeze(-1)

	
	def unet_encoder(self,x_in):
		### Unet Encoder
		x = []
		
		x.append(self.act(self.uEnc12(self.act(self.uEnc11(x_in)))))
		x.append(self.uEnc2(x[-1]))
		x.append(self.uEnc3(x[-1]))
		x.append(self.uEnc4(x[-1]))
		x.append(self.uEnc5(x[-1]))

		return x

	def decoder(self,x_enc,z=None):
		if not self.unet:
				### Concatenate latent vector to U-net bottleneck
				x = self.act(self.bot21(z.unsqueeze(2)))
				x = self.act(self.bot22(x.permute(0,2,1)))
				x = self.act(self.bot23(x))
				x = self.act(self.bot24(x))

				x = x.view(x.shape[0],x.shape[1],
						int(self.h/64),int(self.w/64))
				x = torch.cat((x,x_enc[-1]),dim=1)
				x = self.dec5(x)
		else:
				x = self.dec5(x_enc[-1])
		
		### Shared decoder
		x = torch.cat((x,x_enc[-2]),dim=1)
		x = self.dec4(x)
		x = torch.cat((x,x_enc[-3]),dim=1)
		x = self.dec3(x)
		x = torch.cat((x,x_enc[-4]),dim=1)
		x = self.dec2(x)
		x = torch.cat((x,x_enc[-5]),dim=1)

		x = self.act(self.dec11(x))
		x = self.dec12(x)

		return x

	def forward(self, x):
		kl = torch.zeros(1).to(device)
		z = 0.
		# Unet encoder result
		x_enc = self.unet_encoder(x)

		# VAE regularisation
		if not self.unet:
				emb = self.vae_encoder(x)

				# Split encoder outputs into a mean and variance vector
				mu, log_var = torch.chunk(emb, 2, dim=1)

				# Make sure that the log variance is positive
				log_var = softplus(log_var)
				sigma = torch.exp(log_var / 2)
				
				# Instantiate a diagonal Gaussian with mean=mu, std=sigma
				# This is the approximate latent distribution q(z|x)
				posterior = Independent(Normal(loc=mu,scale=sigma),1)
				z = posterior.rsample()

				# Instantiate a standard Gaussian with mean=mu_0, std=sigma_0
				# This is the prior distribution p(z)
				prior = Independent(Normal(loc=self.mu_0,scale=self.sigma_0),1)

				# Estimate the KLD between q(z|x)|| p(z)
				kl = KLD(posterior,prior).sum() 	

		# Outputs for MSE
		xHat = self.decoder(x_enc,z)

		return kl, xHat
#%%
vae = uVAE(nlatent=8, unet=False, nhid=16)
vae.load_state_dict(torch.load(r'./lungVAE.pt'))
vae = vae.to(device)
# %%
# Open an image from UCSD-CXR dataset
# and pass it through lungVAE to get its output (ie the reconstruction)
normal_folder = r'../DATASETS/ucsd_cxr/chest_xray/train/NORMAL/'
normal_jpgs = os.listdir(normal_folder)
img = cv2.imread(os.path.join(normal_folder, normal_jpgs[0]))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (512, 640))
img = torch.Tensor(img)
print(img.shape)
out = vae(img.unsqueeze(0).unsqueeze(0).to(device))
print(out[1].shape)
fig, ax = plt.subplots(1,2, figsize=(10,5))
ax[0].imshow(img.detach().cpu().numpy(), cmap='gray')
ax[0].set_title('original image')
ax[1].imshow(out[1][0][0].detach().cpu().numpy(), cmap='gray')
ax[1].set_title('reconstructed with lungVAE')

#%%
# Train VAE with ucsd_cxr data
tf = T.Compose([
    T.Grayscale(),
    T.Resize((640, 512)),
    T.ToTensor(),
    T.Normalize((0.4823,),(0.2363,))
])
train_path = r'C:\Users\lab402\Projects\DATASETS\ucsd_cxr\chest_xray\train'
train_dataset = ImageFolder(train_path, transform=tf)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

test_path = r'C:\Users\lab402\Projects\DATASETS\ucsd_cxr\chest_xray\test'
test_dataset = ImageFolder(test_path, transform=tf)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

# %%
def vae_train_epoch(epoch, vae, device, dataloader, optimizer):
    vae = vae.to(device)
    vae.train()
    train_loss = []
    for image_batch, _ in tqdm(dataloader):
        image_batch = image_batch.to(device)
        kl, xhat = vae(image_batch) # reconstructed image (ie, z decoded to image)
        loss = ((image_batch - xhat)**2).sum() + kl # for some reason, doesn't work with nn.MSELoss()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss.append(loss.detach().cpu().numpy())
    print(f'Epoch: {epoch}, Train loss: {np.mean(train_loss)}')
    return np.mean(train_loss)
#%%
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
            _, rec_img = vae(img)
            
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr = 0.001
    myuvae = uVAE(nlatent=8, unet=False, nhid=16).to(device)
    optim = torch.optim.Adam(myuvae.parameters(), lr=lr)
    
    num_epochs = 50
    losses = {'train loss':[], 'val loss':[]}
    for epoch in range(1, num_epochs+1):
        train_loss = vae_train_epoch(epoch, myuvae, device, train_loader, optim)
        losses['train loss'].append(train_loss)
        plot_vae_outputs(myuvae, test_dataset, n=2)


# %%
savepath = r'./myuvae.pt'
torch.save(myuvae.state_dict(), savepath)
#%%
m = uVAE(nlatent=8, unet=False, nhid=16)
m.load_state_dict(torch.load(r'./myuvae.pt'))
# %%
m = m.to(device)
imgs, _ = next(iter(train_loader))
_, rec_imgs = m(imgs.to(device))
print(imgs.shape, rec_imgs.shape)
fig, ax = plt.subplots(4, 4, figsize=(8,8))
for i, (img, rec) in enumerate(zip(imgs, rec_imgs)):
    ax[i//4, i%4].imshow(img[0].detach().cpu().numpy(), cmap='gray')
    ax[2 + i//4, i%4].imshow(rec[0].detach().cpu().numpy(), cmap='gray')




# %%
img = cv2.imread(r'../DATASETS/ucsd_cxr/chest_xray/val/NORMAL/NORMAL2-IM-1427-0001.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (512, 640))
img = torch.Tensor(img)
# %%
fig, ax = plt.subplots(1,1, figsize=(4,4))
ax.imshow(img.detach().cpu().numpy(), cmap='gray')

#%%
_, rec = m(img.unsqueeze(0).unsqueeze(0).to(device))
# %%
fig, ax = plt.subplots(1,1, figsize=(4,4))
ax.imshow(rec[0][0].detach().cpu().numpy(), cmap='gray')
#%%
enc0 = torch.randn((8, 16, 640, 512)).to(device)
enc1 = torch.randn((8, 32, 160, 128)).to(device)
enc2 = torch.randn((8, 64, 40, 32)).to(device)
enc3 = torch.randn((8, 128, 20, 16)).to(device)
enc4 = torch.randn((8, 256, 10, 8)).to(device)
x_enc = [enc0, enc1, enc2, enc3, enc4]
# %%
x = torch.randn((8, 1, 640, 512)).to(device)
emb = m.vae_encoder(x)
print('emb.shape:', emb.shape)
mu, log_var = torch.chunk(emb, 2, dim=1)
print('mu.shape', mu.shape, 'log_var.shape', log_var.shape)
log_var = softplus(log_var)
sigma = torch.exp(log_var/2)

posterior = Independent(Normal(loc=mu, scale=sigma), 1)
z = posterior.rsample()
print('z.shape:', z.shape)

# nlatent = 8
# prior = Independent(Normal(loc=torch.zeros((1,nlatent)).to(device),
#                            scale=torch.ones((1,nlatent)).to(device)),1)
# %%
z = z.to(device)
out = m.decoder(x_enc, z)
print(out.shape)
fig, ax = plt.subplots(2, 4, figsize=(8, 4))
for i, img in enumerate(out):
    ax[i//4, i%4].imshow(img[0].detach().cpu().numpy(), cmap='gray')
    
    
#%%
images, labels = next(iter(train_loader))
x_enc = m.unet_encoder(images.to(device))
# %%
for i in range(0, 4+1):
    print(i, x_enc[i].shape)
    print(x_enc[i].max(), x_enc[i].min())
    print()


# %%
enc0 = torch.randn((8, 16, 640, 512)).to(device)
enc1 = torch.randn((8, 32, 160, 128)).to(device)
enc2 = torch.randn((8, 64, 40, 32)).to(device)
enc3 = torch.randn((8, 128, 20, 16)).to(device)
enc4 = torch.randn((8, 256, 10, 8)).to(device)
x_enc = [enc0, enc1, enc2, enc3, enc4]
for i in range(0, 4+1):
    print(x_enc[i].max(), x_enc[i].min())
#%%
enc0 = torch.abs(torch.randn((8, 16, 640, 512))).to(device)
enc1 = torch.abs(torch.randn((8, 32, 160, 128))).to(device)
enc2 = torch.abs(torch.randn((8, 64, 40, 32))).to(device)
enc3 = torch.abs(torch.randn((8, 128, 20, 16))).to(device)
enc4 = torch.abs(torch.randn((8, 256, 10, 8))).to(device)
x_enc = [enc0, enc1, enc2, enc3, enc4]
for i in range(0, 4+1):
    print(x_enc[i].max(), x_enc[i].min())
    
#%%
z = z.to(device)
out = m.decoder(x_enc, z)
print(out.shape)
fig, ax = plt.subplots(2, 4, figsize=(8, 4))
for i, img in enumerate(out):
    ax[i//4, i%4].imshow(img[0].detach().cpu().numpy(), cmap='gray')