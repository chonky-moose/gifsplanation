#%%
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F
import torchvision
import torchvision.transforms as T

from mnist_autoencoder import Encoder, Decoder, Autoencoder, train_epoch, test_epoch, plot_ae_outputs
from mnist_classifier import Mnist_CNN, test_model

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

#%%
# Load the autoencoder ae and test

latent_dim = 4
ae = Autoencoder(Encoder, Decoder, latent_dim).to(device)
loadpath = os.path.join(os.getcwd(), 'mnist_ae.pt')
ae.load_state_dict(torch.load(loadpath))

plot_ae_outputs(ae, n=10)

#%%
# Load the classifier and test
classifier = Mnist_CNN((1,28,28), 10).to(device)
loadpath = os.path.join(os.getcwd(), 'mnist_cnn.pt')
classifier.load_state_dict(torch.load(loadpath))

test_model(classifier)

#%%
def compute_attribution(image, classifier, ae, target):
    image = torch.unsqueeze(image, 0).to(device)
    z = ae.encode(image).detach() # latent representation of image
    z.requires_grad = True
    xp = ae.decode(z) # reconstructed image outputted by autoencoder ae
    
    # pred = how likely does the model think that 'xp' belongs to the 'target' class
    pred = F.sigmoid(classifier(xp))[:, target]
    dzdxp = torch.autograd.grad((pred), z)[0]
    return z, xp, pred, dzdxp

def compute_shift(image, classifier, ae, target, lam):
    z, xp, pred, dzdxp = compute_attribution(image, classifier, ae, target)
    xpp = ae.decode(z+dzdxp*lam).detach()
    pred_xpp = F.sigmoid(classifier(xpp))[:, target]
    return xp, xpp, pred, pred_xpp
# %%
# Get an image of a handwritten 0 (or any other digit)
tf = T.Compose([
    T.ToTensor(),
    T.Normalize((0.1307,), (0.3081,))
])
trainset = torchvision.datasets.MNIST(r'../DATASETS/mnist', train=True,
                                      download=True, transform=tf)
t = trainset.targets.numpy()
t_idx = {i : np.where(t==i)[0][0] for i in range(10)}
print(t, type(t), len(t))
print(t_idx)
print(t_idx[0])

img_handwritten0, _ = trainset[t_idx[0]]
fig, ax = plt.subplots(1,1, figsize=(3,3))
ax.imshow(img_handwritten0[0], cmap='gray')
ax.set_title('Original input image')

xp, xpp, pred, pred_xpp = compute_shift(img_handwritten0, classifier, ae, 8, 150)

fig, ax = plt.subplots(1, 2, figsize=(6, 3))
ax[0].imshow(xp[0][0].detach().cpu().numpy(), cmap='gray')
ax[0].set_title('Reconstructed using autoencoder')
ax[1].imshow(xpp[0][0].detach().cpu().numpy(), cmap='gray')
ax[1].set_title('Reconstructed using shifted z')
print(pred.item(), pred_xpp.item())
# %%
