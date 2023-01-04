#%%
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import plotly.express as px
import pandas as pd

import torch
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F

from mnist_autoencoder import Encoder, Decoder, Autoencoder, plot_ae_outputs
from mnist_classifier import Mnist_CNN, test_model
from mnist_vae import VariationalAutoencoder, plot_vae_outputs


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#%%
data_dir = r'../DATASETS'
tf = T.Compose([
    T.ToTensor(),
    T.Normalize((0.1307,),(0.3081,))
])
train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True,
                                           transform=tf)
test_dataset = torchvision.datasets.MNIST(data_dir, train=False, download=True,
                                          transform=tf)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1,
                                            shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1,
                                          shuffle=True)


#%%
loadpath = os.path.join(os.getcwd(), 'mnist_ae_z2.pt')
ae = Autoencoder(2)
ae.load_state_dict(torch.load(loadpath))
# plot_ae_outputs(ae, n=10) # test that ae loaded properly

loadpath = os.path.join(os.getcwd(), 'mnist_cnn.pt')
classifier = Mnist_CNN((1,28,28), 10)
classifier.load_state_dict(torch.load(loadpath))
# test_model(classifier) # test that classifier loaded properly

vae = VariationalAutoencoder(2)
vae.load_state_dict(torch.load(r'./mnist_vae_z2.pt'))
# plot_vae_outputs(vae) # test that vae loaded properly


# %%
def encode_input(vae, dataset):
    '''
    Return a DataFrame of latent variables.
    
    param dataset: instance of torch.utils.data.Dataset class
    e.g., an instance of torchvision.datasets.mnist.MNIST
    '''
    rows = []
    vae = vae.to(device)
    vae.eval()
    for sample in tqdm(dataset):
        img = sample[0].unsqueeze(0).to(device)
        label = sample[1]
        with torch.no_grad():
            z = vae.encode(img)
        z = z.flatten().cpu().numpy()
        row = {f"z{i}":enc for i, enc in enumerate(z)}
        row['label'] = label
        rows.append(row)
    df = pd.DataFrame(rows)
    return df

def visualize_shifts(latents_df, classifier, vae, original, target, lambdas):
    '''
    Visulize shifted images;
    Return df containing latent variables and shifted latent variables.
    
    param original: input image of shape (b, c, h, w) e.g. (1, 1, 28, 28) for MNIST
    param target: digit we want to shift towards
    param lambdas: array of 10 numbers e.g. np.linspace(1, 30, 10) marking increasing amount of shift
    '''
    vae = vae.to(device)
    vae.eval()
    classifier = classifier.to(device)
    classifier.eval()
    
    fig, ax = plt.subplots(1, 2, figsize=(4,2))
    ax[0].imshow(original[0][0])
    ax[0].set_title('original image')
    
    z = vae.encode(original.to(device)).detach()
    z.requires_grad = True
    print('latent variable z: ', z)
    Dz = vae.decode(z)
    ax[1].imshow(Dz[0][0].detach().cpu().numpy())
    ax[1].set_title('VAE reconstruction')
    
    # classifier prediction for the target digit
    classifier_pred = F.sigmoid(classifier(Dz))[:, target]
    dfDz_dz = torch.autograd.grad((classifier_pred), z)[0]
    
    fig, ax = plt.subplots(2, 5, figsize=(30, 10))
    shifted_zs = []
    for i, lam in enumerate(lambdas):
        zp = z + lam * dfDz_dz
        D_zp = vae.decode(zp)
        ax[i//5, i%5].imshow(D_zp[0][0].detach().cpu().numpy())
        f_D_zp = F.sigmoid(classifier(D_zp))[:, target]
        ax[i//5, i%5].set_title(f_D_zp.item(), fontsize=20)
        
        zp_ = zp.detach().flatten().cpu().numpy()
        row = np.append(zp_, "Shifted")
        shifted_zs.append(row)
    shifted_zs_df = pd.DataFrame(shifted_zs, columns=['z0', 'z1', 'label'])
    df = pd.concat([latents_df, shifted_zs_df])
    df['dot_size'] = [50 if x=='Shifted' else 10 for x in df['label']]
    return df

def visualize_scatter(df):
    return px.scatter(df, x='z0', y='z1', color=df.label.astype(str), size='dot_size',
               opacity=0.5)

#%%
# z_df = encode_input(vae, test_dataset)
#%%
# digit = (1)
# labels = train_dataset.targets.numpy()
# labels_idx = {i : np.where(labels==i)[0][0] for i in range(10)}
# original_img, _ = train_dataset[labels_idx[digit]]
# original_img = original_img.unsqueeze(0)
# df = visualize_shifts(z_df, classifier, vae, original_img, 0, np.linspace(1, 10, 10))
#%%
# visualize_scatter(df)

#%% Run with vanilla autoencoder instead of VAE
# z_df = encode_input(ae, test_dataset)
# digit = (1)
# labels = train_dataset.targets.numpy()
# labels_idx = {i : np.where(labels==i)[0][0] for i in range(10)}
# original_img, _ = train_dataset[labels_idx[digit]]
# original_img = np.expand_dims(original_img, 0)
# df = visualize_shifts(z_df, classifier, ae, original_img, 0, np.linspace(1, 10, 10))

#%%
'''
encoded_samples = []
for sample in tqdm(test_dataset):
    img = sample[0].unsqueeze(0).to(device)
    label = sample[1]
    vae.eval()
    with torch.no_grad():
        encoded_img = vae.encode(img)
    encoded_img = encoded_img.flatten().cpu().numpy()
    encoded_sample = {f"encoded variable {i}":enc for i, enc in enumerate(encoded_img)}
    encoded_sample['label'] = label
    encoded_samples.append(encoded_sample)
encoded_samples = pd.DataFrame(encoded_samples)
encoded_samples
'''
# %%
'''
# Visualization of latent space but using VAE instead of vanilla AE

# Get an original image of a digit
digit = 1
targets = train_dataset.targets.numpy()
targets_idx = {i : np.where(targets==i)[0][0] for i in range(10)}
x, _ = train_dataset[targets_idx[digit]]

fig, ax = plt.subplots(1, 2, figsize=(6,3))
ax[0].imshow(x[0].detach().numpy())
ax[0].set_title('original handwritten image')

x = x.unsqueeze(0).to(device)
z = vae.encode(x).detach()
z.requires_grad = True
print('latent variable:', z)
Dz = vae.decode(z)
ax[1].imshow(Dz[0][0].detach().cpu().numpy())
ax[1].set_title('VAE reconstruction')

prediction_for = 9 # shift towards this digit
classifier_pred = F.sigmoid(classifier(Dz))[:, prediction_for]
print(f"classifier probability of '{str(prediction_for)}': ", classifier_pred)

dfDz_dz = torch.autograd.grad((classifier_pred), z)[0]
print("df(D(z))/dz:", dfDz_dz)

# shift the latent space and visualize
lams = np.linspace(1, 30, 10) # lambdas (how much to shift)
# the interval of linspace is very dependent on how steep the gradient is
# ie, if the gradient is steep, using small lambdas is enough to shift
# whereas if the gradient is small, you need large lambdas to shift the latent
# variable enough so that it enters into the territory of other digits
fig, ax = plt.subplots(2, 5, figsize=(30, 10))
for i, lam in enumerate(lams):
    shifted_z = z + lam*dfDz_dz
    D_shifted_z = vae.decode(shifted_z)
    ax[i//5, i%5].imshow(D_shifted_z[0][0].detach().cpu().numpy())
    fd_shifted_z = F.sigmoid(classifier(D_shifted_z))[:, prediction_for]
    ax[i//5, i%5].set_title(fd_shifted_z.item(), fontsize=20)
'''

#%%
'''
# Add shifted latent variables to DataFrame of latent variables
rows = []
for i, lam in enumerate(lams):
    zp = z+lam*dfDz_dz
    D_zp = vae.decode(zp)
    zp_ = zp.detach().flatten().cpu().numpy()
    row = np.append(zp_, "Shifted")
    rows.append(row)
df = pd.DataFrame(rows, columns=['encoded variable 0', 'encoded variable 1', 'label'])
df2 = pd.concat([encoded_samples, df])
df2['dot_size']  = [50 if x=='Shifted' else 10 for x in df2['label']]
print(df2.tail(15))

# Visualize the latent space
px.scatter(df2, x='encoded variable 0', y='encoded variable 1',
           color=df2.label.astype(str), size='dot_size', opacity=0.5)
'''