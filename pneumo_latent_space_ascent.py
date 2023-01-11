#%%
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

import timm
from antixk_vae import VQVAE

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as T

import torchxrayvision as xrv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%%
# Visualize some examples
tf = T.Compose([T.Resize((64, 64)),
                T.Grayscale(num_output_channels=1),
                T.ToTensor()])
path = r'C:\Users\lab402\Projects\DATASETS\ucsd_cxr\chest_xray\train'
dataset = ImageFolder(path, transform=tf)
dataloader = DataLoader(dataset, batch_size=9, shuffle=True)

images, labels = next(iter(dataloader))
print(images.shape, labels.shape)
fig, ax = plt.subplots(3,3, figsize=(6,6))
for i, img in enumerate(images):
    ax[i//3, i%3].imshow(img[0], cmap='gray')
    ax[i//3, i%3].set_title(labels[i].item())

#%%
model = timm.create_model('tf_efficientnet_b4_ns', pretrained=True)
for param in model.parameters():
    param.requires_grad = False

model.conv_stem = nn.Sequential(
    nn.Conv2d(1, 48, kernel_size=(3,3), stride=(2,2), bias=False)
)
model.classifier = nn.Sequential(
    nn.Linear(in_features=1792, out_features=625),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(in_features=625, out_features=256),
    nn.ReLU(),
    nn.Linear(in_features=256, out_features=2)
)
model.load_state_dict(torch.load(r'./ucsdcxr_ch1_classifier.pt'))
# %%
model.eval()
output = model(images)
print(output)
print(torch.argmax(output, dim=1))
print(labels)


#%%
vqvae = VQVAE(in_channels=1, embedding_dim=5000, num_embeddings=5000)
vqvae.load_state_dict(torch.load(r'../gifsplanation_large_files/vqvae_px256_dim_5000_epoch30.pt'))







#%%
#######################################################################
#######################################################################
#######################################################################
#######################################################################

#%%
# torchxrayvision's model
model = xrv.models.DenseNet(weights='densenet121-res224-rsna')
model.eval()
#%%
help(model)
#%%
output = model(images)
print(torch.argmax(output, dim=1))
print(labels)


# %%
vqvae = VQVAE(in_channels=1, embedding_dim=5000, num_embeddings=5000)
vqvae.load_state_dict(torch.load(r'../gifsplanation_large_files/vqvae_px256_dim_5000_epoch30.pt'))
# %%
z = vqvae.encode(images)