#%%
import torchxrayvision as xrv

import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# %%
ae = xrv.autoencoders.ResNetAE(weights='101-elastic')
ae = ae.to(device)
# %%
tf = T.Compose([
    T.Grayscale(),
    # T.Resize((640, 512)),
    T.Resize((256,256)),
    T.ToTensor(),
    T.Normalize((0.4823,),(0.2363,))
])
train_path = r'C:\Users\lab402\Projects\DATASETS\ucsd_cxr\chest_xray\train'
train_dataset = ImageFolder(train_path, transform=tf)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

test_path = r'C:\Users\lab402\Projects\DATASETS\ucsd_cxr\chest_xray\test'
test_dataset = ImageFolder(test_path, transform=tf)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
# %%
help(ae)
# %%
images, labels = next(iter(train_loader))
print(images.shape, labels)
# %%
z = ae.encode(images.to(device))
# %%
xhat = ae.decode(z.to(device))

#%%
fig, ax = plt.subplots(2,4, figsize=(8,4))
for i, img in enumerate(xhat):
    ax[i//4, i%4].imshow(img[0].detach().cpu().numpy(), cmap='gray')
# %%
