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

#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loadpath = os.path.join(os.getcwd(), 'mnist_ae_z2.pt')
ae = Autoencoder(Encoder, Decoder, 2)
ae.load_state_dict(torch.load(loadpath, map_location=torch.device('cpu')))

plot_ae_outputs(ae, n=10)

loadpath = os.path.join(os.getcwd(), 'mnist_cnn.pt')
classifier = Mnist_CNN((1,28,28), 10)
classifier.load_state_dict(torch.load(loadpath, map_location=torch.device('cpu')))
test_model(classifier)



# %%
encoded_samples = []
for sample in tqdm(test_dataset):
    img = sample[0].unsqueeze(0).to(device)
    label = sample[1]
    ae.eval()
    with torch.no_grad():
        encoded_img = ae.encode(img)
    encoded_img = encoded_img.flatten().cpu().numpy()
    encoded_sample = {f"encoded variable {i}":enc for i, enc in enumerate(encoded_img)}
    encoded_sample['label'] = label
    encoded_samples.append(encoded_sample)
encoded_samples = pd.DataFrame(encoded_samples)
#%%
px.scatter(encoded_samples, x='encoded variable 0', y='encoded variable 1',
           color=encoded_samples.label.astype(str), opacity=0.5)
# %%
# tSNE visualization
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2)
tsne_results = tsne.fit_transform(encoded_samples.drop(['label'],axis=1))
fig = px.scatter(tsne_results, x=0, y=1,
                 color=encoded_samples.label.astype(str),
                 labels={'0': 'tsne-2d-one', '1': 'tsne-2d-two'})
fig.show()
# %%
# Pick a point in latent space and then decode
z = torch.Tensor(([100, 0],))
decoded = ae.decode(z).detach().numpy()

fig, ax = plt.subplots(1,1, figsize=(3,3))
ax.imshow(decoded[0][0], cmap='gray')

#%%
# Compute and shift by gradient

fig, ax = plt.subplots(1, 2, figsize=(6,3))

# Get an original image of 0
targets = train_dataset.targets.numpy()
targets_idx = {i : np.where(targets==i)[0][0] for i in range(10)}
x, _ = train_dataset[targets_idx[0]]
ax[0].imshow(x[0].detach().numpy())
ax[0].set_title('original handwritten image')
x = x.unsqueeze(0)
z = ae.encode(x).detach()
z.requires_grad = True
print('latent variable:', z)

Dz = ae.decode(z)
ax[1].imshow(Dz[0][0].detach().numpy())
ax[1].set_title('AE reconstruction')

prediction_for = 7
classifier_pred = F.sigmoid(classifier(Dz))[:, prediction_for]
print(f"classifier probability of '{str(prediction_for)}': ", classifier_pred)

dfDz_dz = torch.autograd.grad((classifier_pred), z)[0]
print("df(D(z))/dz:", dfDz_dz)

# shift the latent space and visualize
# 0-> 7
lams = np.linspace(1, 100, 10)
fig, ax = plt.subplots(2, 5, figsize=(30, 10))
for i, lam in enumerate(lams):
    shifted_z = z + lam*dfDz_dz
    D_shifted_z = ae.decode(shifted_z)
    ax[i//5, i%5].imshow(D_shifted_z[0][0].detach().numpy())
    fd_shifted_z = F.sigmoid(classifier(D_shifted_z))[:, prediction_for]
    ax[i//5, i%5].set_title(fd_shifted_z.item(), fontsize=20)

#%%
rows = []
for i, lam in enumerate(lams):
    shifted_z = z+lam*dfDz_dz
    D_shifted_z = ae.decode(shifted_z)
    fD_shifted_z = torch.argmax(torch.sigmoid(classifier(D_shifted_z)))
    
    shifted_z_ = shifted_z.detach().flatten().cpu().numpy()
    row = np.append(shifted_z_, "Shifted_"+str(fD_shifted_z.item()))
    rows.append(row)
df = pd.DataFrame(rows, columns=['encoded variable 0', 'encoded variable 1', 'label'])
df


#%%
df2 = pd.concat([encoded_samples, df])
df2.tail(15)

#%%
px.scatter(df2, x='encoded variable 0', y='encoded variable 1',
           color=df2.label.astype(str), opacity=0.5)