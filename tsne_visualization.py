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


#%%
loadpath = os.path.join(os.getcwd(), 'mnist_ae_z2.pt')
ae = Autoencoder(Encoder, Decoder, 2)
ae.load_state_dict(torch.load(loadpath))

plot_ae_outputs(ae, n=10)

loadpath = os.path.join(os.getcwd(), 'mnist_cnn.pt')
classifier = Mnist_CNN((1,28,28), 10)
classifier.load_state_dict(torch.load(loadpath))
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
z = z.to(device)
decoded = ae.decode(z).detach().cpu().numpy()

fig, ax = plt.subplots(1,1, figsize=(3,3))
ax.imshow(decoded[0][0], cmap='gray')




#%%
#################################
### Visualize shifted images ###
##################################

# Get an original image of 7
targets = train_dataset.targets.numpy()
targets_idx = {i : np.where(targets==i)[0][0] for i in range(10)}
x, _ = train_dataset[targets_idx[7]]

fig, ax = plt.subplots(1, 2, figsize=(6,3))
ax[0].imshow(x[0].detach().numpy())
ax[0].set_title('original handwritten image')
x = x.unsqueeze(0).to(device)
z = ae.encode(x).detach()
z.requires_grad = True
print('latent variable:', z)
Dz = ae.decode(z)
ax[1].imshow(Dz[0][0].detach().cpu().numpy())
ax[1].set_title('AE reconstruction')

prediction_for = 0
classifier_pred = F.sigmoid(classifier(Dz))[:, prediction_for]
print(f"classifier probability of '{str(prediction_for)}': ", classifier_pred)

dfDz_dz = torch.autograd.grad((classifier_pred), z)[0]
print("df(D(z))/dz:", dfDz_dz)

# shift the latent space and visualize
lams = np.linspace(1, 5000, 10) # lambdas (how much to shift)
# the interval of linspace is very dependent on how steep the gradient is
# ie, if the gradient is steep, using small lambdas is enough to shift
# whereas if the gradient is small, you need large lambdas to shift the latent
# variable enough so that it enters into the territory of other digits
fig, ax = plt.subplots(2, 5, figsize=(30, 10))
for i, lam in enumerate(lams):
    shifted_z = z + lam*dfDz_dz
    D_shifted_z = ae.decode(shifted_z)
    ax[i//5, i%5].imshow(D_shifted_z[0][0].detach().cpu().numpy())
    fd_shifted_z = F.sigmoid(classifier(D_shifted_z))[:, prediction_for]
    ax[i//5, i%5].set_title(fd_shifted_z.item(), fontsize=20)

#%%
# Making a dataframe with rows for shifted latent variables
# for px.scatter visualization
rows = []
for i, lam in enumerate(lams):
    shifted_z = z+lam*dfDz_dz
    D_shifted_z = ae.decode(shifted_z)
    fD_shifted_z = torch.argmax(torch.sigmoid(classifier(D_shifted_z)))
    
    shifted_z_ = shifted_z.detach().flatten().cpu().numpy()
    row = np.append(shifted_z_, "Shifted")
    rows.append(row)
df = pd.DataFrame(rows, columns=['encoded variable 0', 'encoded variable 1', 'label'])
df2 = pd.concat([encoded_samples, df])
df2['dot_size']  = [50 if x=='Shifted' else 10 for x in df2['label']]

# Visualize
px.scatter(df2, x='encoded variable 0', y='encoded variable 1',
           color=df2.label.astype(str), size='dot_size', opacity=0.5)
