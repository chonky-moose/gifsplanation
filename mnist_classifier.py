#%%
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

tf = T.Compose([
    T.ToTensor(),
    T.Normalize((0.1307,), (0.3081,))
])
trainset = torchvision.datasets.MNIST(r'../DATASETS/mnist', train=True,
                                    download=True, transform=tf)
valset = torchvision.datasets.MNIST(r'../DATASETS/mnist', train=False,
                                    download=True, transform=tf)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# %%
class Mnist_CNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_size[0], 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc1 = nn.Linear(4*4*64, num_classes)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        return x


#%%
def main():

    
    model = Mnist_CNN((1,28,28), 10)
    
    lr = 1e-3
    epochs = 50
    batch_size = 64
    optimizer = torch.optim.Adam(model.parameters(), lr)
    criterion = torch.nn.CrossEntropyLoss()
    train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                            batch_size=batch_size,
                                            shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=valset,
                                            batch_size=batch_size,
                                            shuffle=True)
    print(device, len(train_loader), len(val_loader))


    model.to(device)
    for epoch in range(1, epochs+1):
        train_loss = []
        for i, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        
        test_loss, test_accuracy = [], []
        for i, (images, labels) in tqdm(enumerate(val_loader), total=len(val_loader)):
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            _, predicted = torch.max(preds.data, 1)
            loss = criterion(preds, labels)
            test_loss.append(loss.item())
            test_accuracy.append((predicted==labels).sum().item() / predicted.size(0))
        print(f'epoch: {epoch}, train loss:{np.mean(train_loss)},\
    test loss: {np.mean(test_loss)}, test accuracy: {np.mean(test_accuracy)}')
# %%
# Testing the model
def test_model(model):
    model.to(device)
    loader = torch.utils.data.DataLoader(dataset=valset,
                                        batch_size=16,
                                        shuffle=True)
    xs, ys = next(iter(loader))
    xs, ys = xs.to(device), ys.to(device)
    outputs = model(xs)
    _, yhats = torch.max(outputs.data, 1)

    fig, ax = plt.subplots(4,4, figsize=(8,8))
    for i, x in enumerate(xs):
        x = x.cpu().numpy()
        ax[i//4, i%4].imshow(x[0], cmap='gray')
        ax[i//4, i%4].axis('off')
        ax[i//4, i%4].set_title(f'yhat:{yhats[i].item()}, y:{ys[i].item()}')

#%%
if __name__ == '__main__':
    main()


# %%
# # Save the model
# savepath = r'./mnist_cnn.pt'
# torch.save(model.state_dict(), savepath)

#%%
# # Load and test the model
# loadpath = os.path.join(os.getcwd(), 'mnist_cnn.pt')
# m = Mnist_CNN((1,28,28), 10)
# m.load_state_dict(torch.load(loadpath))

# test_model(m)
