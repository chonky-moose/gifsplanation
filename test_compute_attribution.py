#%%
import matplotlib as mpl
def full_frame(width=None, height=None):
    mpl.rcParams['savefig.pad_inches'] = 0
    figsize = None if width is None else (width, height)
    fig = plt.figure(figsize=figsize)
    ax = plt.axes([0,0,1,1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.autoscale(tight=True)
    
def thresholdf(x, percentile):
    return x * (x > np.percentile(x, percentile))

#%%
import sys
sys.path
# sys.path.append(r'C:\Users\lab402\anaconda3\envs\gifsplanation\Lib\site-packages')

#%%
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
import glob
import numpy as np
import skimage, skimage.filters
import sklearn, sklearn.metrics
# import captum, captum.attr
import torch, torch.nn
import pickle
from PIL import ImageDraw
import pandas as pd
import shutil
import os,sys
import torchvision
import torchxrayvision as xrv

#%%
def compute_attribution(image, method, clf, target, plot=False, ret_params=False,
                        fixrange=None, p=0.0, ae=None, sigma=0, threshold=False):
    '''
    param image: an xray image we want to assess
    param method: choose among ['grad', 'guided', 'integrated', 'latentshift']
    param clf: the classifier model (ie the neural network)
    param ae: autoencoder
    '''
    image = image.clone().detach()
    image_shape = image.shape[-2:]
    def clean(saliency):
        saliency = np.abs(saliency)
        if sigma > 0:
            saliency = skimage.filters.gaussian(saliency, 
                        mode='constant', 
                        sigma=(sigma, sigma), 
                        truncate=3.5)
        if threshold != False:
            saliency = thresholdf(saliency, 95 if threshold == True else threshold)
        return saliency
    
    if "latentshift" in method:
        z = ae.encode(image).detach()
        z.requires_grad = True
        xp = ae.decode(z, image_shape)
        print(F.sigmoid(clf((image*p + xp*(1-p)))))
        pred = F.sigmoid(clf((image*p + xp*(1-p))))[:,clf.pathologies.index(target)]
        print(pred)
        print(pred.shape)
        dzdxp = torch.autograd.grad((pred), z)[0]
        
        cache = {}
        def compute_shift(lam):
            #print(lam)
            if lam not in cache:
                xpp = ae.decode(z+dzdxp*lam, image_shape).detach()
                pred1 = F.sigmoid(clf((image*p + xpp*(1-p))))[:,clf.pathologies.index(target)].detach().cpu().numpy()
                cache[lam] = xpp, pred1
            return cache[lam]
        
        #determine range
        #initial_pred = pred.detach().cpu().numpy()
        _, initial_pred = compute_shift(0)
        
        
        if fixrange:
            lbound,rbound = fixrange
        else:
            #search params
            step = 10

            #left range
            lbound = 0
            last_pred = initial_pred
            while True:
                xpp, cur_pred = compute_shift(lbound)
                #print("lbound",lbound, "last_pred",last_pred, "cur_pred",cur_pred)
                if last_pred < cur_pred:
                    break
                if initial_pred-0.15 > cur_pred:
                    break
                if lbound <= -1000:
                    break
                last_pred = cur_pred
                if np.abs(lbound) < step:
                    lbound = lbound - 1
                else:
                    lbound = lbound - step

            #right range
            rbound = 0
#             last_pred = initial_pred
#             while True:
#                 xpp, cur_pred = compute_shift(rbound)
#                 #print("rbound",rbound, "last_pred",last_pred, "cur_pred",cur_pred)
#                 if last_pred > cur_pred:
#                     break
#                 if initial_pred+0.05 < cur_pred:
#                     break
#                 if rbound >= 1000:
#                     break
#                 last_pred = cur_pred
#                 if np.abs(rbound) < step:
#                     rbound = rbound + 1
#                 else:
#                     rbound = rbound + step
        
        print(initial_pred, lbound,rbound)
        #lambdas = np.arange(lbound,rbound,(rbound+np.abs(lbound))//10)
        lambdas = np.arange(lbound,rbound,np.abs((lbound-rbound)/10))
        ###########################
        
        y = []
        dimgs = []
        xp = ae.decode(z,image_shape)[0][0].unsqueeze(0).unsqueeze(0).detach()
        for lam in lambdas:
            
            xpp, pred = compute_shift(lam)
            dimgs.append(xpp.cpu().numpy())
            y.append(pred)
            
        if ret_params:
            params = {}
            params["dimgs"] = dimgs
            params["lambdas"] = lambdas
            params["y"] = y
            params["initial_pred"] = initial_pred
            return params
        
        if plot:
            
            px = 1/plt.rcParams['figure.dpi']
            full_frame(image[0][0].shape[0]*px,image[0][0].shape[1]*px)
            plt.imshow(image.detach().cpu()[0][0], interpolation='none', cmap="gray")
            plt.title("image")
            plt.show()
            px = 1/plt.rcParams['figure.dpi']
            full_frame(xp[0][0].shape[0]*px,xp[0][0].shape[1]*px)
            plt.imshow(xp.detach().cpu()[0][0], interpolation='none', cmap="gray")
            plt.title("image_recon")
            plt.show()
            
            plt.plot(lambdas,y)
            plt.xlabel("lambda shift");
            plt.ylabel("Prediction of " + target);
            plt.show()
        
        if "-max" in method:
            dimage = np.max(np.abs(xp.cpu().numpy()[0][0] - dimgs[0][0]),0)
        elif "-mean" in method:
            dimage = np.mean(np.abs(xp.cpu().numpy()[0][0] - dimgs[0][0]),0)
        elif "-mm" in method:
            dimage = np.abs(dimgs[0][0][0] - dimgs[-1][0][0])
        elif "-int" in method:
            dimages = []
            for i in range(len(dimgs)-1):
                dimages.append(np.abs(dimgs[i][0][0] - dimgs[i+1][0][0]))
            dimage = np.mean(dimages,0)
        else:
            raise Exception("Unknown mode")
        
        dimage = clean(dimage)
        return dimage
    
    # if method == "grad":
    #     image.requires_grad = True
    #     pred = clf(image)[:,clf.pathologies.index(target)]
    #     dimage = torch.autograd.grad(torch.abs(pred), image)[0]
    #     dimage = dimage.detach().cpu().numpy()[0][0]
    #     dimage = clean(dimage)
    #     return dimage
    
    # if method == "integrated":
    #     attr = captum.attr.IntegratedGradients(clf)
    #     dimage = attr.attribute(image, 
    #                             target=clf.pathologies.index(target),
    #                             n_steps=100, 
    #                             return_convergence_delta=False, 
    #                             internal_batch_size=1)
    #     dimage = dimage.detach().cpu().numpy()[0][0]
    #     dimage = clean(dimage)
    #     return dimage
    
    # if method == "guided":
        
    #     attr = captum.attr.GuidedBackprop(clf)
    #     dimage = attr.attribute(image, target=clf.pathologies.index(target))
    #     dimage = dimage.detach().cpu().numpy()[0][0]
    #     dimage = clean(dimage)
    #     return dimage
    
    # if method == "iterativedelete":
        
    #     lr = 1
    #     grads = []
    #     for i in range(20):
    #         image.requires_grad = True
    #         pred = clf(image)[:,clf.pathologies.index(target)]
    #         #print(pred)
    #         dimage = torch.autograd.grad(torch.abs(pred), image)[0]
    #         dimage = dimage.detach().cpu().numpy()[0][0]
    #         grads.append(dimage)
            
    #         dimage = thresholdf(dimage, 98)
    #         #print(image.shape, dimage.shape)
    #         image = image * torch.Tensor(dimage>0).cuda().unsqueeze(0).unsqueeze(0)
    #         image = image.clone().detach()
            
    #     dimage = np.mean(grads,0)
    #     dimage = clean(dimage)
    #     return dimage
    
#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ae = xrv.autoencoders.ResNetAE(weights='101-elastic').to(device)
ae
#%%
img = skimage.io.imread('pneumonia.jpeg')
img = xrv.datasets.normalize(img, 255)
img = img[None, :, :]

transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                            xrv.datasets.XRayResizer(224)])
img = transform(img)
img = image = torch.from_numpy(img).unsqueeze(0)
img.shape
#%%
model = xrv.models.DenseNet(weights='densenet121-res224-rsna').to(device)
model
# %%
params = compute_attribution(img.to(device), method='latentshift', clf=model,
                             target='Pneumonia', ret_params=True,
                             ae=ae, fixrange=None)
# %%
dimgs = np.concatenate(params['dimgs'], 1)[0]
fig, ax = plt.subplots(1,1, figsize=(8,3), dpi=350)
plt.imshow(np.concatenate(dimgs, 1), interpolation='none', cmap='gray')
plt.axis('off')

# %%
# Check how model makes predictions
outputs = model(img.to(device))
print(outputs)
print(dict(zip(model.pathologies, outputs[0].detach().cpu().numpy())))
# %%
