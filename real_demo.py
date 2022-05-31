#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 17:18:53 2022


"""


import numpy as np
import model, data, kernel, utils, estimator
from objective import *
import tests, tst

import mmdagg as agg

import time 


import matplotlib.pyplot as plt
import matplotlib
# font options
font = {
    #'family' : 'normal',
    #'weight' : 'bold',
    'size'   : 16
}

plt.rc('font', **font)
plt.rc('lines', linewidth=2)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


import os
import pylab
# %load_ext autoreload
# %autoreload 2


import argparse



import torch
from torch.nn import functional as F

import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable

from torchvision.transforms import ToTensor, Normalize, Compose, transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


mnist = MNIST(root='data', 
              train=True, 
              download=True,
              transform=Compose([ToTensor(), Normalize(mean=(0.5,), std=(0.5,))]))

mnist

n = 25
s = 1314
real_images = mnist.train_data[s:s+n ,:,:].numpy()

R, C = 5, 5
plt.figure(figsize=(10,10))
for i in range(n):
    plt.subplot(R, C, i + 1)
    plt.imshow(real_images[i], cmap='gray')
    plt.axis("off")
# plt.show()
plt.savefig("fig/mnist_real.pdf",layout="tight")


### load pretrained models

## non-score based models

### models adapted from https://github.com/csinva/gan-vae-pretrained-pytorch.git

num_gpu = 1 if torch.cuda.is_available() else 0


# DC-GAN  https://arxiv.org/abs/1511.06434
from pretrained.mnist_dcgan.dcgan import Discriminator, Generator

D = Discriminator(ngpu=1).eval()
G = Generator(ngpu=1).eval()


# load weights
D.load_state_dict(torch.load('pretrained/mnist_dcgan/weights/netD_epoch_99.pth'))
G.load_state_dict(torch.load('pretrained/mnist_dcgan/weights/netG_epoch_99.pth'))
if torch.cuda.is_available():
    D = D.cuda()
    G = G.cuda()




batch_size = 25
latent_size = 100

fixed_noise = torch.randn(batch_size, latent_size, 1, 1)
if torch.cuda.is_available():
    fixed_noise = fixed_noise.cuda()
fake_images = G(fixed_noise)


# z = torch.randn(batch_size, latent_size).cuda()
# z = Variable(z)
# fake_images = G(z)

fake_images_np = fake_images.cpu().detach().numpy()
fake_images_np = fake_images_np.reshape(fake_images_np.shape[0], 28, 28)
R, C = 5, 5

plt.figure(figsize=(10,10))
for i in range(batch_size):
    plt.subplot(R, C, i + 1)
    plt.imshow(fake_images_np[i], cmap='gray')
    plt.axis("off")
# plt.show()
plt.savefig("fig/mnist_dcgan.pdf",layout="tight")

outputs = D(fake_images)
print(outputs)



batch_size = 30
latent_size = 100

fixed_noise = torch.randn(batch_size, latent_size, 1, 1)
if torch.cuda.is_available():
    fixed_noise = fixed_noise.cuda()
fake_images = G(fixed_noise)

class GANgenerator:
    def __init__(self, G, latent_size):
        self.G = G
        self.latent_size = latent_size
    def sample(self, batch_size):
        latent_size = self.latent_size
        fixed_noise = torch.randn(batch_size, latent_size, 1, 1)
        if torch.cuda.is_available():
            fixed_noise = fixed_noise.cuda()
        current = self.G(fixed_noise)
        current = current.detach().cpu().numpy()
        n = batch_size
        current = np.reshape(current, (n, 28, 28))
        current = np.reshape(current, (n, 7, 4, 7, 4))
        current = current.mean(axis=(2, 4))
        gen_sample = np.reshape(current, (n, 49))
        return gen_sample
        
DCGAN = data.GANgenerator_mnist(G, latent_size=100)


ds = data.DSGenerator(DCGAN)

m = 100
current = mnist.train_data[:m,:,:].numpy()
current = np.reshape(current, (m, 28, 28))
current = np.reshape(current, (m, 7, 4, 7, 4))
current = current.mean(axis=(2, 4))
train_sample = np.reshape(current, (m, 49))

ds = data.DSSampled(train_sample)

SM_est = estimator.ESScoreSM(lr=0.05, n_epoch=150)
p_est = model.ApproxLogDen(ds, SM_est,n=100)

n=120
fixed_noise = torch.randn(n, latent_size, 1, 1)
current = G(fixed_noise.cuda()).detach().cpu().numpy()


current = np.reshape(current, (n, 28, 28))
current = np.reshape(current, (n, 7, 4, 7, 4))
current = current.mean(axis=(2, 4))
X = np.reshape(current, (n, 49))

current = mnist.test_data[:n,:,:].numpy()
current = np.reshape(current, (n, 7, 4, 7, 4))
current = current.mean(axis=(2, 4))
X = np.reshape(current, (n, 49))

dat = data.Data(X)

sig2 = utils.meddistance(X, subsample=1000)**2
sig2
k = kernel.KGauss(sig2)
k = kernel.KGauss(10.)
kstein_est = tests.SteinMCTest(p_est, k, alpha=alpha, seed=1+i)
ksteine_res = kstein_est.perform_test(dat,return_simulated_stats=True)
ksteine_res


# X = dat.X
idx = np.random.choice(1000,n)
current = mnist.test_data[idx,:,:].numpy()
current = np.reshape(current, (n, 7, 4, 7, 4))
current = current.mean(axis=(2, 4))
Y = np.reshape(current, (n, 49))



import mmdagg as agg

mmdagg_power = 0
for i in range(100):
    start =time.time()
    Y = ds.sample(500, seed=i+13412).X
    mmdagg_power += agg.mmdagg(X[:30,:], Y, l_minus=-2, l_plus=2, 
                           B1=200, B2=200, B3=100)
    end=time.time()
    if i % 10 ==9:
        print(mmdagg_power)
# 0.77 for n=30 N = 200
# 0.07 for n=30 N = 500

mmdagg_power = 0
for i in range(100):
    start =time.time()
    idx = np.random.choice(1000,n)
    current = mnist.test_data[idx,:,:].numpy()
    current = np.reshape(current, (n, 7, 4, 7, 4))
    current = current.mean(axis=(2, 4))
    Y = np.reshape(current, (n, 49))

    mmdagg_power += agg.mmdagg(X, Y, l_minus=-2, l_plus=2, 
                           B1=200, B2=200, B3=100)
    end=time.time()
    if i % 10 ==1:
        print(mmdagg_power)

# VAE
# Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
# https://arxiv.org/abs/1312.6114
import pretrained.mnist_vae.vae as vae
from torchvision import datasets, transforms

net = vae.VAE()
net.load_state_dict(torch.load('pretrained/mnist_vae/weights/vae_epoch_25.pth'))

def to_im(x):
    return x.cpu().detach().numpy().reshape((28, 28))

im0 = datasets.MNIST(root='data', train=True).data[0].float()
im1 = datasets.MNIST(root='data', train=True).data[1].float()
im2 = datasets.MNIST(root='data', train=True).data[2].float()
im0_reconstructed = to_im(net(im0)[0])



plt.subplot(1, 2, 1)
plt.imshow(im0, cmap='gray')
plt.title('original')
plt.subplot(1, 2, 2)
plt.imshow(im0_reconstructed, cmap='gray')
plt.title('reconstruction')
plt.show()


fake_images_np = []
for i in range(25):    
    im_ = datasets.MNIST(root='data', train=True).data[i].float()
    im_rec = to_im(net(im_)[0])
    fake_images_np.append(im_rec)
    
R, C = 5, 5
plt.figure(figsize=(10,10))
for i in range(batch_size):
    plt.subplot(R, C, i + 1)
    plt.imshow(fake_images_np[i], cmap='gray')
    plt.axis("off")
# plt.show()
plt.savefig("fig/mnist_vae.pdf",layout="tight")

#GAN  https://arxiv.org/abs/1406.2661
import pretrained.mnist_gan_mlp.gan_mnist as gan_mnist

D = gan_mnist.D
G = gan_mnist.G

# load weights
D.load_state_dict(torch.load('pretrained/mnist_gan_mlp/weights/D--300.ckpt'))
G.load_state_dict(torch.load('pretrained/mnist_gan_mlp/weights/G--300.ckpt'))
if torch.cuda.is_available():
    D = D.cuda()
    G = G.cuda()

latent_size = 64
batch_size = 25

z = torch.randn(batch_size, latent_size).cuda()
z = Variable(z)
fake_images = G(z)

fake_images_np = fake_images.cpu().detach().numpy()
fake_images_np = fake_images_np.reshape(fake_images_np.shape[0], 28, 28)

R, C = 5, 5
plt.figure(figsize=(10,10))
for i in range(batch_size):
    plt.subplot(R, C, i + 1)
    plt.imshow(fake_images_np[i], cmap='gray')
    plt.axis("off")
# plt.show()
plt.savefig("fig/mnist_gan.pdf",layout="tight")
    

import yaml
import collections
import tqdm
from torchvision.utils import save_image, make_grid
from PIL import Image

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace



with open("pretrained/mnist_ncsn/config.yml", 'r') as f:
    config = yaml.load(f, yaml.BaseLoader)


# with open("configs/anneal.yml", 'r') as f:
#     config = yaml.load(f, yaml.BaseLoader)


config = dict2namespace(config)


states = torch.load('pretrained/mnist_ncsn/checkpoint.pth')

config.data.channels = int(config.data.channels)
config.model.ngf = int(config.model.ngf)
config.model.num_classes = int(config.model.num_classes)


# cd ../ncsn/
# from models import cond_refinenet_dilated as crd

score = crd.CondRefineNetDilated(config)#.to(config.device)
score = torch.nn.DataParallel(score)


score.load_state_dict(states[0])
# optimizer.load_state_dict(states[1])


score.eval()

sigmas = torch.tensor(np.exp(np.linspace(np.log(float(config.model.sigma_begin)), np.log(float(config.model.sigma_end)),
                       config.model.num_classes))).float()#.to(device)

def anneal_Langevin_dynamics(x_mod, scorenet, sigmas, n_steps_each=100, step_lr=0.00002):
    images = []

    with torch.no_grad():
        for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas), desc='annealed Langevin dynamics sampling'):
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2
            for s in range(n_steps_each):
                images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))
                noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
                grad = scorenet(x_mod, labels).to(x_mod.device)
                x_mod = x_mod + step_size * grad + noise
                # print("class: {}, step_size: {}, mean {}, max {}".format(c, step_size, grad.abs().mean(),
                #                                                          grad.abs().max()))

        return images

    

grid_size = 5

imgs = []

samples = torch.rand(grid_size ** 2, 1, 28, 28)
all_samples = anneal_Langevin_dynamics(samples, score, sigmas, 100, 0.00002)


c = 0
x_mod = samples
score(x_mod, labels)

# plot as before
fake_images_np = all_samples[-1]
fake_images_np = fake_images_np.reshape(fake_images_np.shape[0], 28, 28)

R, C = 5, 5
for i in range(batch_size):
    plt.subplot(R, C, i + 1)
    plt.imshow(fake_images_np[i], cmap='gray')
    plt.axis("off")
plt.show()
    

## plot gray images together
for i, sample in enumerate(tqdm.tqdm(all_samples, total=len(all_samples), desc='saving images')):
    sample = sample.view(grid_size ** 2, int(config.data.channels), int(config.data.image_size),
                         int(config.data.image_size))

    if config.data.logit_transform:
        sample = torch.sigmoid(sample)

    image_grid = make_grid(sample, nrow=grid_size)
    if i % 10 == 0:
        im = Image.fromarray(image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
        imgs.append(im)

    save_image(image_grid, os.path.join("mnist/", 'image_{}.png'.format(i)))
    torch.save(sample, os.path.join("mnist/", 'image_raw_{}.pth'.format(i)))



