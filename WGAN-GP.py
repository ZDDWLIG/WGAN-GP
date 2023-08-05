import os

import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from module import WGAN_gp,test
import numpy as np
NoiseSize=100
BatchSize=512
EpochNum=5000
lr=2e-4
Device=th.device('cuda' if th.cuda.is_available() else 'cpu')
#prepare data
trans=transforms.ToTensor()
train_dataset=datasets.CIFAR10(root='/home/gao/Desktop/code/',download=False,train=True,transform=trans)
test_dataset=datasets.CIFAR10(root='/home/gao/Desktop/code/',download=False,train=False,transform=trans)
# train_dataset=datasets.MNIST(root='/home/gao/Desktop/code/',download=True,train=True,transform=trans)
# test_dataset=datasets.MNIST(root='/home/gao/Desktop/code/',download=False,train=False,transform=trans)
train_dataloader=DataLoader(dataset=train_dataset,shuffle=True,batch_size=BatchSize,num_workers=8)
test_dataloader=DataLoader(dataset=test_dataset,shuffle=False,batch_size=BatchSize,num_workers=8)
model=WGAN_gp(noise_size=NoiseSize,batch_size=BatchSize,epoch_num=EpochNum,device=Device,lr=lr)
model.train(train_dataloader)
# model.load(generator_path='/home/gao/PycharmProjects/results/checkpoints/generator_280.pth',
#            discriminator_path='/home/gao/PycharmProjects/results/checkpoints/discriminator_280.pth')
# test(model,testloader=test_dataloader,test_epoch_num=5,device=Device,batch_size=BatchSize)



