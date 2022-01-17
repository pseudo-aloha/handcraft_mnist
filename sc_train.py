
from definition import *
from sc import *
from Claissifier import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from tqdm import tqdm
import numpy as np
import os
from torchvision.datasets import CIFAR10
import torchvision.transforms as tvt
from torch import save, no_grad
import shutil
from torchvision.datasets import MNIST, FashionMNIST

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

__all__ = ['BNNLinear', 'BNNConv2d']


def load_train_data(batch_size=64, sampler=None):
    transform = tvt.Compose([
        tvt.RandomCrop(28, padding=4),
        tvt.RandomHorizontalFlip(),
        tvt.ToTensor(),
        # tvt.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    if sampler is None:
        shuffle = True
    else:
        shuffle = False

    dataset = MNIST(os.path.join('datasets', 'mnist'), train=True,
            download=True, transform=transform)
    dataset = torch.utils.data.Subset(dataset, np.arange(num_train))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
            shuffle=shuffle, sampler=sampler, num_workers=1, pin_memory=True)

    return loader


def load_test_data(batch_size=1000, sampler=None):
    transform =  tvt.Compose([
        tvt.ToTensor(),
        # tvt.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    dataset = MNIST(os.path.join('datasets', 'mnist'), train=False,
            download=True, transform=transform)
    dataset = torch.utils.data.Subset(dataset, np.arange(num_test))
    loader = torch.utils.data.DataLoader( dataset, batch_size=batch_size,
            shuffle=False, sampler=sampler, num_workers=1, pin_memory=True)

    return loader

import importlib
cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')
torch.manual_seed(0)
# if cuda:
#     torch.backends.cudnn.deterministic=True
#     torch.cuda.manual_seed(0)
dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
train_loader = load_train_data(batch_size)
test_loader = load_test_data(batch_size)



# elif FLAGS.bin_type == 'dorefa':
#     classification = DorefaClassifier(model, train_loader, test_loader, device)

if __name__ == "__main__":
    # classification.train(criterion, optimizer, 10, scheduler, "results/bnn_caffenet_cifar10" )
    dnn = DeepNeuralNetwork(sizes=[784, 128, 64, 10])
    dnn.train(train_loader, test_loader)