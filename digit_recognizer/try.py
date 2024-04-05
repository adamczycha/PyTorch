import torch.multiprocessing as mp
mp.set_start_method('spawn')
import torch 
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import random
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

batch_size_train = 32
batch_size_test = 100

class MnistDataset(Dataset):

    def __init__(self):
        #data loading
        train_xy = np.loadtxt('C:/Users/Marcel/Documents/PyTorch/kaggle/digit_recognizer/train.csv', delimiter=',',dtype=np.float32, skiprows=1) 
        self.train_x = torch.from_numpy(train_xy[:,1:])
        self.train_y = torch.from_numpy(train_xy[:,[0]])
        self.n_samples = train_xy.shape[0]
    def __getitem__(self,index):
        return self.train_x[index], self.train_y[index]
        #
    def __len__(self):
        #len(dataset)
        return self.n_samples
    
dataset = MnistDataset()
# features, label = dataset[0]
# print(features,label)

dataloader = DataLoader(dataset=dataset, batch_size = batch_size_train, shuffle=True, num_workers=0)

datatiter = iter(dataloader)
data = next(datatiter)
features,labels = data
print(features,labels)