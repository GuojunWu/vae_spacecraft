#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 00:31:04 2020

@author: guojun
This is for neural network architecture of VAE
"""
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

# Baseline model
class VAE_PCA(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc11 = nn.Linear(20,12)
        self.fc12 = nn.Linear(20,12)
        self.fc2 = nn.Linear(12, 20)

    def encode(self, x):
    

        return self.fc11(x), self.fc12(x)  # no activation function in bottleneck?

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.as_tensor(
            np.random.multivariate_normal(np.zeros(mu.size()[1]), np.eye(mu.size()[1]), size=mu.size()[0]),
            dtype=torch.float32)
        #eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):

        return self.fc2(z)  # sigmoid?

    def forward(self, x):
        mu, logvar = self.encode(x)  # [1,28,28], [1,784]
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

# Add one hidden layer

class VAE_one(nn.Module):
    def __init__(self,hu,bu):
        super().__init__()
        self.fc1 = nn.Linear(20,hu)
        self.fc21 = nn.Linear(hu,bu)
        self.fc22 = nn.Linear(hu,bu)
        self.fc3 = nn.Linear(bu, hu)
        self.fc4 = nn.Linear(hu,20)

        

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)  # no activation function in bottleneck?

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.as_tensor(
            np.random.multivariate_normal(np.zeros(mu.size()[1]), np.eye(mu.size()[1]), size=mu.size()[0]),
            dtype=torch.float32)
        #eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        h2 = F.relu(self.fc3(z))
        return self.fc4(h2)  # sigmoid?

    def forward(self, x):
        mu, logvar = self.encode(x)  # [1,28,28], [1,784]
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

class VAE_two(nn.Module):
    def __init__(self,hu1,hu2,bu):
        super().__init__()
        self.fc1 = nn.Linear(20,hu1)
        self.fc2 = nn.Linear(hu1,hu2)
        self.fc31 = nn.Linear(hu2,bu)
        self.fc32 = nn.Linear(hu2,bu)
        self.fc4 = nn.Linear(bu,hu2)
        self.fc5 = nn.Linear(hu2,hu1)
        self.fc6 = nn.Linear(hu1,20)
        

        

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)  # no activation function in bottleneck?

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.as_tensor(
            np.random.multivariate_normal(np.zeros(mu.size()[1]), np.eye(mu.size()[1]), size=mu.size()[0]),
            dtype=torch.float32)
        #eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        h3 = F.relu(self.fc4(z))
        h4 = F.relu(self.fc5(h3))
        return self.fc6(h4)  # sigmoid?

    def forward(self, x):
        mu, logvar = self.encode(x)  # [1,28,28], [1,784]
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z


class VAE_three(nn.Module):
    def __init__(self,hu1,hu2,hu3,bu):
        super().__init__()
        self.fc1 = nn.Linear(20,hu1)
        self.fc2 = nn.Linear(hu1,hu2)
        self.fc3 = nn.Linear(hu2,hu3)
        self.fc41 = nn.Linear(hu3,bu)
        self.fc42 = nn.Linear(hu3,bu)
        self.fc5 = nn.Linear(bu,hu3)
        self.fc6 = nn.Linear(hu3,hu2)
        self.fc7 = nn.Linear(hu2,hu1)
        self.fc8 = nn.Linear(hu1,20)
        

        

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        return self.fc41(h3), self.fc42(h3)  # no activation function in bottleneck?

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.as_tensor(
            np.random.multivariate_normal(np.zeros(mu.size()[1]), np.eye(mu.size()[1]), size=mu.size()[0]),
            dtype=torch.float32)
        #eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        h4 = F.relu(self.fc5(z))
        h5 = F.relu(self.fc6(h4))
        h6 = F.relu(self.fc7(h5))
        return self.fc8(h6)  # sigmoid?

    def forward(self, x):
        mu, logvar = self.encode(x)  # [1,28,28], [1,784]
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z
    

class VAE_four(nn.Module):
    def __init__(self,hu1,hu2,hu3,hu4,bu):
        super().__init__()
        self.fc1 = nn.Linear(20,hu1)
        self.fc2 = nn.Linear(hu1,hu2)
        self.fc3 = nn.Linear(hu2,hu3)
        self.fc4 = nn.Linear(hu3,hu4)
        self.fc51 = nn.Linear(hu4,bu)
        self.fc52 = nn.Linear(hu4,bu)
        self.fc6 = nn.Linear(bu,hu4)
        self.fc7 = nn.Linear(hu4,hu3)
        self.fc8 = nn.Linear(hu3,hu2)
        self.fc9 = nn.Linear(hu2,hu1)
        self.fc10 = nn.Linear(hu1,20)

        

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        h4 = F.relu(self.fc4(h3))
        return self.fc51(h4), self.fc52(h4)  # no activation function in bottleneck?

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.as_tensor(
            np.random.multivariate_normal(np.zeros(mu.size()[1]), np.eye(mu.size()[1]), size=mu.size()[0]),
            dtype=torch.float32)
        #eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        h5 = F.relu(self.fc6(z))
        h6 = F.relu(self.fc7(h5))
        h7 = F.relu(self.fc8(h6))
        h8 = F.relu(self.fc9(h7))
        return self.fc10(h8)  # sigmoid?

    def forward(self, x):
        mu, logvar = self.encode(x)  # [1,28,28], [1,784]
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z
    



if __name__ == '__main__':
    lala = []
    lala = VAE()