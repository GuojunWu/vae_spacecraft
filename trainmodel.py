#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 00:06:14 2020

@author: guojun
"""
import torch
from torch import optim
from torch.nn import functional as F

import math

import numpy as np

from scipy import stats

import matplotlib.pyplot as plt
from scipy.stats import norm

import pickle

class Model:
    def __init__(self):
        
        self.device = None
        self.network = None
        self.regularization = None
        
        ## Training ##
        
        # Reconstruction loss #
        self.train_re_batch = []
        self.train_re_t_batch = []
        self.train_re_epoch = []
        self.train_re_t_epoch = []
        
        # KLD #
        self.train_kld_epoch = []
        
        # Total loss#
        self.train_loss_epoch = []
        
        # z list#
        self.train_z_epoch = []
        
        ## Evaluation ##
        
        # Reconstruction loss#
        self.val_re_epoch = []
        self.val_re_t_epoch = []
        
        # KLD #
        self.val_kld_epoch = []
#        self.val_gof_epoch = []
        
        # Total loss#
        self.val_loss_epoch = []
        
        # z list#
        self.val_z_epoch = []
        
class Model_Train:
    def __init__(self,device,network,regularization):
        self.model = Model()
        self.device = device
        
        self.model.network = network
        self.optimizer = optim.Adam(network.parameters(), lr=1e-3)
        
        self.model.regularization = regularization
        
     # DUMP
    def dump(self, file):
        with open(file, "wb") as pickle_out:
            pickle.dump(self.model, pickle_out)
            
            
    # LOAD
    def load(self, file):
        with open(file, "rb") as pickle_in:
            self.model = pickle.load(pickle_in)
            
    def loss_function(self, recon_x, x, mu, logvar):  # per batch size
        """
        loss: total loss
        re: reconstruction loss
        kld: kl divergence
        re_obs: loss for each time point (sum over sensors): [sample_size]
        re_all: loss for each time point for each sensor: [sample_size, sensors] 
        """
        re = F.mse_loss(recon_x, x, reduction='sum')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = re + self.model.regularization * kld
        re_time = F.mse_loss(recon_x, x, reduction='none').sum(dim=1)
        re_all = F.mse_loss(recon_x, x, reduction='none')
        
        return loss, re, kld, re_time, re_all
    
    
    def training(self, epoch,train_loader):
        
        self.model.network.train()
        
        mses=[] # mses for all time point (summation along all sensors)
        
        train_loss = 0
        train_re = 0
        train_kld = 0
        
        z_list = []
        
        for batch_idx, data in enumerate(train_loader): 
            data = data.to(self.device)
            data = data.float()
            recon_batch, mu, logvar, z = self.model.network(data)
            loss, mse, kld, mse_time, _= self.loss_function(recon_batch, data, mu, logvar)
            
            self.optimizer.zero_grad()  
            loss.backward() 
            self.optimizer.step()
            
            # Record average reconstruction loss for every batch
            self.model.train_re_batch.append(mse.item()/len(data)) 
            # Record t value of average reconstruction loss for every batch
            self.model.train_re_t_batch.append(mse_time.std().item()/math.sqrt(len(data)))
            
            # Record reconstruction loss for all time points (for computing std of one epoch)
            mses.append(mse_time.detach().numpy())
            
            
            # Summation of loss for one epoch
            train_loss += loss.item()  
            train_re += mse.item()
            train_kld += kld.item()
            
            # Sampling vector on latent space
            z_numpy = z.detach().numpy()
            z_list.append(z_numpy)
     
        z_list = np.asarray(z_list).reshape(len(train_loader.dataset),z.size()[1])
        self.model.train_z_epoch.append(z_list)
        
        # print("Success:", epoch)
        # Record average loss for every epoch (len(train_loader.dataset) = total training size)
        self.model.train_loss_epoch.append(train_loss/len(train_loader.dataset))
        self.model.train_kld_epoch.append(train_kld/len(train_loader.dataset))
        self.model.train_re_epoch.append(train_re/len(train_loader.dataset))
             
        # Calculate std of average reconstruction loss for one epoch
        mse_std = np.asarray(mses).std()
        # Record t value of average reconstruction loss for every epoch
        self.model.train_re_t_epoch.append(mse_std/math.sqrt(len(train_loader.dataset)))
    
    def val_evaluating(self, epoch,val_loader):
    
        self.model.network.eval()
        
        mses = [] # mse for all time point
    
        val_loss = 0
        val_re = 0
        val_kld = 0
        z_list = []
    
        
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                data = data.to(self.device)
                data = data.float()
                
                recon_batch, mu, logvar, z = self.model.network(data)
    
                loss, mse, kld, mse_time,_= self.loss_function(recon_batch, data, mu, logvar)  # calculate loss
                
    
                # Don't need mse for every batch, only for every epoch
    
                # Record reconstruction loss for al time points
                mses.append(mse_time.numpy())
    
                # Summation of loss for one epoch
                val_loss += loss.item()
                val_re += mse.item()
                val_kld += kld.item()
                
                # Sampling vector on latent space
                z_numpy = z.numpy()
                z_list.append(z_numpy)
                
        
        # Shapiro-Wilk's statistic
        z_list = np.asarray(z_list).reshape(len(val_loader.dataset),z.size()[1])
        self.model.val_z_epoch.append(z_list)
#        stat = 0
#        for column in z_list.T:
#            a = stats.shapiro(column)
#            stat += a[0]
    
        print(epoch,'====> Test set loss: {:.4f} Reconstruction loss: {:.4f} KLD: {:.4f}'.format(val_loss/len(val_loader.dataset), val_re/len(val_loader.dataset), val_kld/len(val_loader.dataset)))
    
        # Record average loss for every epoch
        self.model.val_loss_epoch.append(val_loss/len(val_loader.dataset))
        self.model.val_kld_epoch.append(val_kld/len(val_loader.dataset))
        self.model.val_re_epoch.append(val_re/len(val_loader.dataset))
#        self.model.val_gof_epoch.append(stat/z.size()[1])
    
        # Calculate std of average reconstruction loss for one epoch
        mse_std = np.asarray(mses).std()
        self.model.val_re_t_epoch.append(mse_std/math.sqrt(len(val_loader.dataset)))
        
    
    
    def gof(self,epoch,size = 2000):
         # size: selected size; z_size: total size pf z
        train_z = self.model.train_z_epoch[epoch-1] # [obs, dims]
        valid_z = self.model.val_z_epoch[epoch-1]
        
        z_dim = train_z.shape[1]
        train_idx = np.random.choice(train_z.shape[0],size,replace=False)
        valid_idx = np.random.choice(valid_z.shape[0],size,replace = False)
        tr_z_sample = train_z[train_idx,:]
        val_z_sample = valid_z[valid_idx,:]
        tr_stat = 0
        for tr_dim in tr_z_sample.T:
            a = stats.shapiro(tr_dim)
            tr_stat += a[0]
        tr_stat = tr_stat/z_dim
        val_stat = 0
        for val_dim in val_z_sample.T:
            a = stats.shapiro(val_dim)
            val_stat += a[0]
        val_stat = val_stat/z_dim  
        
        return tr_stat, val_stat
        
    
    def plot_re(self):
        train_batch = self.model.train_re_batch
        train_epoch = self.model.train_re_epoch
        valid_epoch = self.model.val_re_epoch
        
#        train_t_batch = self.model.train_re_t_batch
#        train_t_epoch = self.model.train_re_t_epoch
#        valid_t_epoch = self.model.val_re_t_epoch
        
        
        """
        Fig 1: Reconstruction loss v.s. baches (train)
        Fig 2: Reconstruction loss v.s. epochs (Compare train, validation)
        Fig 3: Reconstruction loss v.s. epochs (train)
        Fig 4: Reconstruction loss v.s. epochs (test)
        """
        fig = plt.figure(figsize=(10, 6))
        
        ax1 = fig.add_subplot(221)
        ax1.plot(train_batch,label = "Training Set",color = "b")
#        ax1.fill_between(range(1,len(train_batch)+1),train_t_batch+2*train_t_batch,
#                     train_batch-2*train_t_batch, alpha=0.1, color="g")
        ax1.set_xlim(1,len(train_batch))
        ax1.set(xlabel = "Batch",ylabel = "Reconstruction Loss")
        ax1.legend()
        
        ax2 = fig.add_subplot(222)
        ax2.plot(train_epoch,label = "Training Set",color = "b")
        ax2.plot(valid_epoch,label = "Validation Set", color = "g")
        ax2.set_xlim(1,len(train_epoch))
        ax2.set(xlabel = "Epoch",ylabel = "Reconstruction Loss")
        ax2.legend()
        
        
        ax3 = fig.add_subplot(223)
        ax3.plot(train_epoch,label = "Training Set",color = "b")
#        ax3.fill_between(range(1,len(train_epoch)+1),train_epoch+2*train_t_epoch,
#                     train_epoch-2*train_t_epoch, alpha=0.1, color="g")
        ax3.set_xlim(1,len(train_epoch))
        ax3.set(xlabel = "Epoch",ylabel = "Reconstruction Loss")
        ax3.legend()
        
        
        ax4 = fig.add_subplot(224)
        ax4.plot(valid_epoch,label = "Validation Set",color = "g")
#        ax4.fill_between(range(1,len(valid_epoch)+1),valid_epoch+2*train_t_epoch,
#                     valid_epoch-2*valid_t_epoch, alpha=0.1, color="g")
        ax4.set_xlim(1,len(valid_epoch))
        ax4.set(xlabel = "Epoch",ylabel = "Reconstruction Loss")
        ax4.legend()
        
        fig.tight_layout()
        plt.show()
        
    def plot_three(self):
        
        train_re = self.model.train_re_epoch
        valid_re = self.model.val_re_epoch
        
        train_kld = self.model.train_kld_epoch
        valid_kld = self.model.val_kld_epoch
        
        train_loss = self.model.train_loss_epoch
        valid_loss = self.model.val_loss_epoch
        epoch = len(train_re)
        
        """
        Fig 1: Reconstruction loss v.s. epochs (Compare train, validation)
        Fig 2: KLD v.s. epochs (Compare train, validation)
        Fig 3: Total loss v.s. epochs (Compare train, validation)
        """
        
        fig = plt.figure(figsize = (15,3))
        
        ax1 = fig.add_subplot(131)
        ax1.plot(train_re,label = "Training Set",color = "b")
        ax1.plot(valid_re,label = "Validation Set", color = "g")
        ax1.set_xlim(1,epoch)
        ax1.set(xlabel = "Epoch",ylabel = "Reconstruction loss")
        ax1.legend()
        
        ax2 = fig.add_subplot(132)
        ax2.plot(train_kld,label = "Training Set",color = "b")
        ax2.plot(valid_kld,label = "Validation Set", color = "g")
        ax2.set_xlim(1,epoch)
        ax2.set(xlabel = "Epoch",ylabel = "KL Divergence")
        ax2.legend()
        
        ax3 = fig.add_subplot(133)
        ax3.plot(train_loss,label = "Training Set",color = "b")
        ax3.plot(valid_loss,label = "Validation Set", color = "g")
        ax3.set_xlim(1,epoch)
        ax3.set(xlabel = "Epoch",ylabel = "Total Loss")
        ax3.legend()
        
        fig.tight_layout()
        plt.show()
    
    def plot_z(self,epoch):
        train_z = self.model.train_z_epoch[epoch -1]
        valid_z = self.model.val_z_epoch[epoch-1]
        
        for i in range(0,train_z.shape[1]): # dimenison of latent space
            train_x = train_z[:,i]
            valid_x = valid_z[:,i]
            (train_mu, train_sigma) = norm.fit(train_x)
            (valid_mu,valid_sigma) = norm.fit(valid_x)
            
            fig = plt.figure(figsize=(12, 2))
            
            ax1 = fig.add_subplot(121)
            _, train_bins, _ = ax1.hist(train_x, bins='auto',density = True,color = "darkblue")
            
            train_snd = norm.pdf(train_bins, 0, 1)
            train_nd = norm.pdf(train_bins,train_mu,train_sigma)
            ax1.plot(train_bins, train_snd, 'k--', linewidth=2,label = '$N(0,1)$')
            ax1.plot(train_bins,train_nd, 'r--',linewidth = 2, label = '$N(%.3f, %.3f)$' %(valid_mu,valid_sigma))
            ax1.set(xlabel = 'Sampled latent vector',ylabel = "Probability",title = "Training set")
            ax1.grid(True)
            ax1.legend()
    
        
            
            ax2 = fig.add_subplot(122)
            _,valid_bins, _ = ax2.hist(valid_x, bins='auto',density = True,color = "darkgreen")
            
            valid_snd = norm.pdf(valid_bins, 0, 1)
            valid_nd = norm.pdf(valid_bins,valid_mu,valid_sigma)
            ax2.plot(valid_bins, valid_snd, 'k--', linewidth=2,label = '$N(0,1)$')
            ax2.plot(valid_bins,valid_nd, 'r--',linewidth = 2, label = '$N(%.3f, %.3f)$' %(valid_mu,valid_sigma))
            ax2.set(xlabel = 'Sampled latent vector',ylabel = "Probability",title = "Validation set")
            ax2.grid(True)
            ax2.legend()
            
            
            fig.suptitle("Dimension:"+str(i+1))
            plt.show()
        
        
        
        
        