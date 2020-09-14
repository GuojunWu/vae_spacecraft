#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 13:54:49 2020

@author: guojun

This is for evaluating trained model
"""
import torch
from torch import optim
from torch.nn import functional as F

import numpy as np
import matplotlib.pyplot as plt

class FinalModel:
    
    def __init__(self):
        
        self.network = None
        self.regularization = None
        
        self.reconstruct_data = None
        self.reconstruct_error = None
        self.original_data = None
        self.mean_error = []
        self.mu_data = None
        self.logvar_data = None
        self.z_data = None
  
      
class Final_Evaluate:
    
    def __init__(self, device, network,regularization):
        self.finalmodel = FinalModel() 
        
        self.finalmodel.network = network
        self.optimizer = optim.Adam(network.parameters(), lr=1e-3)
        self.device = device
        
        self.finalmodel.regularization = regularization
        
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
        loss = re + self.finalmodel.regularization * kld
        re_time = F.mse_loss(recon_x, x, reduction='none').sum(dim=1)
        re_all = F.mse_loss(recon_x, x, reduction='none')
        
        return loss, re, kld, re_time, re_all
    
    
    def final_evaluate_re(self,data_loader):
    
        self.finalmodel.network.eval()
        
        datas = []
        mses = [] # mse for all time point
        reconx = []
        
    
        
        with torch.no_grad():
            for batch_idx, data in enumerate(data_loader):
                data = data.to(self.device)
                data = data.float()
                
                datas.append(data.numpy())
                
                recon_batch, mu, logvar, z = self.finalmodel.network(data)
                
                reconx.append(recon_batch.numpy())
    
                loss, mse, kld, _ , mse_all = self.loss_function(recon_batch, data, mu, logvar)  
                
                mses.append(mse_all.numpy()) 
            
        reconx_np = np.asarray(reconx)
        reconx_np = reconx_np.reshape(len(data_loader.dataset),-1) #[observations, sensors]
        self.finalmodel.reconstruct_data = reconx_np
    
        mses_np = np.asarray(mses)
        mses_np = mses_np.reshape(len(data_loader.dataset),-1)
        self.finalmodel.reconstruct_error = mses_np
        
        datas_np = np.asarray(datas)
        datas_np = datas_np.reshape(len(data_loader.dataset),-1)
        self.finalmodel.original_data = datas_np
    
   
    def mean_response_error(self,mean_response):
        
        
        for idx, signal in enumerate(np.transpose(self.finalmodel.original_data).tolist()): 
            mean = mean_response[idx]
            norm = (signal - mean)*(signal-mean)         
            self.finalmodel.mean_error.append(norm)            
        
        self.finalmodel.mean_error = np.transpose(self.finalmodel.mean_error)
    
#    def plot_re(self):
#        
#        
#        for i in range(0, self.finalmodel.original_data.shape[1]):
#            
#            
#            
#            plt.plot(np.transpose(data)[i])
#            plt.ylabel("Signal: " + self.dataset.sensors[i])
#            plt.xlabel("Time (days)")
#            plt.show()
#
#            plt.plot(self.finalmodel.original_data[i])
#            plt.ylabel("Signal: " + self.dataset.sensors[i])
#            plt.xlabel("Time (days)")
#            plt.show()

        
    def anomaly_score(self,data_loader):
        self.finalmodel.network.eval()
        
        
        mu_list = []
        logvar_list = [] # mse for all time point
        z_list = []
    
        
        with torch.no_grad():
            for batch_idx, data in enumerate(data_loader):
                data = data.to(self.device)
                data = data.float()
                
                recon_batch, mu, logvar, z = self.finalmodel.network(data)
                
                mu_list.append(mu.numpy())
    
                logvar_list.append(logvar.numpy())
                z_list.append(z.numpy())
                
            
        mu_list_np = np.asarray(mu_list)
        mu_list_np = mu_list_np.reshape(len(data_loader.dataset),-1)
        self.finalmodel.mu_data = mu_list_np
        
        
        logvar_list_np = np.asarray(logvar_list)
        logvar_list_np = logvar_list_np.reshape(len(data_loader.dataset),-1)
        self.finalmodel.logvar_data = logvar_list_np
        
        
        z_list_np = np.asarray(z_list)
        z_list_np = z_list_np.reshape(len(data_loader.dataset),-1)
        self.finalmodel.z_data = z_list_np
        
        
        
        
        
      
