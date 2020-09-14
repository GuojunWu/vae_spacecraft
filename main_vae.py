# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 12:03:25 2020

@author: Administrator
"""

import dataexport as dp
import architecture as ac
import trainmodel as tm
import evaluatemodel as em

import argparse
import torch

import matplotlib.pyplot as plt

import numpy as np
import math
from scipy.stats import norm
from scipy import stats
import pandas as pd

if __name__ == '__main__':

#%% Obtain subsamples 
    
    # obtain dataset
    exporter = dp.Dataset_Exporter()
    exporter.load("Sample")
    dataset = exporter.dataset.data
    
    # Subsample size
    train_size = 100000
    valid_size = 50000
    test_size = 50000

    # Subsampling approach    
    def startpoint_criteria(start_index, dataset, size):
        mean_list = []
        std_list =  []
        
        for start_time in start_index:
            mean = np.mean(dataset[start_time:(start_time+size),:],0)
            std = np.std(dataset[start_time:(start_time+size),:],0)
            mean_list.append(mean)
            std_list.append(std)
             
        return np.asarray(mean_list), np.asarray(std_list)

    def startpoint_error(mean_list,std_list,whole_dataset):
        mean_error = []
        std_error = []
        
        for sample in mean_list:
            error = np.sum(np.power((sample - np.mean(whole_dataset,0)), 2))
            mean_error.append(error)
        for sample in std_list:
            error = np.sum(np.power((sample - np.std(whole_dataset,0)), 2))
            std_error.append(error)
        return np.asarray(mean_error), np.asarray(std_error)
    
    n_start = 19
    np.random.seed(123)
    dataselect_size = train_size + valid_size + test_size
    dataselect_start = np.sort(np.random.choice(dataset.shape[0]-dataselect_size, n_start, replace=False))
    
    [dataselect_mean_list, dataselect_std_list] = startpoint_criteria(dataselect_start, dataset, dataselect_size)
    [dataselect_mean_error,dataselect_std_error] = startpoint_error(dataselect_mean_list, dataselect_std_list,dataset)
  
    # Plots for comparing candidate subsamples
    x1 = dataselect_mean_error
    x2 = dataselect_std_error
    til = "Test set"

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(x1,'o',label = "Mean")
    ax1.plot(x2,'o',label = "Std")
    ax1.set(xlabel = "Subsample",ylabel = "Sum of squared error",title = til)
    ax1.legend()


    list = [1]
    x1 = dataselect_mean_list
    x2 = dataselect_std_list
    fig = plt.figure(figsize=(14, 4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    for ind, line in enumerate(x1[list]):   
          
        ax1.plot(np.absolute(line-np.mean(dataset,0)),label = "Subsample "+str(list[ind]))
        ax1.set_xlabel('Sensor', fontsize=15)
        ax1.set_ylabel('Difference',fontsize=15)
        ax1.set(title = "Mean")
        ax1.legend()
    
    
    for ind, line in enumerate(x2[list]):   
      
        ax2.plot(np.absolute(line-np.std(dataset,0)),label = "Subsample "+str(list[ind]))
        ax2.set_xlabel('Sensor', fontsize=15)
        ax2.set_ylabel('Difference',fontsize=15)
        ax2.set(title = "Std")
        ax2.legend()
        
    
    n = 1
    lab = "Validation set"
    fig = plt.figure(figsize=(14, 4))
    ax1 = fig.add_subplot(121)
    ax1.plot(np.log(np.absolute(np.mean(dataset,0))),label = "whole dataset")
    ax1.plot(np.log(np.absolute(x1[n])),label = lab)
    ax1.set_xlabel('Sensor', fontsize=15)
    ax1.set_ylabel('Log(|Mean|)',fontsize=15)
    ax1.set(title = "Mean of sensor value across time points")
    ax1.legend()
    
    
    ax2 = fig.add_subplot(122)
    ax2.plot(np.log(np.std(dataset,0)),label = "whole dataset")
    ax2.plot(np.log(x2[n]),label = lab)
    ax2.set_xlabel('Sensor', fontsize=15)
    ax2.set_ylabel('Log(std)',fontsize=15)
    ax2.set(title = "Std of sensor value across time points")
    ax2.legend()
    
    # Select subsample
    dataselect = dataset[dataselect_start[1]:(dataselect_start[1]+dataselect_size),:]
    train_split = dataselect.shape[0]//2
    valid_split = dataselect.shape[0]//4
    
    train_set = dataselect[:train_split,]
    valid_set = dataselect[train_split:(train_split+valid_split),]
    test_set = dataselect[(train_split+valid_split):,]
  
    train_size = 100000
    valid_size = 50000
    test_size = 50000
    
    # Normalized
    train_normalized = []
    validation_normalized = []
    test_normalized = []
    
    for idx, signal in enumerate(np.transpose(train_set)): 
        mean = signal.mean()
        std = signal.std()
        normal = lambda x: (x - mean)/std
        
        train_normalized.append(normal(signal))
        validation_normalized.append(normal(np.transpose(valid_set)[idx,:]))
        test_normalized.append(normal(np.transpose(test_set)[idx,:]))
            
    train_normalized = np.transpose(train_normalized)   
    validation_normalized = np.transpose(validation_normalized)
    test_normalized = np.transpose(test_normalized)
    
#%% Train model
    train_set = torch.as_tensor(train_normalized,dtype=torch.float32) # torch [time,sensors]
    validation_set = torch.as_tensor(validation_normalized,dtype=torch.float32) 
    test_set = torch.as_tensor(test_normalized,dtype=torch.float32) 
    
    # Set Parameters
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=20000, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--epochs', type=int, default=2000, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--regularization', type=float, default=0.1,metavar='N',
                        help='value of regularization (default:1)')
    args = parser.parse_args()
    # print(args)
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    

  
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(validation_set, batch_size = 10000, shuffle = True, **kwargs)

    # Different neural network configurations    
    # model 16101005
    network_3_16101005_1 = ac.VAE_three(16,10,10,5).to(device)
    
    model_3_16101005_1 = tm.Model_Train(device,network_3_16101005_1,1)
    
    
    for epoch in range(1, args.epochs + 1):
        model_3_16101005_1 .training(epoch,train_loader)
        model_3_16101005_1 .val_evaluating(epoch,val_loader)
    
    print("model_3_16101005_1: Successfully Complete ",epoch," epochs") 

    network_3_16101005_001 = ac.VAE_three(16,10,10,5).to(device)
    
    model_3_16101005_001 = tm.Model_Train(device,network_3_16101005_001,0.01)
    
    
    for epoch in range(1, args.epochs + 1):
        model_3_16101005_001 .training(epoch,train_loader)
        model_3_16101005_001 .val_evaluating(epoch,val_loader)
    
    print("model_3_16101005_001: Successfully Complete ",epoch," epochs") 


    network_3_16101005_03 = ac.VAE_three(16,10,10,5).to(device)
    
    model_3_16101005_03 = tm.Model_Train(device,network_3_16101005_03,0.3)
    
    
    for epoch in range(1, args.epochs + 1):
        model_3_16101005_03 .training(epoch,train_loader)
        model_3_16101005_03 .val_evaluating(epoch,val_loader)
    
    print("model_3_16101005_1: Successfully Complete ",epoch," epochs") 


    network_3_16101005_2 = ac.VAE_three(16,10,10,5).to(device)
    
    model_3_16101005_2 = tm.Model_Train(device,network_3_16101005_2,2)
    
    
    for epoch in range(1, args.epochs + 1):
        model_3_16101005_2 .training(epoch,train_loader)
        model_3_16101005_2 .val_evaluating(epoch,val_loader)
    
    print("model_3_16101005_2: Successfully Complete ",epoch," epochs")  
    
   
    network_3_16101005_4 = ac.VAE_three(16,10,10,5).to(device)
    model_3_16101005_4 = tm.Model_Train(device,network_3_16101005_4,4)
    
    
    for epoch in range(1, args.epochs + 1):
        model_3_16101005_4 .training(epoch,train_loader)
        model_3_16101005_4 .val_evaluating(epoch,val_loader)
    
    print("model_3_16101005_4: Successfully Complete ",epoch," epochs")  
    
    
    network_3_16101005_10 = ac.VAE_three(16,10,10,5).to(device)
    model_3_16101005_10 = tm.Model_Train(device,network_3_16101005_10,10)
    
    
    for epoch in range(1, args.epochs + 1):
        model_3_16101005_10 .training(epoch,train_loader)
        model_3_16101005_10 .val_evaluating(epoch,val_loader)
    
    print("model_3_16101005_10: Successfully Complete ",epoch," epochs")  

    
    # model 161005
    network_2_161005_001 = ac.VAE_two(16,10,5).to(device)
    
    model_2_161005_001 = tm.Model_Train(device,network_2_161005_001,0.01)
    
    
    for epoch in range(1, args.epochs + 1):
        model_2_161005_001 .training(epoch,train_loader)
        model_2_161005_001 .val_evaluating(epoch,val_loader)
    
    print("model_2_161005_001: Successfully Complete ",epoch," epochs") 


    network_2_161005_03 = ac.VAE_two(16,10,5).to(device)
    
    model_2_161005_03 = tm.Model_Train(device,network_2_161005_03,0.3)
    
    
    for epoch in range(1, args.epochs + 1):
        model_2_161005_03 .training(epoch,train_loader)
        model_2_161005_03 .val_evaluating(epoch,val_loader)
    
    print("model_2_161005_1: Successfully Complete ",epoch," epochs") 


    network_2_161005_2 = ac.VAE_two(16,10,5).to(device)
    
    model_2_161005_2 = tm.Model_Train(device,network_2_161005_2,2)
    
    
    for epoch in range(1, args.epochs + 1):
        model_2_161005_2 .training(epoch,train_loader)
        model_2_161005_2 .val_evaluating(epoch,val_loader)
    
    print("model_2_161005_2: Successfully Complete ",epoch," epochs")  
    
   
    network_2_161005_4 = ac.VAE_two(16,10,5).to(device)
    model_2_161005_4 = tm.Model_Train(device,network_2_161005_4,4)
    
    
    for epoch in range(1, args.epochs + 1):
        model_2_161005_4 .training(epoch,train_loader)
        model_2_161005_4 .val_evaluating(epoch,val_loader)
    
    print("model_2_161005_4: Successfully Complete ",epoch," epochs")  
    
    
    network_2_161005_10 = ac.VAE_two(16,10,5).to(device)
    model_2_161005_10 = tm.Model_Train(device,network_2_161005_10,10)
    
    
    for epoch in range(1, args.epochs + 1):
        model_2_161005_10 .training(epoch,train_loader)
        model_2_161005_10 .val_evaluating(epoch,val_loader)
    
    print("model_2_161005_10: Successfully Complete ",epoch," epochs") 


    network_2_161005_1 = ac.VAE_two(16,10,5).to(device)
    model_2_161005_1 = tm.Model_Train(device,network_2_161005_1,1)
    
    
    for epoch in range(1, args.epochs + 1):
        model_2_161005_1 .training(epoch,train_loader)
        model_2_161005_1 .val_evaluating(epoch,val_loader)
    
    print("model_2_161005_1: Successfully Complete ",epoch," epochs")     
#%% model performance
    
    # reconstruction error and kl divergence
    print(model_3_16101005_001.model.val_re_epoch[1999],model_3_16101005_001.model.val_kld_epoch[1999])

    # learning curve
    model_3_16101005_1.plot_three()
    
    # multivariate shapiro wilk's test on validation set
    tr_gof, val_gof = model_3_16101005_1.gof(1999,50000)
    print(val_gof)
    # Critical value
    def c_msw(dim,size=2000,alpha = 0.1):
        y = math.log(size)
        mu_n = -1.5861 - 0.31082*y - 0.083751*math.pow(y,2) + 0.0038915*math.pow(y,3)
        sigma_n = math.exp(-0.4803-0.082676*y+0.0030302*math.pow(y,2))
        var_n = pow(sigma_n,2)
        var_1 = math.log((dim-1+math.exp(var_n))/dim)
        mu_1 = mu_n + 0.5*var_n-0.5*var_1
        
        t = norm.ppf(alpha)
        c = 1 - math.exp(mu_1+math.sqrt(var_1)*t)
        
        return c    
    c_msw(5,50000,0.1)
    # multivariate shapiro wilk's test for uniform distribution      
    s1 = np.random.uniform(0,1,50000)*2*math.sqrt(3)-math.sqrt(3)
    a1 = stats.shapiro(s1)
    a1[0]

#%% Model evaluation on test set
    final_network_1 = model_3_16101005_1.model.network
    data_loader = torch.utils.data.DataLoader(test_set, batch_size=validation_set.shape[0], shuffle=False, **kwargs)
    final_te = em.Final_Evaluate(device, final_network_1,args.regularization)
    final_te.anomaly_score(data_loader) 
    
    #  mean squared error
    org_te = final_te.finalmodel.original_data
    rec_te = final_te.finalmodel.reconstruct_data
    error = (org_te - rec_te)
    se = error * error
    mse = se.sum(1)
    print(mse.mean())
    print(mse.std())
    
    # mean suquared error of mean predictor
    mean_predictor = train_set.mean(axis = 0).numpy()
    error = (org_te - mean_predictor)
    se = error * error
    mse = se.sum(1)
    print(mse.mean())
    print(mse.std())
    
    # Maximum mean discrepancy for disentangled representation
    import MMD
    np.random.seed(0)
    z_1 = final_te.finalmodel.z_data
    np.random.seed(0)
    X_1 = np.expand_dims(np.random.choice(np.sum(np.multiply(z_1,z_1),1),size=10000, replace=False),axis = -1) 
    sigma,MMDXY= MMD.MMD_Calculate(C,X_1,sigma=-1,SelectSigma=2)
    print(sigma,MMDXY)
    #MMD in uniform distriubtion
    C = np.expand_dims(np.random.chisquare(5, 10000),axis = -1)
    s1 = np.expand_dims(np.random.uniform(5-math.sqrt(120)/2,5+math.sqrt(120)/2,10000),axis = -1)
    sigma,MMDXY= MMD.MMD_Calculate(C,s1,sigma=-1,SelectSigma=2)
    print(sigma,MMDXY)


#%% Anomaly detection
    
    # Euclidean distance
    z = final_te.finalmodel.z_data
    x = np.sum(np.multiply(z,z),1)

    sort_x = np.sort(x, axis=None)

    sum_1 = 0
    cum_prob = []
    for i in range(0,len(sort_x)):
        cum_prob.append((i+1)/len(sort_x))
        
        
    cum_prob = np.asarray(cum_prob)
    
    # 10 largest elements
    n = 1
    list_2 = [i[0] for i in sorted(enumerate(x), key=lambda m:m[1])]
    x[list_2[-10:]]
    list_2[-10:]

