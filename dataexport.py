#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 18:42:40 2020

@author: guojun



"""

import numpy as np
import pickle
from dmss.io.hdf5io import readorig
import matplotlib.pyplot as plt

class Dataset:
    def __init__(self):
        self.sensors = []
        self.timing = []
        self.data = []
        
        self.train = []
        self.validation = []
        self.test = []
        
        self.train_normalized = []
        self.validation_normalized = []
        self.test_normalized = []
        
        self.train_size = None
        self.validation_size = None
        self.test_size = None
        
        self.start = None
        
        self.train_timing = []
        self.validation_timing = []
        
class Dataset_Exporter:
    
    def __init__(self):
        self.dataset = Dataset()
    
    # BUILD
    
    def build(self,mission,sensors):
        # Init dataset object
        self.dataset.sensors = sensors
        
        # Reading the information from the sensors
        data = {}
        for name in sensors:
            # Read sensor data
            time, signal, info = readorig(name, mission=mission,timeInDays = True, time0 = True)
            print("Read", len(time), " points of parameter", name, "starting at time", time[0])
            data[name] = {'time' : time, 'signal' : signal} # data is a dictionary
            
            
        # Determine timing variables
        startTime = max([data[name]['time'][0] for name in sensors])
        endTime = min([data[name]['time'][-1] for name in sensors])
        stepTime = np.median([np.median(np.diff(data[name]['time'])) for name in sensors])
        
        # Fill in timing information
        self.dataset.timing = np.arange(startTime, endTime, stepTime)
        
        # Homogenize timesteps by resampling the data
#        self.dataset.data = []
        [self.dataset.data.append(np.interp(self.dataset.timing, data[name]['time'], data[name]['signal'])) for name in sensors]      
        self.dataset.data = np.transpose(self.dataset.data) # numpy array (observations, sensors)
        
        
    # DUMP
    def dump(self, file):
        with open(file, "wb") as pickle_out:
            pickle.dump(self.dataset, pickle_out)
            
            
    # LOAD
    def load(self, file):
        with open(file, "rb") as pickle_in:
            self.dataset = pickle.load(pickle_in)
            
    
    # SAMPLE
    def sampling_squence(self,train_size,validation_size,test_size,start = 0):
        
        self.dataset.train_size = train_size
        self.dataset.validation_size = validation_size
        self.dataset.test_size = test_size
    
        self.dataset.start = start
        
        self.dataset.train = self.dataset.data[start:(start+train_size),:]
        self.dataset.validation = self.dataset.data[(start+train_size):(start+train_size+validation_size),:]
        self.dataset.test = self.dataset.data[(start+train_size+validation_size):(start+train_size+validation_size+test_size),:]
        
        
        self.dataset.train_timing = self.dataset.timing[start:(start+train_size)]
        self.dataset.validation_timing = self.dataset.timing[(start+train_size):(start+train_size+validation_size)]
    
    def sampling_random(self,train_size,validation_size):
        
        self.dataset.train_size = train_size
        self.dataset.validation_size = validation_size
        
        index = np.random.choice(self.dataset.data.shape[0], train_size+validation_size, replace=False)
        samples = self.dataset.data[index,:]  
        
        self.dataset.train = samples[0:train_size,:]
        self.dataset.validation = samples[train_size:train_size+validation_size,:]
        
        
        
        
    
    # NORMALIZE
    def normalize(self):
        
#        self.dataset.train_normalized = []
#        self.dataset.validation_normalized = []
        
        for idx, signal in enumerate(np.transpose(self.dataset.train)): 
            mean = signal.mean()
            std = signal.std()
            norm = lambda x: (x - mean)/std
            
            self.dataset.train_normalized.append(norm(signal))
            self.dataset.validation_normalized.append(norm(np.transpose(self.dataset.validation)[idx,:]))
            self.dataset.test_normalized.append(norm(np.transpose(self.dataset.test)[idx,:]))
            
        self.dataset.train_normalized = np.transpose(self.dataset.train_normalized)   
        self.dataset.validation_normalized = np.transpose(self.dataset.validation_normalized)
        self.dataset.test_normalized = np.transpose(self.dataset.test_normalized)
#
#        for signal in np.transpose(self.dataset.train):
#            
#            norm = lambda x: (x - mean)/std
#            train_mean.append(mean)
#            train_std.append(std)
#            
#            self.dataset.train_normalized.append(norm(signal))
#        self.dataset.train_normalized = np.transpose(self.dataset.train_normalized)
        
#        for signal in np.transpose(self.dataset.validation):
#            norm = lambda x: (x - signal.mean())/signal.std()
#            self.dataset.validation_normalized.append(norm(signal))
#        self.dataset.validation_normalized = np.transpose(self.dataset.validation_normalized)
        
    # PLOT
    def plot(self, normalized=True,train = True, random = False):
        
        if normalized and train:
            data = self.dataset.train_normalized
            time = self.dataset.train_timing
    
        elif normalized and not train:
            data = self.dataset.validation_normalized
            time = self.dataset.validation_timing
        elif not normalized and train:
            data = self.dataset.train
            time = self.dataset.train_timing
        else:
            data = self.dataset.validation
            time = self.dataset.validation_timing
        
        for i in range(0, len(self.dataset.sensors)):
            plt.plot(time, np.transpose(data)[i])
            plt.ylabel("Signal: " + self.dataset.sensors[i])
            plt.xlabel("Time (days)")
            plt.show()

        


    
    
        
        
    
        
        
        