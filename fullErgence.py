# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 14:32:49 2019

@author: ddjan
"""
import os
import numpy as np
import ergence_feature_extraction as erg
import ANN_Model as ann
import matplotlib.pyplot as plt
import torch
import ray
from ray import tune
import gc
from scipy.stats import loguniform
import random
import json

# Initial settings
test = 1			# test trained NN on test sets?
iterations = 2000	# number of iterations for training NN
imax = 4*10*46	  # optional: total number of parameter loops
num_audio = len(os.listdir("/home/djanke3/Documents/Audio/DNS-Challenge/noisy"))
train_split = (0.6, 0.2, 0.2)

GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory

# List of learned parameters
config = {"tau" : loguniform.rvs(0.0001, 1, size = 1)[0],
          "spr" : loguniform.rvs(0.01, 100, size = 1)[0],
          "spf" : loguniform.rvs(0.01, 100, size = 1)[0],
          "nlr" : loguniform.rvs(0.01, 100, size = 1)[0],
          "nlf" : loguniform.rvs(0.01, 100, size = 1)[0],
          "spread" : random.choice([True, False])} #,
          #"hl" : tune.choice([1,2,3]),
          #"fls" : tune.choice(list(range(3,13))),
          #"sls" : tune.choice(list(range(3,13))),
          #"tls" : tune.choice(list(range(3,13)))}

param = {"tau" : 0.025,
          "spr" : 1,
          "spf" : 1,
          "nlr" : 0.1,
          "nlf" : 1,
          "spread" : True}


## Read all audio data ###################################################################################################
def find_num_epochs():
    kwargs = {"features" : ["diff", "n"], "bpa_params" : {"bands" : 9, "fmin" : 50}}
    efg = erg.AudioFeatureGenerator(train_split, 400, load_to_memory = True,
                                    data_fraction = 0.000001, **kwargs)
    model = ann.ANNModelTorch((efg.bpa.bands * len(kwargs["features"]), 18, 12, 6, 1))
    epoch_accs = np.zeros((2000,))
    for i in range(2000):
        print(i, end = " ")
        model.fit(efg.batch_audio_generator, epochs = 1,
                  batch_size = 1, verbose = False, pos_weight = efg.pos_weight)
        epoch_accs[i] = model.predict_accuracy(efg.batch_audio_generator)
        print(epoch_accs[i])
        
    plt.plot(epoch_accs)
    
    # Result is that no more than 200 Epochs are required


def find_train_size():
    kwargs = {"features" : ["diff", "n"]}
    epoch_accs = np.zeros((2, 2000))
    efg = erg.AudioFeatureGenerator(train_split, 400, load_to_memory = True, **kwargs)
    for i,frac in enumerate(np.logspace(-6, 0, 100)):
        efg.data_fraction = frac
        model = ann.ANNModelTorch((efg.bpa.bands * len(kwargs["features"]), 12, 6, 1))
        print(i, end = " ")
        model.fit(efg.batch_audio_generator, epochs = 1000,
                  batch_size = 1, verbose = False, pos_weight = efg.pos_weight)
        epoch_accs[:, i] = model.predict_accuracy(efg.batch_audio_generator)
        
        print(epoch_accs[:,i])
        if epoch_accs[0, i-1] * 0.9975 < epoch_accs[1, i-1] < epoch_accs[0, i-1] * 1.0025 and \
            epoch_accs[0, i] * 0.9975 < epoch_accs[1, i] < epoch_accs[0, i] * 1.0025: break
        
        del model
        x = np.logspace(-5, 0, 100)[:i + 1]
        plt.plot(x, epoch_accs[:, :i + 1].T)
        plt.show()
        
    x = np.logspace(-5, 0, 100)[:i + 1]
    epoch_accs = epoch_accs[:, :i + 1].T
        
    plt.plot(x, epoch_accs)
    plt.show()
    
    # Result is that only 0.02% of the training samples are needed to closely
    # represent the entire training dataset
    

    
  
#@ray.remote(num_cpus = 6, num_gpus = 1)
def feature_random_search(samples, epochs):
    zero_fill = len(str(samples - 1))
    acc_json = {}
    for i in range(samples):
        config = {"envelope_tau" : loguniform.rvs(0.0001, 1, size = 1)[0],
              "sp_decay_rise" : loguniform.rvs(0.01, 100, size = 1)[0],
              "sp_decay_fall" : loguniform.rvs(0.01, 100, size = 1)[0],
              "nl_decay_rise" : loguniform.rvs(0.01, 100, size = 1)[0],
              "nl_decay_fall" : loguniform.rvs(0.01, 100, size = 1)[0],
              "spread" : random.choice([True, False])}
        
        trial_name = "trial" + str(i).zfill(zero_fill)
        
        print(trial_name)
    
        train_kwargs = {"features" : ["diff", "n"], "data_fraction" : 0.0002, "fa_params" : config}
        val_kwargs = {"features" : ["diff", "n"], "data_fraction" : 1, "fa_params" : config}
        
        efg = erg.AudioFeatureGenerator(train_split, 400, load_to_memory = True, **train_kwargs)
        efgv = erg.AudioFeatureGenerator(train_split, 400, load_to_memory = True, **val_kwargs)
        model = ann.ANNModelTorch((efg.bpa.bands * len(train_kwargs["features"]), 12, 6, 1))
        
        trial_accs = np.zeros((3, epochs))
        for j in range(epochs):
            if j == 1: print("Training", end = " ")
            model.fit(efg.batch_audio_generator, epochs = 1, batch_size = 1, verbose = False)
            train_acc = model.predict_accuracy(efg.batch_audio_generator)
            val_acc = model.predict_accuracy(efgv.batch_audio_generator,
                                             generator_kwargs = {"mode" : "train", "data_set" : "val"})
            trial_accs[:2, j] = train_acc
            trial_accs[2, j] = val_acc
            if ((j + 1) % 10) == 0: print(int((j + 1)/10), end = " ")
            
        print('\n')
        
        acc_json[trial_name] = {"config" : config,
                                "accs" : {"train" : list(trial_accs[0, :]),
                                          "train_val" : list(trial_accs[1, :]),
                                          "val" : list(trial_accs[2, :])},
                                "max_acc" : list(trial_accs.max(axis = 1)),
                                "avg_acc" : list(np.mean(trial_accs[:, int(epochs/2):], axis = 1))}
    
        with open("random_feature_trials0.json", 'w') as jf:
            json.dump(acc_json, jf)




#find_num_epochs()
#find_train_size()

feature_random_search(samples = 400, epochs = 400)
#Result: tau = 0.025, spr = 56, spf = 20, nlr = 1, nlf = 20, spread = False