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
from scipy.stats import loguniform
import random
import json
from sklearn.metrics import f1_score

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

fa_params = {"envelope_tau" : 0.025,
             "sp_decay_rise" : 56,
             "sp_decay_fall" : 20,
             "nl_decay_rise" : 1,
             "nl_decay_fall" : 20,
             "spread" : False}


## Read all audio data ###################################################################################################

def find_num_epochs():
    kwargs = {"features" : ["diff", "n"], "bpa_params" : {"bands" : 6}, "fa_params" : fa_params}
    efg = erg.AudioFeatureGenerator(train_split, 400, load_to_memory = True,
                                    data_fraction = 0.0002, **kwargs)
    model = ann.ANNModelTorch((efg.fa.bpa.bands * len(kwargs["features"]), 12, 6, 1))
    epoch_accs = np.zeros((2000,2))
    es = ann.myEarlyStop(patience = 50, delta = 0.0001)
    for i in range(2000):
        print(i, end = " ")
        model.fit(efg.batch_audio_generator, epochs = 1, batch_size = 1,
                  verbose = False, pos_weight = efg.pos_weight, early_stop_train = True)
        epoch_accs[i, :] = model.predict_accuracy(efg.batch_audio_generator)
        #es(epoch_accs[i,1], model)
        #if es.stop: break
        print(epoch_accs[i, :])
        plt.plot(epoch_accs[:i + 1, :])
        plt.show()
        
    plt.plot(epoch_accs[: i + 1, :])
    plt.show()
    
    # Result is that no more than 200 Epochs are required


def find_train_size():
    new_random = False
    
    kwargs = {"features" : ["diff", "n"], "bpa_params" : {"bands" : 6},
              "fa_params" : fa_params}
    epoch_accs = np.zeros((2000, 4))
    efg = erg.AudioFeatureGenerator(train_split, 400, load_to_memory = True, **kwargs)
    try:
        efg.X["train"] = np.load("X{}all.npy".format("train"))
        efg.y["train"] = np.load("y{}all.npy".format("train"))
        efg.X["val"] = np.load("X{}all.npy".format("val"))
        efg.y["val"] = np.load("y{}all.npy".format("val"))
        print("all loaded")
    except:
        pass
    
    x = np.logspace(-6, -2, 40)
    for repeats in range(10):
        for i,frac in enumerate(x):
            efg.data_fraction = frac
            model = ann.ANNModelTorch((efg.fa.bpa.bands * len(efg.features), 12, 6, 3, 1), device = "cpu")
            print(i, frac, end = " ")
            loss = 0.5
            goal_loss = 0.3
            while not model.es.stop:
                loss = model.fit(efg.batch_audio_generator, epochs=10, learning_rate = 0.001 ** (goal_loss/loss),
                          early_stop_train = True, verbose=200,
                          generator_kwargs={"mode":"train", "new_random":new_random})
                
            epoch_accs[i, :2] = model.predict_accuracy(efg.batch_audio_generator,
                                                       generator_kwargs = {"mode" :"eval",
                                                                           "new_random":new_random})
            epoch_accs[i, 2:] = model.predict_accuracy(efg.batch_audio_generator,
                                                       generator_kwargs = {"mode" :"eval",
                                                                           "data_set" : "val"})
            #if i > 1: break
            print(epoch_accs[i, :])
            
            range_start = max(i-9, 0)
            if max(np.absolute(epoch_accs[range_start : i + 1,1] \
                               - epoch_accs[range_start : i + 1, 0]).max(), 
                   np.absolute(epoch_accs[range_start : i + 1,3] \
                               - epoch_accs[range_start : i + 1, 2]).max()) < 0.0025:
                break
            
            del model
            x_part = x[:i + 1]
            plt.plot(x_part, epoch_accs[:i + 1, :])
            plt.xscale("log")
            plt.legend(["train", "train_all", "val", "val_all"], loc = "lower right")
            plt.show()
            
        x_part = x[:i + 1]
        epoch_accs = epoch_accs[:i + 1, :]
        
        data = np.vstack((x_part,epoch_accs.T))
        np.savetxt("find_train_size{}.csv".format(repeats), data, delimiter = ',')
            
        plt.plot(x_part, epoch_accs)
        plt.legend(["train", "train_all", "val", "val_all"], loc = "lower right")
        plt.xscale("log")
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
    
        train_kwargs = {"features" : ["diff", "n"], "data_fraction" : 0.0002,
                        "bpa_params" : {"bands" :6}, "fa_params" : config}
        val_kwargs = {"features" : ["diff", "n"], "data_fraction" : 0.0002,
                        "bpa_params" : {"bands" :6}, "fa_params" : config}
        
        efg = erg.AudioFeatureGenerator(train_split, 400, load_to_memory = True, **train_kwargs)
        efgv = erg.AudioFeatureGenerator(train_split, 400, load_to_memory = True, **val_kwargs)
        model = ann.ANNModelTorch((efg.fa.bpa.bands * len(train_kwargs["features"]), 12, 6, 1))
        
        model.fit(efg.batch_audio_generator, epochs = 2000, batch_size = 1,
                  verbose = False, early_stop_train = True)
        train_acc = model.predict_accuracy(efg.batch_audio_generator,
                                           generator_kwargs = {"mode" : "train"})
        val_acc = model.predict_accuracy(efgv.batch_audio_generator,
                                         generator_kwargs = {"mode" : "train",
                                                             "data_set" : "val"})
        for i,X,y in efg.batch_audio_generator(mode = "train"):
            train_f1 = f1_score(y, model.predict(X).cpu().detach().numpy())
        for i,X,y in efgv.batch_audio_generator(mode = "train", data_set = "val"):
            val_f1 = f1_score(y, model.predict(X).cpu().detach().numpy())
            
        max_weight = 0
        for module in model.model:
            if 'weight' in module.state_dict():
                max_weight = max([np.absolute(module.weight.cpu().detach().numpy()).max(),
                                 np.absolute(module.bias.cpu().detach().numpy()).max(),
                                 max_weight])
            
        print('\n')
        
        acc_json[trial_name] = {"config" : config,
                                "accs" : {"train" : train_acc,
                                          "val" : val_acc},
                                "f1" : {"train" : float(train_f1),
                                          "val" : float(val_f1)},
                                "max_weight" : float(max_weight)
                                }
        
        #print(acc_json)
    
        with open("random_feature_trials0.json", 'w') as jf:
            json.dump(acc_json, jf, indent = 4)

        del model
        del efg
        del efgv


#find_num_epochs()
find_train_size()

#feature_random_search(samples = 400, epochs = 2000)
#Result: tau = 0.025, spr = 56, spf = 20, nlr = 1, nlf = 20, spread = False