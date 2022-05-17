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

# Initial settings
test = 1			# test trained NN on test sets?
iterations = 2000	# number of iterations for training NN
imax = 4*10*46	  # optional: total number of parameter loops
num_audio = len(os.listdir("/home/djanke3/Documents/Audio/DNS-Challenge/noisy"))
train_split = (0.6, 0.2, 0.2)

GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory

# List of learned parameters
config = {"tau" : tune.loguniform(0.001, 1),
          "spr" : tune.loguniform(0.01, 100),
          "spf" : tune.loguniform(0.01, 100),
          "nlr" : tune.loguniform(0.01, 100),
          "nlf" : tune.loguniform(0.01, 100),
          "spread" : tune.choice([True, False])} #,
          #"hl" : tune.choice([1,2,3]),
          #"fls" : tune.choice(list(range(3,13))),
          #"sls" : tune.choice(list(range(3,13))),
          #"tls" : tune.choice(list(range(3,13)))}

param = {"tau" : 0.25,
          "spr" : 1,
          "spf" : 1,
          "nlr" : 0.1,
          "nlf" : 1,
          "spread" : False}


## Read all audio data ###################################################################################################
    
def calc_batch_size():
    tf = train_files[0]
    lf = train_label_files[0]
    labels = np.loadtxt(ad.label_dir + lf, delimiter = ',')
    sr, audio = ad.read_audio(ad.noisy_dir + tf)
    
    bpa = erg.BandpassArray(audio = audio, sample_rate = sr, bands = 6)
    bpa.expand(5, 5)
    fa = erg.FeatureArray(bpa, envelope_tau = param["tau"],
                          decay_rate_unit = 1,
                          sp_decay_rise = param["spr"],
                          sp_decay_fall = param["spf"],
                          nl_decay_rise = param["nlr"],
                          nl_decay_fall = param["nlf"],
                          spread = param["spread"])
    fa.bpa.multithread_filter()
    fa.multithread_features()
    fa.truncate()
    feature_label_bytes = fa.X.nbytes + labels.nbytes
    total_bytes = int(train_split[0] * len(train_files)) * feature_label_bytes
    
    num_batches = int(total_bytes / (GPU_MEMORY * 0.4)) + 1
    #print(fa.X.nbytes, labels.nbytes)
    i = 0
    while (len(train_files) % (num_batches + i) != 0): i += 1
    num_batches += i
    return int(len(train_files) / num_batches)
  
#@ray.remote(num_cpus = 6, num_gpus = 1)
def feature_random_search(param):
    #print(ray.get_gpu_ids())
    audio_features = []
    audio_labels = []
    Xtemp = None
    ytemp = None
    batch_size = calc_batch_size()
    
    for i,(tf, lf) in enumerate(zip(train_files, train_label_files)):
        labels = np.loadtxt(ad.label_dir + lf, delimiter = ',')
        sr, audio = ad.read_audio(ad.noisy_dir + tf)
        
        bpa = erg.BandpassArray(audio = audio, sample_rate = sr, bands = 6)
        bpa.expand(5, 5)
        fa = erg.FeatureArray(bpa, envelope_tau = param["tau"],
                              decay_rate_unit = 1,
                              sp_decay_rise = param["spr"],
                              sp_decay_fall = param["spf"],
                              nl_decay_rise = param["nlr"],
                              nl_decay_fall = param["nlf"],
                              spread = param["spread"])
        fa.bpa.multithread_filter()
        fa.multithread_features()
        fa.truncate()
        if (i+1) % 100 == 0 : print(i + 1)
        
        if Xtemp is None:
            Xtemp = fa.X.copy()[:,::2]
            ytemp = labels.copy()[::2]
        else:
            Xtemp = np.append(Xtemp, fa.X[:,::2], axis = 1)
            ytemp = np.append(ytemp, labels[::2])
        #print(Xtemp.nbytes, ytemp.nbytes)
        if (i+1) % batch_size * 2 == 0 or (i + 1 == len(train_files)):
            audio_features += [Xtemp.T.copy()]
            audio_labels += [ytemp.T.copy()]
            Xtemp = None
            ytemp = None
            
        del labels
        del audio
        del bpa
        del fa
            
            
    model = ann.ANNModelTorch((12, 12, 6, 1))
    torch.cuda.empty_cache()
    for i in range(100):
        model.fit(audio_features, audio_labels, epochs = 1, batch_size = 1, verbose = False)
        acc = model.predict_accuracy_all(audio_features, audio_labels).mean()
        tune.report(mean_accuracy = acc)
        
    del audio_features
    del audio_labels
    del Xtemp
    del ytemp
    del model

ray.shutdown()
ray.init(num_cpus = 12, num_gpus = 2)
#feature_random_search(param)
analysis = tune.run(feature_random_search,
                    num_samples = 1000,
                    config=config,
                    resources_per_trial = {"cpu" : 6, "gpu" : 1},
                    verbose = 2)