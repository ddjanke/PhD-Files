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
config = {"tau" : tune.loguniform(0.0001, 1),
          "spr" : tune.loguniform(0.01, 100),
          "spf" : tune.loguniform(0.01, 100),
          "nlr" : tune.loguniform(0.01, 100),
          "nlf" : tune.loguniform(0.01, 100),
          "spread" : tune.choice([True, False])} #,
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

data = np.loadtxt("data_banknote_authentication.txt", delimiter = ",")[152:]
np.random.shuffle(data)
y = data[152:,-1]
X = data[152:,:-1]
print(X.shape)
print(y.sum(), y.mean())

def data_gen():
    data = np.loadtxt("data_banknote_authentication.txt", delimiter = ",")[152:]
    np.random.shuffle(data)
    y = data[:,-1]
    X = data[:,:-1]
    for i in range(4):
        yield (i, X[304*i : 304 * (i + 1), :], y[304*i : 304 * (i + 1)])


## Read all audio data ###################################################################################################
def find_num_epochs():
    #kwargs = {"features" : ["diff", "n"]}
    #efg = erg.AudioFeatureGenerator(train_split, 400, load_to_memory = True, **kwargs)
    model = ann.ANNModelTorch((4, 12, 6, 1))
    epoch_accs = np.zeros((2000,))
    for i in range(2000):
        print(i, end = " ")
        model.fit(data_gen, epochs = 1,
                  batch_size = 1, verbose = False, pos_weight = 1)
        epoch_accs[i] = model.predict_accuracy(data_gen)
        print(epoch_accs[i])
        
    plt.plot(epoch_accs)
    
    # Result is that no more than 200 Epochs are required


def find_train_size():
    kwargs = {"features" : ["diff", "n"]}
    efg_all = erg.AudioFeatureGenerator(train_split, 400, load_to_memory = True, **kwargs)
    epoch_accs = np.zeros((2, 2000))
    for i,frac in enumerate(np.logspace(-5, 0, 100)):
        efg_t = erg.AudioFeatureGenerator(train_split, 400, load_to_memory = True,
                                          train_fraction = frac, **kwargs)
        model = ann.ANNModelTorch((efg_all.bpa.bands * len(kwargs["features"]), 12, 6, 1))
        print(i, end = " ")
        model.fit(efg_t.batch_audio_generator, epochs = 200,
                  batch_size = 1, verbose = False, pos_weight = efg_t.pos_weight)
        epoch_accs[0, i] = model.predict_accuracy(efg_t.batch_audio_generator)
        epoch_accs[1, i] = model.predict_accuracy(efg_all.batch_audio_generator)
        
        print(epoch_accs[:,i])
        if epoch_accs[0, i-1] * 0.999 < epoch_accs[1, i-1] < epoch_accs[0, i-1] * 1.001 and \
            epoch_accs[0, i] * 0.999 < epoch_accs[1, i] < epoch_accs[0, i] * 1.001: break
        
        del efg_t
        del model
        x = np.logspace(-5, 0, 100)[:i + 1]
        plt.plot(x, epoch_accs[:, :i + 1].T)
        plt.show()
        
    x = np.logspace(-5, 0, 100)[:i + 1]
    epoch_accs = epoch_accs[:, :i + 1].T
        
    plt.plot(x, epoch_accs)
    plt.show()
    
    # Result is that no more than 200 Epochs are required
    
find_num_epochs()
#find_train_size()

'''
def find_train_size():
    efg = erg.AudioFeatureGenerator(train_split, 400)
    model = ann.ANNModelTorch((12, 12, 6, 1))
    torch.cuda.empty_cache()
    for i in range(100):
        model.fit(efg.batch_audio_generator, epochs = 1, batch_size = 1, verbose = False)
        acc = model.predict_accuracy_all(audio_features, audio_labels).mean()
        tune.report(mean_accuracy = acc)
        
    del audio_features
    del audio_labels
    del Xtemp
    del ytemp
    del model 
    
  
#@ray.remote(num_cpus = 6, num_gpus = 1)
def feature_random_search(param):
    efg = erg.AudioFeatureGenerator(train_split, 400, fa_params = param)

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
'''