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
import json
from sklearn.metrics import f1_score

# Initial settings
train_split = (0.6, 0.2, 0.2)
total_files = 400
GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory
features = ["diff", "n"]
bands = 6

# List of learned parameters
fa_params = {"envelope_tau" : 0.025,
             "sp_decay_rise" : 25,
             "sp_decay_fall" : 2,
             "nl_decay_rise" : 13.5,
             "nl_decay_fall" : 45.5,
             "spread" : False}

bpa_params = {"bands" : bands}

nn_shape = (bands * len(features), 15, 9, 3, 1)

kwargs = {"features" : features,
          "total_files" : total_files,
          "load_to_memory" : True,
          "fa_params" : fa_params,
          "bpa_params" : bpa_params}

efg = erg.AudioFeatureGenerator(train_split, data_fraction = 0.0002, **kwargs)

num_features = len(features) * bands
try:
    raise TypeError()
    noisy_parameters = np.load("noisy_parameters.npy")
except:
    num_weights = ann.count_synapses(nn_shape)
    noisy_features = np.random.normal(loc = 1.0, scale = 0.1, size = [1000, num_features])
    noisy_weights_mult = np.random.normal(loc = 1, scale = 0.035, size = [1000, num_weights])
    noisy_weights_add = np.random.normal(loc = 0, scale = 1 * 0.07, size = [1000, num_weights])
    noisy_parameters = np.concatenate([noisy_features, noisy_weights_mult, noisy_weights_add], axis = 1)
    np.save("noisy_parameters.npy", noisy_parameters)


## Train and Predict ##########################################################################################

do_values = np.linspace(0, 1, 11)
zero_fill = len(str(len(do_values) - 1))

try:
    #raise TypeError()
    with open("Dropout_w_Noise.json", 'r') as jf:
        trials = json.load(jf)
    done = [int(t[-zero_fill:]) for t in trials.keys()]
    skip_to = max(done) + 1
except:
    trials = {}
    skip_to = 0


for nn_shape in [(bands * len(features), 15, 9, 3, 1),
                 (bands * len(features), 12, 6, 1),
                 (bands * len(features), 9, 1)]:
    try:
        raise TypeError()
        noisy_parameters = np.load("noisy_parameters.npy")
    except:
        num_weights = ann.count_synapses(nn_shape)
        noisy_features = np.random.normal(loc = 1.0, scale = 0.1, size = [1000, num_features])
        noisy_weights_mult = np.random.normal(loc = 1, scale = 0.035, size = [1000, num_weights])
        noisy_weights_add = np.random.normal(loc = 0, scale = 1 * 0.07, size = [1000, num_weights])
        noisy_parameters = np.concatenate([noisy_features, noisy_weights_mult, noisy_weights_add], axis = 1)
        np.save("noisy_parameters.npy", noisy_parameters)
        
    for i,do in enumerate(do_values):
        for j in range(5):
            if i < skip_to: continue
            
            print("DO value {}: {}".format(i, do))
            trial_name = "trial" + str(j) + str(nn_shape) + str(i).zfill(zero_fill)
            model = ann.ANNModelTorch(nn_shape, dropout = do)
            print("Training...")
            model.fit(efg.batch_audio_generator, epochs = 800, learning_rate = 0.002,
                      early_stop_train = False)
            
            
            #input(model.state_dict())
            trial_accs = np.zeros((2, len(noisy_parameters)))
            print(("Testing..."), end = " ")
            for j,n in enumerate(noisy_parameters):
                noisy_features = n[:num_features]
                noisy_weights = n[num_features:]
                model.add_noise(noisy_weights)
                train_acc = model.predict_accuracy(efg.batch_audio_generator,
                                                   feature_noise = noisy_features,
                                                   generator_kwargs = {"mode" : "train"})
                val_acc = model.predict_accuracy(efg.batch_audio_generator,
                                                 feature_noise = noisy_features,
                                                 generator_kwargs = {"mode" : "train",
                                                                     "data_set" : "val"})
                    
                trial_accs[0, j] = train_acc
                trial_accs[1, j] = val_acc
                if (j+1) % 100 == 0: print(j+1, end = " ")
                
            trials[trial_name] = {"dropout" : do,
                                  "nodes" : nn_shape,
                                  "accs" : {"train" : list(trial_accs[0, :]),
                                            "val" : list(trial_accs[1, :])}}
            
            print()
            del model
            with open("Dropout_w_Noise.json", 'w') as jf:
                    json.dump(trials, jf)
        
