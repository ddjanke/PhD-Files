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

save_file = "SerialNoiseResults.json"

with open("/home/djanke3/Documents/computer_identity.json", 'r') as ci:
    computer_identity = json.load(ci)

# List of learned parameters
fa_params = {"envelope_tau" : 0.025,
             "sp_decay_rise" : 25,
             "sp_decay_fall" : 2,
             "nl_decay_rise" : 13.5,
             "nl_decay_fall" : 45.5,
             "spread" : False}

bpa_params = {"bands" : bands}

kwargs = {"features" : features,
          "total_files" : total_files,
          "load_to_memory" : True,
          "fa_params" : fa_params,
          "bpa_params" : bpa_params}

num_features = len(features) * bands
nn_shape = (num_features, 12, 6, 1)

efg = erg.AudioFeatureGenerator(train_split, data_fraction = 0.0002, **kwargs)

def load_previous():
    try:
        with open(save_file, 'r') as jf:
            trials = json.load(jf)
    except:
        trials = {}
    return trials
    
print("Generating Audio:")
for g in efg.batch_audio_generator(mode = "train", data_set = "train"): continue
for g in efg.batch_audio_generator(mode = "train", data_set = "val"): continue

trials = load_previous()
tot_trials = 100
done = 0
for t in range(tot_trials):
    trial_name = "trial" + str(t)
    done += 1
    this_trial = trials.get(trial_name, {})
            
    if "acc" in this_trial: continue
            
    print("\n" + trial_name,"({} of {})".format(done, tot_trials), end = ' ')
    
    try:
        raise TypeError()
        noisy_parameters = np.load("noisy_parameters.npy")
    except:
        num_weights = ann.count_synapses(nn_shape)
        noisy_features = np.random.normal(loc = 1.0, scale = 0.1, size = [1000, num_features])
        noisy_weights_mult = np.random.normal(loc = 1, scale = 0.035, size = [1000, num_weights])
        noisy_weights_add = np.random.normal(loc = 0, scale = 1 * 0.07, size = [1000, num_weights])
        noisy_parameters = np.concatenate([noisy_features, noisy_weights_mult, noisy_weights_add], axis = 1)
        #np.save("noisy_parameters.npy", noisy_parameters)
    
    
    ## Train and Predict ##################################################################################
    model = ann.ANNModelTorch(nn_shape, activation = "Tanh")
    print("Training...")
    model.fit(efg.batch_audio_generator, epochs = 2000,
              learning_rate = 0.002, early_stop_train = True)
        
    #input(model.state_dict())
    trial_accs = np.zeros((2, len(noisy_parameters)))
    trial_f1s = np.zeros((2, len(noisy_parameters)))
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
        for i,X,y in efg.batch_audio_generator(mode = "train"):
            trial_f1s[0,j] = f1_score(y, model.predict(X).cpu().detach().numpy())
        for i,X,y in efg.batch_audio_generator(mode = "train", data_set = "val"):
            trial_f1s[1,j] = f1_score(y, model.predict(X).cpu().detach().numpy())
            
        trial_accs[0, j] = train_acc
        trial_accs[1, j] = val_acc
        if (j+1) % 100 == 0: print(j+1, end = " ")
    
    trials = load_previous()
    trials[trial_name] = {"shape" :str(nn_shape),
                          "accs" : {"train" : list(trial_accs[0, :]),
                                    "val" : list(trial_accs[1, :])},
                          "f1s" : {"train" : list(trial_f1s[0, :]),
                                    "val" : list(trial_f1s[1, :])}}
    
    print()
    del model
    with open(save_file, 'w') as jf:
            json.dump(trials, jf)
        
