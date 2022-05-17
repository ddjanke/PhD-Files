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
import datetime
import time
import syncthing_test
import shared_simulate

# Initial settings
train_split = (0.6, 0.2, 0.2)
total_files = 400
GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory
features = ["diff", "n"]
bands = 6

save_file = "NetworkShapeAnalysis_noNoise.json"
activation = "CompTanh"
sparsity = 1

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

efg = erg.AudioFeatureGenerator(train_split, data_fraction = 0.0002, **kwargs)

def combinations(values, n):
    return np.array(np.meshgrid(*([values] * n))).T.reshape(-1,n)

def sparse_valid(nodes, sparsity):
    return all([n <= (sparsity - 1) * nodes[i + 1] for i,n in enumerate(nodes[:-2])])

def count_trials(max_hl, values):
    return sum([len(values) ** hl for hl in range(1,max_hl + 1)])

def get_finished(filename, this_comp, other_comp):
    this_trials = shared_simulate.read_trials(filename, this_comp)
    other_trials = {} #shared_simulate.read_trials(filename, other_comp)
    finished = set(list(this_trials.keys()) + list(other_trials.keys()))
    #finished.remove("trial100_1200")
    return finished, this_trials
    
print("Generating Audio:")
for g in efg.batch_audio_generator(mode = "train", data_set = "train", load=True): continue
for g in efg.batch_audio_generator(mode = "train", data_set = "val", load=True): continue

this_comp = computer_identity["number"]
other_comp = (this_comp + 1) % 2
finished, trials = shared_simulate.get_finished(save_file, this_comp, 3)
done = 0

values = [3, 6, 9, 12, 15]
tot_trials = 3 * len(values) ** 3 + len(values) ** 4 #count_trials(4, values)

for hl in range(2,3): #,5):
    #hl = 5 - hl
    layer_combos = combinations(values, hl)
    #print(layer_combos)
    if sparsity != 1:
        layer_combos = [lc for lc in layer_combos if sparse_valid(lc)]
        
    print("Simulating network with {} hidden layers, {} shape combinations" \
          .format(hl, len(layer_combos)))
    repeats = int(max(len(values) ** 3 / len(layer_combos), 1))
    
    for lc in layer_combos[::1]:
        for r in range(repeats):
            num_features = len(features) * bands
            nn_shape = (num_features, *lc, 1)
            
            trial_name = "trial" + str(nn_shape)
            if repeats > 1: trial_name += str(r)
            
            # Check if this trial already done or running
            done += 1
            
            finished, _ = shared_simulate.get_finished(save_file, this_comp, 3)
            if trial_name in finished: continue
            trials[trial_name] = this_comp
            shared_simulate.write_trials(trials, save_file, this_comp)
            
            if trial_name in finished: continue
            
            print("\n" + trial_name,"({} of {})".format(done, tot_trials), end = ' ')
    
            try:
                raise TypeError()
                noisy_parameters = np.load("noisy_parameters.npy")
            except:
                num_weights = ann.count_synapses(nn_shape)
                noisy_parameters = np.concatenate((np.ones((1,num_features + num_weights)),\
                                                   np.zeros((1,num_weights))), axis=1)
                #noisy_features = np.random.normal(loc = 1.0, scale = 0.1, size = [1000, num_features])
                #noisy_weights_mult = np.random.normal(loc = 1, scale = 0.035, size = [1000, num_weights])
                #noisy_weights_add = np.random.normal(loc = 0, scale = 1 * 0.07, size = [1000, num_weights])
                #noisy_parameters = np.concatenate([noisy_features, noisy_weights_mult, noisy_weights_add], axis = 1)
                #np.save("noisy_parameters.npy", noisy_parameters)
    
    
            ## Train and Predict ##################################################################################
            model = ann.ANNModelTorch(nn_shape, activation = activation)
            print("Training...")
            model.fit(efg.batch_audio_generator, epochs = 2000, learning_rate = 0.002,
                      early_stop_train = True, verbose = 200)
        
                
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
        
            
            print()
            del model
            
            trials[trial_name] = {"shape" :str(nn_shape),
                                  "accs" : {"train" : list(trial_accs[0, :]),
                                            "val" : list(trial_accs[1, :])}}
            
            print()
            shared_simulate.write_trials(trials, save_file, this_comp)
        
