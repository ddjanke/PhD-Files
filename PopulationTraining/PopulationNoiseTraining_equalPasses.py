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
import shared_simulate
import math

# Initial settings
train_split = (0.6, 0.2, 0.2)
total_files = 400
GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory
features = ["diff", "n"]
bands = 6

test_size = 1000

save_file = "PopulationNoiseResults_growingLR_equalPasses.json"
const = False
data = "audio"

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
nn_shape = (num_features, 12, 9, 6, 3, 1)

efg = erg.AudioFeatureGenerator(train_split, data_fraction = 0.0002, **kwargs)

def get_finished(filename, this_comp, total):
    this_trials = shared_simulate.read_trials(filename, this_comp, 1)
    other_trials = {} #shared_simulate.read_trials(filename, (this_comp + 1) % total, total - 1)
    finished = set(list(this_trials.keys()) + list(other_trials.keys()))
    #finished.remove("trial100_1200")
    return finished, this_trials

if data == "cancer":
    dataset = datasets.load_breast_cancer()
    X = dataset.data
    y = dataset.target
    
    xpd = pd.DataFrame(X)
    corr = xpd.corr()
    keep = []
    delete = []
    for i in range(X.shape[1]):
        if i in delete: continue
        keep.append(i)
        for j in range(X.shape[1]):
            if j in keep: continue
            if corr.loc[j,i] > 0.8:
                delete.append(j)
    xpd = xpd[keep]
    dataset.data = xpd.values

    
elif data == "grid":
    dataset = datasets.load_iris()
    df = pd.read_csv("electrical_grid_stability.csv")
    #print(df.head())
    y = df["stabf"]
    y = (y == "unstable").astype(int)
    #y[y == "b"] = 0
    dataset.target = y.values
    #print(y.sum())

    X = df.drop(["stab", "stabf"], axis = 1)

    dataset.data = X.values
    
print("Generating Audio:")
for g in efg.batch_audio_generator(mode = "train", data_set = "train"): continue
for g in efg.batch_audio_generator(mode = "train", data_set = "val"): continue

this_comp = computer_identity["number"]
other_comp = (this_comp + 1) % 2
finished, trials = get_finished(save_file, this_comp, other_comp)
done = 0

try:
    noisy_parameters = np.load("pop_noisy_parameters_large.npy")
except:
    num_weights = ann.count_synapses(nn_shape)
    noisy_features = np.random.normal(loc = 1.0, scale = 0.1, size = [test_size, num_features])
    noisy_weights_mult = np.random.normal(loc = 1, scale = 0.035, size = [test_size, num_weights])
    noisy_weights_add = np.random.normal(loc = 0, scale = 1 * 0.07, size = [test_size, num_weights])
    noisy_parameters = np.concatenate([noisy_features, noisy_weights_mult, noisy_weights_add], axis = 1)
    np.save("pop_noisy_parameters_large.npy", noisy_parameters)

print(finished)
print()


pop_size = [1, 5, 10, 20, 40]
passes_min = 200
passes_max = 4000

for p in pop_size[::1]: #[::int((this_comp - 0.5) * 2)]
    
    epochs = passes_min // p
    
    for j in range(5):
        #j = 4 - j
        model = ann.ANNModelTorch(nn_shape, activation = "Tanh")
        for i in range(passes_max // passes_min + 1):
            trial_name = "trial{}_{}_{}".format(p, (i + 1) * passes_min, j)
            done += 1
            
            #finished, _ = get_finished(save_file, this_comp, other_comp)
            if trial_name in finished: continue
            trials[trial_name] = this_comp
            shared_simulate.write_trials(trials, save_file, this_comp)
            
            if trial_name in finished: continue
                    
            print("\n" + trial_name,"({} of {})".format(done, len(pop_size) * passes_max // passes_min * 5), end = ' ')
            
            
            ## Train and Predict ##################################################################################
            print("Training...")
            lr = 0.002 if (p == 1) else max(0.002, 0.002 * p / (i + 1))
            model.population_fit(efg.batch_audio_generator,
                                 epochs = epochs,
                                 population = p,
                                 learning_rate = lr, #max(0.001, 0.002 * 10 / (i + 1)),
                                 early_stop_train = False,
                                 verbose = 10,
                                 noise_at_input = False)
                
            #input(model.state_dict())
            trial_accs = np.zeros((2, test_size))
        
            train_acc = model.population_accuracy(efg.batch_audio_generator,
                                                  generator_kwargs = {"mode" : "train"},
                                                  population = test_size,
                                                  feature_noise = noisy_parameters[:, :num_features],
                                                  average = False,
                                                  weight_noise = noisy_parameters[:, num_features:])
            
            val_acc = model.population_accuracy(efg.batch_audio_generator,
                                                generator_kwargs = {"mode" : "train",
                                                                    "data_set" : "val"},
                                                population = test_size,
                                                feature_noise = noisy_parameters[:, :num_features],
                                                average = False,
                                                weight_noise = noisy_parameters[:, num_features:])
                    
            
            trials[trial_name] = {"shape" :str(nn_shape),
                                  "accs" : {"train" : train_acc,
                                            "val" : val_acc}}
            
            print()
            shared_simulate.write_trials(trials, save_file, this_comp)
        
        del model
        
