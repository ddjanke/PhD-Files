# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 14:32:49 2019

@author: ddjan
"""

import numpy as np
import ANN_Model as ann
import json
from sklearn.metrics import f1_score
import datetime
import time
import syncthing_test
from sklearn import datasets
import pandas as pd
import shared_simulate

activation = "ReLU4"
data = "grid"
save_file = "ReLU_thresh0_{}_{}.json".format(data, activation)

with open("/home/djanke3/Documents/computer_identity.json", 'r') as ci:
    computer_identity = json.load(ci)

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
    df = pd.read_csv("datasets/electrical_grid_stability.csv")
    #print(df.head())
    y = df["stabf"]
    y = (y == "unstable").astype(int)
    #y[y == "b"] = 0
    dataset.target = y.values
    #print(y.sum())

    X = df.drop(["stab", "stabf"], axis = 1)

    dataset.data = X.values

else:
    dataset = datasets.load_iris()

def combinations(values, n):
    return np.array(np.meshgrid(*([values] * n))).T.reshape(-1,n)

def count_trials(max_hl, values):
    return sum([len(values) ** hl for hl in range(1,max_hl + 1)])

def get_finished(filename, this_comp, other_comp):
    this_trials = shared_simulate.read_trials(filename, this_comp)
    other_trials = {} #shared_simulate.read_trials(filename, other_comp)
    finished = set(list(this_trials.keys()) + list(other_trials.keys()))
    #finished.remove("trial100_1200")
    return finished, this_trials

    
print("Generating Audio:")
def data_generator():
    X,y = dataset.data, dataset.target
    X -= X.mean(axis = 0)
    X /= X.std()
    yield 0,X,y

    
for i,X,y in data_generator():
    xpd = pd.DataFrame(X)
    corr = xpd.corr()
    num_features = X.shape[1]
    num_classes = y.max()
    if num_classes > 1: num_classes += 1
    print(X.shape, y.shape)
    
    
for activation in ["ReLU1", "ReLU2", "ReLU4", "Tanh", "CompTanh", "CompTanh2"]:

    save_file = "ReLU_thresh0_{}_{}.json".format(data, activation)
    
    this_comp = computer_identity["number"]
    other_comp = (this_comp + 1) % 2
    finished, trials = shared_simulate.get_finished(save_file, this_comp, 3)
    done = 0
    
    values = [3, 6, 9, 12, 15]
    tot_trials = 3 * len(values) ** 3 + len(values) ** 4 #count_trials(4, values)
    done = 0
    
    for hl in range(2,5):
        nn_shape = [num_features] + (hl - 1) * [12] + [6, 1]
    
        for lc in [nn_shape]:
            for r in range(50):
                if this_comp == 0: r = 49 - r
                trial_name = "trial" + str(nn_shape) + str(r)
                
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
                    noisy_features = np.random.normal(loc = 1.0, scale = 0.1, size = [1000, num_features])
                    noisy_weights_mult = np.random.normal(loc = 1, scale = 0.035, size = [1000, num_weights])
                    noisy_weights_add = np.random.normal(loc = 0, scale = 1 * 0.07, size = [1000, num_weights])
                    noisy_parameters = np.concatenate([noisy_features, noisy_weights_mult, noisy_weights_add], axis = 1)
                    #np.save("noisy_parameters.npy", noisy_parameters)
        
        
                ## Train and Predict ##################################################################################
                model = ann.ANNModelTorch(nn_shape, activation = activation)
                print("Training...")
                model.fit(data_generator, epochs = 10000, learning_rate = 0.002,
                          early_stop_train = True, verbose = 200, generator_kwargs={})
            
                    
                #input(model.state_dict())
                trial_accs = np.zeros((1, len(noisy_parameters)))
                print(("Testing..."), end = " ")
                ideal_acc = model.predict_accuracy(data_generator,
                                                   generator_kwargs = {})
                print(ideal_acc)
                for j,n in enumerate(noisy_parameters):
                    noisy_features = n[:num_features]
                    noisy_weights = n[num_features:]
                    model.add_noise(noisy_weights)
                    train_acc = model.predict_accuracy(data_generator,
                                                       feature_noise = noisy_features, generator_kwargs={})
                        
                    trial_accs[0, j] = train_acc
                    if (j+1) % 100 == 0: print(j+1, end = " ")
            
                
                print()
                del model
                
                trials[trial_name] = {"shape" :str(nn_shape),
                                      "accs" : {"train" : list(trial_accs[0, :])}
                                      }
                
                print()
                shared_simulate.write_trials(trials, save_file, this_comp)
            
