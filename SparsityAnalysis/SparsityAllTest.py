import os
import numpy as np
import ergence_feature_extraction as erg
import ann_sparse as ann
import matplotlib.pyplot as plt
import torch
import json
from sklearn.metrics import f1_score
import datetime
import time
import syncthing_test
import shared_simulate
import NN_optimize

# Initial settings
train_split = (0.6, 0.2, 0.2)
total_files = 400
GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory
features = ["diff", "n"]
bands = 6
num_features = len(features) * bands

dataset = ["voice", "power"][1]
activation = "CompTanh"
save_file = "SparsityTest_" + dataset + activation + "_HL{}_{}_.json"
sparsity = 3

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

if dataset == "voice":
    efg = erg.AudioFeatureGenerator(train_split, data_fraction = 0.0002, **kwargs)
    
    print("Generating Audio:")
    for g in efg.batch_audio_generator(mode = "train", data_set = "train", save=True, load=True): continue
    for g in efg.batch_audio_generator(mode = "train", data_set = "val", save=True, load=True): continue
    datagen = efg.batch_audio_generator
    train_kwargs = {"mode" : "train"}
else:
    datagen = NN_optimize.data_generator
    train_kwargs = {}

this_comp = computer_identity["number"]
done = 0

if __name__ == "__main__":
    
    accs = []
    shapes = ([12, 12, 12, 6, 3, 1],[12,12,6,3,1])
    sparsity_type = {"prune":[False,False],
                     "graft":[False,True],
                     "random":[True,False]}
    
    first_check = True
    for nn_shape in shapes:
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
    
        for strategy in ("all", "allsoft", "once")[::1]:
            s = strategy if strategy == "allsoft" else "all"
            filename = save_file.format(len(nn_shape) - 2, s)
            first_check = True
            print(filename)
            finished, trials = shared_simulate.get_finished(filename, this_comp, 3)
            for mask_rate in [10, 20, 50, 100]:
                if strategy == "once":
                    mask_start = mask_rate * 10
                    mask_rate = float("inf")
                else:
                    if mask_rate > 20: continue
                    mask_start = 0
                    #if strategy == "allsoft" and mask_rate > 10: continue
                    
                for p in [1,2,4,8]:
                    if strategy != "allsoft" and p != 1: continue
                    
                    for n in range(50):
                        if this_comp == 0: n = 49 - n
                        if strategy == "all":
                            trial_name = "{}_{}_{}_{}".format(str(nn_shape), strategy, mask_rate, str(n))
                        elif strategy == "once":
                            trial_name = "{}_{}_{}_{}".format(str(nn_shape), strategy, mask_start, str(n))
                        elif strategy == "allsoft":
                            trial_name = "{}_{}_{}_{}".format(str(nn_shape), strategy + str(p), mask_rate, str(n))
                            
                        if not first_check:
                            finished, _ = shared_simulate.get_finished(filename, this_comp, 3)
                        if trial_name in finished: continue
                        first_check = False
                        print(trial_name)
                        trials[trial_name] = this_comp
                        shared_simulate.write_trials(trials, filename, this_comp)
                        
                        model = ann.ANNSparse(list(nn_shape), device="cuda", activation="CompTanh", sparsity=3,
                                      strategy = strategy, mask_start = mask_start,
                                      mask_rate = mask_rate, soft_power = p)
                        #model.mask_stop.similarity_stop = True
                        
                        #epochs = 400
                        epochs_each = 10
                        goal_loss = 0.3 if dataset == "voice" else 0.1
                        loss = goal_loss * 10
                        #for e in range(epochs // epochs_each):
                        
                        if strategy == "allsoft":
                            model.es.patience = 10 * mask_rate
                            while not model.es.stop:
                                loss = model.fit(datagen, epochs=epochs_each, learning_rate = 0.001 ** (goal_loss/loss),
                                          early_stop_train = True, generator_kwargs=train_kwargs, verbose=200)
                            model.es = ann.myEarlyStop(patience = 100, delta = 0.0001, mode = "loss")
                            for l in model.linear_layers:
                                model.model[l].mask_from_weight()
                            print("Soft Done")
                            max_in = 0
                            for l in model.linear_layers[::-1]:
                                m = model.model[l]
                                max_in = max(max_in, m.mask.sum(axis = 1).max())
                            print(max_in)
                        
                        first = True
                        while not model.mask_stop.stop or (strategy == "once" and model.total_epochs < mask_start + 1):
                            if not first:
                                model.mask_stop.stop = True
                            loss = model.fit(datagen, epochs=epochs_each, learning_rate = 0.001 ** (goal_loss/loss),
                                      generator_kwargs=train_kwargs, verbose=200)
                            first = strategy != "allsoft"
                            
                        if strategy.startswith("all"):
                            #model.back_prune()
                            print(model.total_epochs, model.nodes)
                        
                        
                        while not model.es.stop:
                            loss = model.fit(datagen, epochs=epochs_each, learning_rate = 0.001 ** (goal_loss/loss),
                                      early_stop_train = True, generator_kwargs=train_kwargs, verbose=200)
                        
                        print(model.total_epochs, "Testing...", end = " ")
                        ideal_acc = model.predict_accuracy(datagen,
                                                           generator_kwargs = train_kwargs)
                        print(ideal_acc)
                        trial_accs = np.zeros((2, len(noisy_parameters)))                        
                        noisy_parameters = np.concatenate([noisy_features,
                                                           noisy_weights_mult[:,:model.synapses],
                                                           noisy_weights_add[:,:model.synapses]],
                                                          axis = 1)
                        for j,n in enumerate(noisy_parameters):
                            noisy_feat = n[:num_features]
                            noisy_weigh = n[num_features:]
                            model.add_noise(noisy_weigh)
                            train_acc = model.predict_accuracy(datagen,
                                                               feature_noise = noisy_feat,
                                                               generator_kwargs = train_kwargs)
                            trial_accs[0, j] = train_acc
                            if dataset == "voice":
                                val_acc = model.predict_accuracy(efg.batch_audio_generator,
                                                             feature_noise = noisy_feat,
                                                             generator_kwargs = {"mode" : "train",
                                                                                 "data_set" : "val"})
                                trial_accs[1, j] = val_acc
                            
        
                            if (j+1) % 100 == 0: print(j+1, end = " ")
                        
                        print()
                        
                        trials[trial_name] = {"shape" :str(nn_shape),
                                              "idealacc": ideal_acc,
                                              "epochs": model.total_epochs,
                                              "accs" : {"train" : list(trial_accs[0, :]),
                                                        "val" : list(trial_accs[1, :])}}
                        
                        shared_simulate.write_trials(trials, filename, this_comp)