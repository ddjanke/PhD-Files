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
save_file = "SparsityTest_" + dataset + activation + "_12full_gradual.json"
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

sparsity = 3
this_comp = computer_identity["number"]
done = 0
#trials = {k:v for k,v in trials.items() if "graft" not in k}

if __name__ == "__main__":
    
    accs = []
    shapes = ([12, 20, 20, 20, 20, 1], [12, 20, 20, 20, 1])
    sparsity_type = {"prune":[False,False],
                     "graft":[False,True],
                     "random":[True,False]}
    
    first_check = True
    for nn_shape in shapes:

        n_bits = 1    
        for n in range(50):
            if this_comp == 0: n = 49 - n
            
            model = ann.ANNModelTorch(list(nn_shape), device="cuda", activation="CompTanh", sparsity=1)
            
            #epochs = 400
            epochs_each = 10
            goal_loss = 0.3 if dataset == "voice" else 0.1
            loss = goal_loss * 10
            #for e in range(epochs // epochs_each):
            
            max_in = 0
            loss = model.fit(datagen, epochs=3900, learning_rate = 0.001,
                             early_stop_train=False,
                             generator_kwargs=train_kwargs, verbose=100,
                             power_of_2 = 0.001, quantize_rate = 800, quantize_bits=n_bits)
            
                
            print(model.total_epochs, "Testing...", end = " ")
            ideal_acc = model.predict_accuracy(datagen,
                                               generator_kwargs = train_kwargs)
            print(ideal_acc, end = " ")
                        
            for l in model.linear_layers:
                model.model[l].quantize_weights(n_bits)
            
            quant_acc = model.predict_accuracy(datagen,
                                               generator_kwargs = train_kwargs)
            print(quant_acc)
            
