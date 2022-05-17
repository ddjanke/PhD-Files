import os
import numpy as np
import ergence_feature_extraction as erg
import ANN_Model as ann
import matplotlib.pyplot as plt
import torch
import json
import shared_simulate
import NN_optimize

with open("/home/djanke3/Documents/computer_identity.json", 'r') as ci:
    computer_identity = json.load(ci)
    
this_comp = computer_identity["number"]

n_bits = 3 - this_comp
dataset = 1

dataset = ["voice", "grid", "mnist", "mnist_train", "mnist_test"][dataset]
NN_optimize.dataset = NN_optimize.get_dataset(dataset)
NN_optimize.set_parameters()

quant_noise_profile = [{},
                       {"profile":"uniform", "args":(-1/3, 1/3)},
                       {"profile":"normal", "args":(0,0.08)},
                       {"profile":"normal", "args":(0,0.03)}][n_bits]

activation = ["ReLU1", "ReLU2"] #, "ReLU4"]
weight_noise_kwargs = [{}] #, quant_noise_profile]
power_of_2 = [0.01, 0.001, 0.0001, 0][::-1]
quantize_rate = [0] #, 400, 800, 1200]






datagen = NN_optimize.data_generator
datagen_test = NN_optimize.data_generator_test
train_kwargs = {"batches" : 1}


filename = "QuantizedTrain2_{}_{}.json".format(n_bits,dataset)


if __name__ == "__main__":
    
    accs = []
    shapes = ([NN_optimize.num_features, 20, 20, NN_optimize.num_classes],
              [NN_optimize.num_features, 20, 20, 20, NN_optimize.num_classes])
    sparsity_type = {"prune":[False,False],
                     "graft":[False,True],
                     "random":[True,False]}
    trials = {}
    first_check = True
    
    finished, trials = shared_simulate.get_finished(filename, this_comp, 3)
    #print(len(finished))
    #print(finished)
    
    for nn_shape in shapes:
        for a in activation[::1]:
            for wnk in weight_noise_kwargs:
                for po2 in power_of_2[::1]:
                    for qr in quantize_rate:
                        for n in range(10):
                            if this_comp == 0: n = 9 - n
                            trial_name = "{}_{}_{}_{}_{}_{}".format(nn_shape, a, wnk.get("profile","none"),
                                                                    po2, qr, n)
                            
                            
                            
                            if not first_check:
                                finished, _ = shared_simulate.get_finished(filename, this_comp, 3)
                            if trial_name in finished: continue
                            first_check = False
                            print(trial_name)
                            trials[trial_name] = this_comp
                            shared_simulate.write_trials(trials, filename, this_comp)
                            
                            model = ann.ANNModelTorch(list(nn_shape), device="cuda", activation=a, sparsity=1)
                            
                            #epochs = 400
                            epochs_each = 10
                            goal_loss = 0.3 if dataset == "voice" else 0.05
                            loss = goal_loss * 10
                            #for e in range(epochs // epochs_each):
                            
                            max_in = 0
                            if qr == 0:
                                model.es.patience = 1000
                                while not model.es.stop:
                                    loss = model.fit(datagen, epochs=20, learning_rate = 0.001 ** (goal_loss/loss),
                                                 early_stop_train=True,
                                                 generator_kwargs=train_kwargs, verbose=200,
                                                 weight_noise_kwargs = wnk,
                                                 feature_noise_kwargs = {},
                                                 power_of_2 = po2, quantize_rate = 0, quantize_bits=n_bits)
                            else:
                                model.es.patience = max(400, qr * 4)
                                epochs_each = 20
                                while not model.quant_stop.stop:
                                    loss = model.fit(datagen, epochs=10000, learning_rate = 0.001 ** (goal_loss/loss),
                                                     early_stop_train=False,
                                                     generator_kwargs=train_kwargs, verbose=200,
                                                     weight_noise_kwargs = wnk,
                                                     feature_noise_kwargs = {},
                                                     power_of_2 = po2, quantize_rate = qr, quantize_bits=n_bits)
                            
                                
                            print(model.total_epochs, "Testing...", end = " ")
                            ideal_acc = model.predict_accuracy(datagen,
                                                               generator_kwargs = train_kwargs)
                            print(ideal_acc, end = " ")
                                        
                            for l in model.linear_layers:
                                model.model[l].quantize_weights(n_bits)
                            
                            quant_acc = model.predict_accuracy(datagen, generator_kwargs = {})
                            print(quant_acc, end = " ")
                            
                            if dataset == "mnist_train":
                                quant_acc_test = model.predict_accuracy(datagen_test, generator_kwargs = {})
                                print(quant_acc_test)
                            else:
                                quant_acc_test = 0
                            
                            trials[trial_name] = {"ideal": ideal_acc,
                                                 "quant_train": quant_acc,
                                                 "quant_test": quant_acc_test}
                            print('\n')
            
                            shared_simulate.write_trials(trials, filename, this_comp)