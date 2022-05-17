# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 14:32:49 2019

@author: ddjan
"""
import os
import numpy as np
import pandas as pd
import ergence_feature_extraction as erg
import ANN_Model as ann
import ann_sparse
import matplotlib.pyplot as plt
import torch
import json
from sklearn import datasets
import shared_simulate

# Initial settings
train_split = (0.6, 0.2, 0.2)
total_files = 400
GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory
features = ["diff", "n"]
bands = 6

rng = np.random.default_rng()


activation = "Tanh"
data = "grid"
save_file = "NetworkShapeAnalysis_{}_{}.json".format(data, activation)

with open("/home/djanke3/Documents/computer_identity.json", 'r') as ci:
    computer_identity = json.load(ci)
def get_dataset(data):
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
        
    elif data == "drive":
        dataset = datasets.load_iris()
        df = pd.read_csv("Sensorless_drive_diagnosis.txt", sep = '\s+', header = None)
        dataset.target = (df[48] - 1).values
        X = df.drop([48], axis = 1)
        corr = X.corr()
        keep = []
        delete = []
        for i in range(X.shape[1]):
            if i in delete: continue
            keep.append(i)
            for j in range(X.shape[1]):
                if j in keep: continue
                if corr.loc[j,i] > 0.8:
                    delete.append(j)
        X = X[keep]
        dataset.data = X.values
        
    elif data == "ionosphere":
        dataset = datasets.load_iris()
        df = pd.read_csv("ionosphere.data", header = None)
        #print(df.head())
        y = df[34]
        y = (y == "g").astype(int)
        #y[y == "b"] = 0
        dataset.target = y.values
        print(y.sum())
    
        X = df.drop([34], axis = 1)
        corr = X.corr()
        keep = []
        delete = []
        for i in range(X.shape[1]):
            if i in delete: continue
            keep.append(i)
            for j in range(X.shape[1]):
                if j in keep: continue
                if corr.loc[j,i] > 0.8:
                    delete.append(j)
        X = X[keep]
        dataset.data = X.values
        
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
        
    elif data == "mnist":
        dataset = datasets.load_iris()
        X = pd.read_csv("MNIST_X.csv", header = None)
        y = pd.read_csv("MNIST_y.csv", header = None).astype(int)[0] - 1
        dataset.target = y.values
        
        corr = X.corr()
        keep = []
        delete = []
        for i in range(X.shape[1]):
            if i in delete: continue
            keep.append(i)
            for j in range(X.shape[1]):
                if j in keep: continue
                if corr.loc[j,i] > 0.9:
                    delete.append(j)
        
        X = X[keep]
        dataset.data = X.values
        
    elif data == "mnist_train":
        dataset = datasets.load_iris()
        X = pd.read_csv("mnist_train.csv", header = None)
        y = X.astype(int).iloc[:,0]
        X = X.iloc[:,1:]/255
        dataset.target = y.values
        '''
        corr = X.corr()
        keep = []
        delete = []
        for i in range(X.shape[1]):
            if i in delete: continue
            keep.append(i)
            for j in range(X.shape[1]):
                if j in keep: continue
                if corr.iloc[j,i] > 0.9:
                    delete.append(j)
        
        X = X.iloc[:,keep]
        '''
        dataset.data = X.values
        
    elif data == "mnist_test":
        dataset = datasets.load_iris()
        X = pd.read_csv("mnist_test.csv", header = None)
        y = X.astype(int).iloc[:,0]
        X = X.iloc[:,1:]/255
        dataset.target = y.values
        '''
        corr = X.corr()
        keep = []
        delete = []
        for i in range(X.shape[1]):
            if i in delete: continue
            keep.append(i)
            for j in range(X.shape[1]):
                if j in keep: continue
                if corr.iloc[j,i] > 0.9:
                    delete.append(j)
        
        X = X.iloc[:,keep]
        '''
        dataset.data = X.values
    
    else:
        dataset = datasets.load_iris()
        
    return dataset

dataset = get_dataset(data)
d_test = get_dataset("mnist_test")

print("Generating Audio:")
def data_generator(batches = 1):
    X,y = dataset.data, dataset.target
    X -= X.mean(axis = 0)
    X /= X.std()
    if batches >= 1:
        len_batch = len(y) // batches
        for i in range(batches):
            xi = rng.choice(len(y), size = len_batch, replace=False)
            yield i,X[xi,:],y[xi]
    else:
        len_batch = int(len(y) * batches)
        xi = rng.choice(len(y), size = len_batch, replace=False)
        yield 0,X[xi,:],y[xi]
    
def data_generator_test(batches = 1):
    X,y = d_test.data, d_test.target
    X -= X.mean(axis = 0)
    X /= X.std()
    len_batch = len(y) // batches
    for i in range(batches):
        xi = rng.choice(len(y), size = len_batch, replace=False)
        yield i,X[xi,:],y[xi]


num_features= 12
num_classes = 1
def set_parameters():
    global num_features
    global num_classes
    for i,X,y in data_generator():
        xpd = pd.DataFrame(X)
        num_features = X.shape[1]
        num_classes = y.max()
        if num_classes > 1: num_classes += 1
        print(X.shape, y.shape)
        
set_parameters()

if __name__ == "__main__":
    
    accs = []
    nn_shape = [num_features, 12, 9, 6, 4, num_classes]
    models = []
    for n in range(40):
        model = ann_sparse.ANNSparse(nn_shape, activation="CompTanh", sparsity=3,
                                  strategy = "allsoft", mask_start = 0, mask_rate = 20)

        #model.strategy = "all"
        epochs = 800
        epochs_each = 20
        goal_loss = 0.1
        loss = goal_loss * 10
                    
        if model.strategy == "allsoft":
            while not model.es.stop:
                loss = model.fit(data_generator, epochs=epochs_each, learning_rate = 0.001 ** (goal_loss/loss),
                          early_stop_train = True, generator_kwargs={}, verbose=20)
            model.es = ann.myEarlyStop(patience = 100, delta = 0.0001, mode = "loss")
            #model.strategy = "all"
            for l in model.linear_layers:
                model.model[l].mask_from_weight()
            print("Soft Done")
            
        
        first = True
        while not model.mask_stop.stop: #for e in range(epochs // epochs_each):
            if not first:
                model.mask_stop.stop = True
            loss = model.fit(data_generator, epochs=epochs_each, learning_rate = 0.002 ** (goal_loss/loss),
                      early_stop_train=False, generator_kwargs={}, verbose=20)
            first = False
            
            '''
            for l in model.linear_layers[::-1]:
                m = model.model[l]
                print(m.weight)
                #'''

        initial_acc = model.predict_accuracy(data_generator, generator_kwargs = {})
        #model.set_activation("CompTanh")
        model.fit(data_generator, epochs = 100000, learning_rate = 0.002,
                          early_stop_train = True, verbose = 100, generator_kwargs = {})
        train_acc = model.predict_accuracy(data_generator, generator_kwargs = {})
        accs.append(train_acc)
        #print(model.model.state_dict())
        #for k,v in model.model.state_dict().items():
        #    print(k)
        #    print(v)
        for l in model.linear_layers[::-1]:
            m = model.model[l]
            print(m.mask.sum(axis = 1))
        #input()
            
        models.append((loss, train_acc, model.total_epochs))
    
    for m in sorted(models, key=lambda k: k[0]):
        print(m)