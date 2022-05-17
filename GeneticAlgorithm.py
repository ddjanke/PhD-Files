#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 16:13:45 2020

@author: djanke3
"""



import ANN_Model as ann
import numpy as np
import random
from NN_optimize import data_generator

hidden_layers = list(range(2,6))
min_nodes = 3
max_nodes = 24
num_features = 12



def sparse_valid(nodes, sparsity):
    if nodes[-1] != sparsity or nodes[0] < num_features / (sparsity - 1): return False
    return all([n <= (sparsity - 1) * nodes[i + 1] for i,n in enumerate(nodes[:-1])])

def new_model(nodes, activation, sparsity):
    model = ann.ANNModelTorch(nodes,
                              activation=activation,
                              sparsity=sparsity,
                              sparse_randomize=True)
    model.set_sparse_params("each", 400)
    return model

class Population:
    def __init__(self, size, nodes, sparsity):
        self.size = size
        self.nodes = nodes
        self.sparsity = sparsity
        
    def populate(self):
        self.citizens = []
        for _ in range(self.size):
            model = new_model(self.nodes, "Tanh", self.sparsity)
            
            self.citizens.append([0, model])
            
    def select(self, total, top):
        self.citizens = sorted(self.citizens, key=lambda k: k[0])
        bottom = random.sample(self.citizens[:-top], total - top)
        self.breeders = self.citizens[-top:] + bottom
        for b in self.breeders:
            b[1].load_initial_state()
            b[1].total_epochs = 0
    
    @staticmethod
    def crossover(first, second):
        max_swap = min(first.shape[0], second.shape[0])
        scale = max_swap / 2  
        n = int(abs(np.random.normal(loc=0, scale=scale, size = 1)))
        n = min(n, max_swap)
        first_rows = random.sample(list(range(first.shape[0])), n)
        second_rows = random.sample(list(range(first.shape[0])), n)
        first[first_rows,:], second[second_rows,:] = second[second_rows,:], first[first_rows,:]
        return first, second
    
    @staticmethod    
    def mutate(mask):
        scale = mask.shape[0] / 3
        n = int(abs(np.random.normal(loc=0, scale=scale, size = 1)))
        n = min(mask.shape[0],n)
        rows = random.sample(list(range(mask.shape[0])), n)
        for r in rows:
            zero = random.choice(np.where(mask[r:] == 0))
            one = random.choice(np.where(mask[r:] == 1))
            mask[r, zero], mask[r, one] = (1,0)
            
        scale = 1/3
        n = int(abs(np.random.normal(loc=0, scale=scale, size = 1)))
        if n and mask.shape[0] > 1:
            rows = random.sample(list(range(mask.shape[0])), 2)
            mask[rows[0],:], mask[rows[1],:] = mask[rows[1],:], mask[rows[0],:]
            
        return mask
            
    def breed(self):
        new_count = self.size - len(self.breeders)
        self.citizens = []
        for b in self.breeders:
            #model = new_model(self.nodes, "Tanh", self.sparsity)
            #model.set_mask(b.get_mask())
            self.citizens.append(b)
            
        for _ in range(new_count // 2):
            b1,b2 = random.sample(self.breeders, 2)
            mask1 = b1[1].get_mask()
            mask2 = b2[1].get_mask()
            
            for i,(m1,m2) in enumerate(zip(mask1,mask2)):
                mask1[i],mask2[i] = self.crossover(m1,m2)
                mask1[i] = self.mutate(mask1[i])
                mask2[i] = self.mutate(mask2[i])
            
            m1 = new_model(self.nodes, "Tanh", self.sparsity)
            #m1.model.load_state_dict(b1.initial_state)
            m1.set_mask(mask1)
            m2 = new_model(self.nodes, "Tanh", self.sparsity)
            #m2.model.load_state_dict(b2.initial_state)
            m2.set_mask(mask2)
            self.citizens += [[0,m1],[0,m2]]
            
 
nodes = [12, 12, 9, 6, 3, 1]
epochs = 200
epochs_each = 10
goal_loss = 0.1
loss = goal_loss * 10

pop = Population(20, nodes, 4)
pop.populate()
for g in range(10):
    print("Generation",g)
    for i,c in enumerate(pop.citizens):
        #print(c[1].get_mask())
        if c[0] > 0: continue
        model = c[1]
        for e in range(epochs // epochs_each):
            loss = model.fit(data_generator, epochs=epochs_each, learning_rate = 0.01 ** (goal_loss/loss) * 4,
                      early_stop_train=False, generator_kwargs={}, verbose=50)
            
        initial_acc = model.predict_accuracy(data_generator, generator_kwargs = {})
        pop.citizens[i][0] = initial_acc
    pop.select(6, 4)
    pop.breed()
  
    
for c in sorted(pop.citizens, key=lambda k: k[0]):
    print(c[0])