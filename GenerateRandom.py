# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 16:48:42 2019

@author: ddjan
"""
import numpy as np
import ANN_Model as ann

numfil = 5
nodes = (numfil, 2*numfil, 1)
synapses = ann.countSynapses(nodes)

all_rand = np.vstack([np.random.normal(1,x/3,(1000,5*numfil+2*synapses)) for x in [0.01,0.05,0.1,0.15,0.2]])
print(all_rand.shape)
np.savetxt('AllVarsLargeTest.csv',all_rand,delimiter=',')