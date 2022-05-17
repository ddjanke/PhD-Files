# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 10:31:55 2019

@author: ddjan
"""

import numpy as np
import ANN_Model
from copy import deepcopy
#from dask.distributed import Client

def debugInit(num_out,num_in):
	tot = (1+num_in)*num_out
	W = np.reshape(np.sin(np.arange(1,tot+1)),(num_out,1+num_in))
	return W/10

'''
#This section is for gradient checking
in_size = 6;
hidden = 5
hidden2 = 5
hidden3 = 5
num = 2
m = 10

Theta1 = debugInit(hidden,in_size)
Theta2 = debugInit(hidden2,hidden)
Theta3 = debugInit(num,hidden2)
#Theta4 = debugInit(num,hidden3)
X = debugInit(m,in_size-1)
y = 1 + np.mod(np.arange(1,m+1),num)
weights = [Theta1,Theta2,Theta3]#,Theta4]
#nn_params = np.hstack((Theta1.flatten('F'),Theta2.flatten('F')))
test = ANN_Model.ANN_Model((in_size,hidden,hidden2,num),2)
test.setWeights(weights)
'''


#This section is for MNIST
test = ANN_Model.ANN_Model((400,25,10),1)

with open('MNIST_X.csv','r') as f:
	X = np.loadtxt(f,delimiter=',')
with open('MNIST_y.csv','r') as f:
	y = np.loadtxt(f,delimiter=',')
	


print('starting')
weights = test.weights
g = test.nnCostFunction(weights,X,y)
print('cost done')


#This section for MNIST
x = test.fit(X,y,1)

ynew = test.predict(X)
agree = sum(np.ravel(ynew)==y)
print(agree/len(y))

#raise SystemExit(0)

'''
# This section for gradient checking

e = 1e-4
wtemp = deepcopy(weights)
numgrad = []

for i,w in enumerate(weights):
	perturb = np.zeros(w.shape)
	numgrad += [np.zeros(w.shape)]
	for j,x in np.ndenumerate(w):
		perturb[j] = e
		wtemp[i] = w+perturb
		#test.setWeights(wtemp)
		loss1 = test.nnCostFunction(wtemp,X,y,'j')
		wtemp[i] = w-perturb
		#test.setWeights(wtemp)
		loss2 = test.nnCostFunction(wtemp,X,y,'j')
		wtemp[i] = deepcopy(w)
		numgrad[i][j] = (loss1-loss2)/(2*e)
		perturb[j] = 0


print(ANN_Model.fullCompare(numgrad,g,False))
print(numgrad)
print(g)
'''