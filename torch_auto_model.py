#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 14:50:13 2020

@author: djanke3
"""

import torch
from torch.nn import Parameter
from torch.nn import functional as F
from torch.nn import init
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import queue
import ergence as erg

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"


nthreads = torch.multiprocessing.cpu_count()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def sort_mask(mask):
	sorted_mask = np.sort(mask,axis = -1)
	ilist = list(range(sorted_mask.shape[0]))
	sorted_mask,ilist = sort_submask(sorted_mask,ilist)
	return sorted_mask, ilist[:,None]

def sort_submask(mask,ilist):
	if len(mask.shape) == 1:
		rows = np.argsort(mask)
		return np.take(mask,rows), np.take(ilist,rows)
	rows = np.argsort(mask[:,0])
	sorted_mask = mask[rows,:]
	ilist = np.take(ilist,rows)
	mi = np.unique(sorted_mask[:,0])
	for val in mi:
		rows = np.argwhere(sorted_mask[:,0] == val).squeeze()
		sub_mask = sorted_mask[rows,1:].squeeze()
		sub_ilist = np.take(ilist, rows)
		if sub_ilist.shape == ():
			continue
		#if sub_mask.shape == ():
		#	continue
		else:
			mask_shape = sorted_mask[rows,1:].shape
			sorted_submask, sorted_ilist = sort_submask(sub_mask,sub_ilist)
			sorted_mask[rows,1:]= sorted_submask.reshape(mask_shape)
			ilist[rows.min():rows.max()+1] = sorted_ilist
	return sorted_mask,ilist
	
'''
class ANNModelParalel(torch.nn.Module):
	def __init__(self, networkshape, activation = torch.nn.Tanh(), device = 'cpu', parallelism = 1):
		super(ANN_Model,self).__init__()
		self.networkshape = networkshape
		self.layers = len(networkshape)
		self.activation = activation
		self.device = device
		self.parallelism = prallelism
		self.initialize_module()
		self.rounds = 0
		self.rounds_limit = 3000
		
	def initialize_module(self):
		layers = []
		for i,ns in enumerate(self.networkshape):
			layers += [Linear_maxN(ns[1],ns[0])]
			#if i < self.layers - 1:
			layers += [self.activation]
			
		chunklen = (m + nthreads - 1) // nthreads
		chunks = [[arr[i * chunklen:(i + 1) * chunklen] for arr in in_arrays]+[weightrix,np.array(nodes)] for i in range(nthreads)]
		
		que = queue.Queue()
		#threads = [threading.Thread(target = npjitGradient,args = chunk) for chunk in chunks]
		threads = [threading.Thread(target = lambda q, args: q.put(npjitGradient(*args)),args = (que,chunk)) for chunk in chunks]
		for thread in threads:
			thread.start()
		for thread in threads:
			thread.join()
		while not que.empty():
			result = que.get()
			J += result[0]/nthreads
			Delta += result[1]/nthreads
		
		self.sequential_model = torch.nn.Sequential(*layers).to(device)
		
	def fit(self, X,y):
		pass

	def forward(self,X):
		splits = iter(x.split(self.split_size, dim=0))
        s_next = next(splits)
        s_prev = self.seq1(s_next).to('cuda:1')
        ret = []

        for s_next in splits:
            # A. s_prev runs on cuda:1
            s_prev = self.seq2(s_prev)
            ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))

            # B. s_next runs on cuda:0, which can run concurrently with A
            s_prev = self.seq1(s_next).to('cuda:1')

        s_prev = self.seq2(s_prev)
        ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))

        return torch.cat(ret)
		
		if self.rounds == self.rounds_limit:
			self.rounds = -1
			print('x')
			#for module in self.sequential_model:
		if self.training:
			self.rounds += 1
		row_swap = np.arange(X.shape[1])[None,:]
		X = self.sequential_model(X)
		#for module in self.model:
		#	if 'weight' in module.state_dict():
		#		X,row_swap = module(X,row_swap)
		#	else:
		#		X = module(X)
		return X
'''

class ANN_Model(torch.nn.Module):
	def __init__(self, networkshape, activation = torch.nn.Tanh(), device = 'cpu', parallelism = 1):
		super(ANN_Model,self).__init__()
		self.networkshape = networkshape
		self.layers = len(networkshape)
		self.activation = activation
		self.device = device
		self.initialize_module()
		self.rounds = 0
		self.rounds_limit = 3000

	def initialize_module(self):
		layers = []
		for i,ns in enumerate(self.networkshape):
			layers += [Linear_maxN(ns[1],ns[0])]
			if i < self.layers - 1:
				layers += [self.activation]
		
		self.model = torch.nn.Sequential(*layers).to(self.device)
		#self.parallel_model = torch.nn.DataParallel(self.sequential_model, device_ids=[0, 0])

	def forward(self,X):
		if self.rounds == self.rounds_limit:
			self.rounds = -1
			#for module in self.sequential_model:
		if self.training:
			self.rounds += 1
		row_swap = np.arange(X.shape[1])[None,:]
		#X = self.sequential_model(X)
		for module in self.model:
			if 'weight' in module.state_dict():
				X,row_swap = module(X,row_swap)
			else:
				X = module(X)
		return X
	
'''
class Linear_masked(torch.nn.Linear):
	def __init__(self,in_features, out_features, mask = None, bias=True, device = 'cuda'):
		super(Linear_masked,self).__init__(in_features, out_features, bias)
		if mask is None:
			self.mask = torch.ones((out_features, in_features), device = device)
		else:
			assert(mask.shape == (out_features,in_features))
			self.mask = mask
			
		if device == 'cuda':
			self.mask = self.mask.cuda()
			
	def forward(self,X):
		sd = self.state_dict()
		w = sd['weight'].cuda()
		sd['weight'] = w*self.mask
		self.load_state_dict(sd)
		return F.linear(X, self.weight, self.bias)
'''
	
class Linear_maxN(torch.nn.Linear):
	def __init__(self,in_features, out_features, maxN = 3, bias=True, device = 'cuda'):
		super(Linear_maxN,self).__init__(in_features, out_features, bias)
		self.maxN = maxN
		self.mask = np.ones((out_features,maxN), dtype = int)
		self.rounds = 0
			
	def forward(self,X,row_swap = None):
		if row_swap is None:
			row_swap = np.arange(X.shape[1])[None,:]
		if self.rounds == 3000:
			row_swap = self.mask_weight(row_swap)
			self.rounds = -1
		if self.training:
			self.rounds += 1
		
		return F.linear(X, self.weight, self.bias), row_swap
	
	def mask_weight(self,row_swap):
		with torch.no_grad():
			sd = self.state_dict()
			w = sd['weight'].cpu().numpy()
			print(w.shape,row_swap.shape)
			w = np.take_along_axis(w, row_swap,axis = 1)
			#print(w)
			mask = np.argpartition(np.abs(w),-self.maxN)[:,-self.maxN:]
			mask, row_swap = sort_mask(mask)
			w = np.take_along_axis(w, row_swap, axis = 0)
			
			mask_onehot = np.zeros(w.shape).astype(int)
			for i,j in enumerate(mask):
				mask_onehot[i,j.astype(int)] = 1
			
			self.mask = mask
			new_weight = torch.from_numpy(w*mask_onehot).cuda()
			#print(new_weight)
			sd['weight'] = new_weight
			self.load_state_dict(sd)
			#input()
			return row_swap.T
		
class VoiceDataset(Dataset):
	
	def __init__(self, audio_features, labels):
		self.X = audio_features
		self.y = labels
		self.length = X.shape[0]
		
	def __len__(self):
		return self.length
	
	def __getitem__(self, idx):
		return self.X[idx,:],self.y[idx]


def initialize_Xrandom(networkshape,m):
	X = torch.zeros([networkshape[0][1],m], requires_grad = False)
	torch.nn.init.xavier_normal_(X)
	return X


networkshape = [(26,12),(18,26),(6,18),(3,6),(1,3)]
input_combos = [(0,1,2,3,4,5)]
m = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

X = initialize_Xrandom(networkshape,m).to(device)

model = ANN_Model(networkshape,device = 'cuda')


#for param in model.parameters():
#	print(param.data,param.size())
'''	
for name, module in model.named_modules():
	print('__________________________________________________________')
	print('NAME: ',name)
	print('MODULE: ',module)
	print(module.state_dict())
	input()

input()
'''
Y = model.forward(X.T)
print(Y)

Audio_directory = "/home/djanke3/Documents/Audio/"
erg.importAudio(Audio_directory)


Xtrain, ytrain, Xval1, yval1, Xval2, yval2 \
= erg.createFeatures(var = None,save = False,train_only = True, noise_adjust = 0.5)

print(Xtrain.shape)

X = torch.as_tensor(Xtrain, dtype = torch.float32)
Y = torch.as_tensor(ytrain, dtype = torch.float32)
X.requires_grad_(True)
model.cuda()
X = X.to(torch.device('cuda'))
Y = Y.to(torch.device('cuda')).squeeze().detach()

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters())

model.eval()
y_pred = model(X).squeeze()
print(y_pred.shape,Y.shape)
before_train = criterion(y_pred, Y)
print(before_train.item())

model.train()
epoch = 60000

for epoch in range(epoch):
	
	optimizer.zero_grad()
	
	y_pred = model(X).squeeze()
	
	loss = criterion(y_pred, Y)
	
	if epoch % 100 == 0:
		print('Epoch {} train loss: {}'.format(epoch, loss.item()))
	
	loss.backward()
	optimizer.step()

model.eval()
y_pred = (model(X).squeeze() > 0.5).float()
print(y_pred[:100], Y[:100])
print(torch.eq(y_pred,Y)[:100])
acc = torch.sum(torch.eq(y_pred,Y).float())/len(Y)

print(acc.data)


for name, module in model.named_modules():
	print('__________________________________________________________')
	print('NAME: ',name)
	print('MODULE: ',module)
	if 'weight' in module.state_dict(): print(module.mask)

