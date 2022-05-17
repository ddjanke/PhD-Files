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
	
def countSynapses(nodes):
	tot = 0
	for i,n in enumerate(nodes[:-1]):
		tot += (n+1)*nodes[i+1]
	return tot


class ANN_Model(torch.nn.Module):
	def __init__(self, networkshape, activation = torch.nn.Tanh(), device = 'cpu', parallelism = 1):
		super(ANN_Model,self).__init__()
		self.networkshape = networkshape
		self.layers = len(networkshape)
		self.activation = activation
		self.device = device
		self.initialize_module()
		self.rounds = 0
		self.rounds_limit = 1000

	def initialize_module(self):
		layers = []
		for i,ns in enumerate(self.networkshape):
			layers += [Linear_wVoltages(ns[1],ns[0], device = device)]
			if i < self.layers - 1:
				layers += [self.activation]
		
		self.model = torch.nn.Sequential(*layers).to(self.device)

	def forward(self,X):
		if self.rounds == self.rounds_limit:
			self.rounds = -1
			rows = np.arange(X.shape[1])[None,:]
			for module in self.model:
				if 'weight' in module.state_dict():
					rows = module.mask_weight(rows)
			self.parallel_model = torch.nn.DataParallel(self.model, device_ids=[0, 1])
			
		if self.training:
			self.rounds += 1
		
		return self.parallel_model(X)
	
class Linear_wVoltages(torch.nn.Linear):
	def __init__(self,in_features, out_features, bias=True, device = 'cpu'):
		super(Linear_wVoltages,self).__init__(in_features, out_features, bias)
		self.activation = torch.nn.Tanh()
		self.rate = 2*torch.ones((out_features,in_features + 1))
		self.maxgain = 1*torch.ones((out_features,in_features + 1))
			
	def forward(self,X):
		weight = self.maxgain[:,1:]*torch.tanh(self.rate[:,1:]*self.weight)
		bias = self.maxgain[:,0]*torch.tanh(self.rate[:,0]*self.bias)
		return F.linear(X, weight, bias)

	
class Linear_maxN(torch.nn.Linear):
	def __init__(self,in_features, out_features, maxN = 3, bias=True, device = 'cpu'):
		super(Linear_maxN,self).__init__(in_features, out_features, bias)
		self.maxN = maxN
		self.mask = np.ones((out_features,maxN), dtype = int)
			
	def forward(self,X):
		return F.linear(X, self.weight, self.bias)
	
	def mask_weight(self,row_swap):
		with torch.no_grad():
			
			sd = self.state_dict()
			w = sd['weight'].cpu().numpy()
			w = np.take_along_axis(w, row_swap,axis = 1)
			
			mask = np.argpartition(np.abs(w),-self.maxN)[:,-self.maxN:]
			mask, row_swap = sort_mask(mask)
			w = np.take_along_axis(w, row_swap, axis = 0)
			
			mask_onehot = np.zeros(w.shape).astype(int)
			for i,j in enumerate(mask):
				mask_onehot[i,j.astype(int)] = 1
			
			self.mask = mask
			new_weight = torch.from_numpy(w*mask_onehot).cuda()
			sd['weight'] = new_weight
			self.load_state_dict(sd)
			
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


####### This section is used for testing the code #####################


def initialize_Xrandom(networkshape,m):
	X = torch.zeros([networkshape[0][1],m], requires_grad = False)
	torch.nn.init.xavier_normal_(X)
	return X

if __name__ == '__main__':

	networkshape = [(26,12),(18,26),(6,18),(3,6),(1,3)]
	m = 1
	
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	
	X = initialize_Xrandom(networkshape,m).to(device)
	
	model = ANN_Model(networkshape,device = device)
	
	
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
	model.to('cuda:0')
	X = X.to('cuda:0')
	Y = Y.to('cuda:0').squeeze().detach()
	
	criterion = torch.nn.BCEWithLogitsLoss()
	optimizer = torch.optim.Adam(model.parameters())
	
	model.eval()
	y_pred = model.forward(X).squeeze()
	print(y_pred.shape,Y.shape)
	before_train = criterion(y_pred, Y)
	print(before_train.item())
	
	model.train()
	epoch = 10000
	
	m = X.shape[0]
	mbs = 10
	audio_data = VoiceDataset(X,Y)
	batch_size = (m + mbs-1)//mbs
	
	for epoch in range(epoch):
		
		optimizer.zero_grad()
		
		y_pred = model.forward(X).squeeze()
		
		loss = criterion(y_pred, Y)
			
		loss.backward()
		optimizer.step()
		
		if epoch % 100 == 0:
			print('Epoch {} train loss: {}'.format(epoch, loss.item()))
		
		
	model.eval()
	y_pred = (model(X).squeeze() > 0.5).float()
	
	acc = torch.sum(torch.eq(y_pred,Y).float())/len(Y)
	
	print(acc.data)
	
	'''
	for name, module in model.named_modules():
		print('__________________________________________________________')
		print('NAME: ',name)
		print('MODULE: ',module)
		if 'weight' in module.state_dict(): print(module.mask)
	'''
