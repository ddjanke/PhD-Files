#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 17:30:24 2020

@author: djanke3
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Parameters and DataLoaders
input_size = 5
output_size = 2

batch_size = 50000
data_size = 1000000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RandomDataset(Dataset):
	
	def __init__(self, size, length):
		self.len = length
		self.data = torch.randn(length, size)

	def __getitem__(self, index):
		return self.data[index]

	def __len__(self):
		 return self.len

class Model(nn.Module):
	
	def __init__(self, input_size, output_size):
		super(Model, self).__init__()
		self.fc = nn.Linear(input_size, output_size)

	def forward(self, input):
		output = self.fc(input)
		#print("\tIn Model: input size", input.size(),"output size", output.size())
		#print(input.device)
		
		return output
	
def parallelforward(model, rand_loader):
	for data in rand_loader:
		input = data.to(device)
		output = model(input)
	
	
rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True)

model0 = Model(input_size, output_size)
print(model0.fc.weight)
if torch.cuda.device_count() > 1:
	print("Let's use", torch.cuda.device_count(), "GPUs!")
	# dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
	model = nn.DataParallel(model0, device_ids=[0, 1])

model.to(device)

for data in rand_loader:
	input = data.to(device)
	output = model(input)
	#print("Outside: input size", input.size(), "output_size", output.size())
print(model0.fc.weight)
	
	