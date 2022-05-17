#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 10:41:58 2020

@author: djanke3
"""

import numpy as np

mask = np.zeros((18,3)).astype(int)
for i in range(18):
	mask[i] = np.sort(np.random.permutation(6)[:3])
	
print(mask)

mask_onehot = np.zeros((18,6)).astype(int)
for i,j in enumerate(mask):
	mask_onehot[i,j.astype(int)] = 1

print(mask_onehot)