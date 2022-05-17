#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 16:13:27 2021

@author: djanke3
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style(rc={'font.family':"Times New Roman", 'font.size': 12, 'dpi': 300})

data = pd.read_csv("find_train_size.csv", header=None)
print(data.head())

percentage = data.iloc[0]
data = data.iloc[1:] * 100

plt.figure(figsize=(4,2))

plt0 = plt.plot(percentage.to_numpy(), data.to_numpy().T)

for i,l in enumerate(plt0):
    l.set_color(['blue','cornflowerblue','red', 'lightcoral'][i])
plt.xscale("log")
plt.legend(["train", "train_all", "val", "val_all"], loc = "lower right")
plt.ylabel("Accuracy (%)", fontsize=12)
plt.xlabel("Fraction of Data Used for Training and Testing", fontsize=12)


figname = "find_train_size.png"
plt.savefig(figname, dpi=300, bbox_inches='tight')