#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 13:38:58 2021

@author: djanke3
"""

import numpy as np
import matplotlib.pyplot as plt

a = np.linspace(-5,5,100)

bsine = np.sin(a)
bsineclip = bsine * 2
bsineclip = np.array([b if abs(b) < 1.8 else (-1 + 2*(b>0))*1.8 for b in bsineclip])

ctanh = 1.8 * np.tanh(10/9*a)

plt.plot(a,bsine, a,bsineclip)
plt.legend(labels=["sine(x)", "2sine(x)"])
plt.ylim([-2,2])
plt.xlabel('$v_{IN}$ (V)')
plt.ylabel('$v_{OUT}$ (V)')
plt.show()

plt.plot(a,2*a,a,ctanh)
plt.legend(labels=["infinate voltage range", "finate voltage range"])
plt.ylim([-3,3])
plt.xlabel('$v_{IN}$ (V)')
plt.ylabel('$v_{OUT}$ (V)')
plt.show()