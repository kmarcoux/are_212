#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 14:51:58 2021

@author: kendramarcoux
"""


import numpy as np
from matplotlib import pyplot as plt
N=100
sigma_u = 1 #Homoskedasticity


from scipy.stats import distributions as iid


D = 1000 # Monte Carlo draws
K =6
A=6
B=6

for k in range(K):
    for a in range(A):
        for b in range (B):
            globals()["mean%s_%s_%s_draws"%(k,a,b)] = []
            globals()["mean_trig_%s_%s_%s_draws"%(k,a,b)] = []

for d in range(D):
    u = iid.norm.rvs(size=(N,1))*sigma_u
    X = iid.norm.rvs(size=(N,1))
    
    for k in range(K):
        for a in range(A):
            for b in range (B):        
                globals()["mean%s_%s_%s"%(k,a,b)] = float((a*X.T**k+b*X.T**(k-1))@u)
                globals()["mean_trig_%s_%s_%s"%(k,a,b)] = float((a*np.sin(b*X.T**(k-1)))@u)
                globals()["mean%s_%s_%s_draws"%(k,a,b)].append(globals()["mean%s_%s_%s"%(k,a,b)])
                globals()["mean_trig_%s_%s_%s_draws"%(k,a,b)].append(globals()["mean_trig_%s_%s_%s"%(k,a,b)])
    

fig, ax = plt.subplots(1, 3, figsize=(15, 7))

for k in range(K):
    for a in range(A):
        for b in range (B):
            ax[1].hist(globals()["mean%s_%s_%s_draws"%(k,a,b)], alpha = 0.5)
            ax[2].hist(globals()["mean_trig_%s_%s_%s_draws"%(k,a,b)], alpha = 0.5)
            
