#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 00:06:46 2021

@author: kendramarcoux
"""
import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt
from scipy.stats import distributions as iid

def dgp(N,beta,gamma,sigma_u,VXZ,het):
    
    # "Square root" of VXZ via eigendecomposition
    lbda,v = np.linalg.eig(VXZ)
    SXZ = v@np.diag(np.sqrt(lbda))
    
    # Generate normal random variates [X*,Z]
    XZ = iid.norm.rvs(size=(N,VXZ.shape[0]))@SXZ.T
    Z = XZ[:,1:]
    
    #Allow for heteroskedasticity (Ex[u^2|Z]\neq \s^2)
    if het == "het":
        sigma_u = (abs(Z)**0.5)[:,[0]]
    else: 
        sigma_u = sigma_u
    
    u = iid.norm.rvs(size=(N,1))*sigma_u
    
    # But X is endogenous...
    X = XZ[:,[0]] + gamma*u
    
    # Calculate y
    y = X*beta + u
    
    return y,X,Z


def bW(W, data):
    y,X,Z = data
    return float(np.linalg.pinv(W@Z.T@X)@W@Z.T@y)

def biv(data):
    y,X,Z = data    
    return float(inv(X.T@Z@inv(Z.T@Z)@Z.T@X)@(X.T@Z@inv(Z.T@Z)@Z.T@y))

def gj(b,y,X,Z):
    """Observations of g_j(b).

    This defines the deviations from the predictions of our model; i.e.,
    e_j = Z_ju_j, where EZ_ju_j=0.

    Can replace this function to testimate a different model.
    """
    return Z*(y - X*b)

def gN(b,data):
    """Averages of g_j(b).

    This is generic for data, to be passed to gj.
    """
    e = gj(b,*data)

    # Check to see more obs. than moments.
    assert e.shape[0] > e.shape[1]
    
    return e.mean(axis=0)

def Omegahat(b,data):
    e = gj(b,*data)

    # Recenter! We have Eu=0 under null.
    # Important to use this information.
    e = e - e.mean(axis=0) 
    
    return e.T@e/e.shape[0]

def J(b,W,data):

    m = gN(b,data) # Sample moments @ b
    N = data[0].shape[0]

    return N*m.T@W@m # Scale by sample size

from scipy.optimize import minimize_scalar

def two_step_gmm(data):

    # First step uses identity weighting matrix
    W1 = np.eye(gj(1,*data).shape[1])

    b1 = minimize_scalar(lambda b: J(b,W1,data)).x 

    # Construct 2nd step weighting matrix using
    # first step estimate of beta
    W2 = inv(Omegahat(b1,data))

    return minimize_scalar(lambda b: J(b,W2,data))


beta = 1
gamma = 1 # Governs effect of u on X
sigma_u = 1

N = 1000
D = 1000 # Monte Carlo draws

ell_options = [1, 4]

fig, ax = plt.subplots(2, 2, figsize=(15, 7))
colcount = 0
rowcount = 0
het = ''

for g in ("Homoskedasticity","Heteroskedasticity"):
    ## But play with Omega if you want to introduce heteroskedascity
    
    if g == "Homoskedasticity":
        het == "het"
    else: 
        het == "nohet"
    
    for iden in ("Just Identified", "Over Identified"):

        if g == "Just Identified":
            ell = 1
        else: 
            ell = 4

        A = np.sqrt(1/np.arange(1,(ell+1)**2+1)).reshape((ell+1,ell+1))

        # Now Var([X,Z]|u) is constructed so guaranteed pos. def.
        VXZ = A.T@A
        
        Q = -VXZ[1:,[0]] # -EZX', or generally Edgj/db'
    
        truth = (beta,gamma,sigma_u,VXZ,het)
    
        
        diag = []
        for q in range(ell):
            d = 2**(1-q)
            diag.append(d)

        W = np.diag(diag)

        bW_draws = []
        bGMM_draws = []
        bIV_draws = []
        for d in range(D):
            data = dgp(N,*truth)
            b = bW(W, data) - beta
            bW_draws.append(b)
            b_iv = biv(data) - beta
            bIV_draws.append(b_iv)
            bGMM = two_step_gmm(data).x - beta
            bGMM_draws.append(bGMM)

        ax[rowcount, colcount].hist(bW_draws,bins=int(np.ceil(np.sqrt(1000))), color='blue', alpha=0.3, label="bW")
        ax[rowcount, colcount].hist(bIV_draws,bins=int(np.ceil(np.sqrt(1000))), color='orange', alpha=0.3, label="GMM")
        ax[rowcount, colcount].hist(bGMM_draws,bins=int(np.ceil(np.sqrt(1000))), color='green', alpha=0.3, label="IV")
        ax[rowcount, colcount].set_title(iden+", with "+g)
        ax[rowcount, colcount].axvline(0, color='r')

        rowcount += 1
    
    rowcount = 0 
    colcount += 1

plt.legend()
_ = fig.suptitle('Monte Carlo simualtion of the bias of b using two step GMM, IV, and b_W') 
