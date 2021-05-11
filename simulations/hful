#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  8 10:37:28 2021

@author: kendramarcoux
"""

import numpy as np
from numpy.linalg import inv
from scipy.stats import distributions as iid
from matplotlib import pyplot as plt


beta = 1     # "Coefficient of interest"
gamma = 1    # Governs effect of u on X
sigma_u = 1  # Note assumption of homoskedasticity

# Let Z have order ell, and X order 1, with Var([X,Z]|u)=VXZ
ell = 4 

# Arbitrary (but deterministic) choice for VXZ = [VX Cov(X,Z);
#                                                 Cov(Z,X) VZ]
# Pinned down by choice of a matrix A...
A = np.sqrt(1/np.arange(1,(ell+1)**2+1)).reshape((ell+1,ell+1)) 

# Now Var([X,Z]|u) is constructed so guaranteed pos. def.
VXZ = A.T@A 

Q = -VXZ[1:,[0]]  # -EZX', or generally Edgj/db'

truth = (beta,gamma,sigma_u,VXZ)

def dgp(N,beta,gamma,sigma_u,VXZ):
    """Generate a tuple of (y,X,Z).

    Satisfies model:
        y = X@beta + u
        E Z'u = 0
        Var(u) = sigma^2
        Cov(X,u) = gamma*sigma_u^2
        Var([X,Z}|u) = VXZ
        u,X,Z mean zero, Gaussian

    Each element of the tuple is an array of N observations.

    Inputs include
    - beta :: the coefficient of interest
    - gamma :: linear effect of disturbance on X
    - sigma_u :: Variance of disturbance
    - VXZ :: Cov([X,Z|u])
    """
    
    u = iid.norm.rvs(size=(N,1))*sigma_u

    # "Square root" of VXZ via eigendecomposition
    lbda,v = np.linalg.eig(VXZ)
    SXZ = v@np.diag(np.sqrt(lbda))

    # Generate normal random variates [X*,Z]
    XZ = iid.norm.rvs(size=(N,VXZ.shape[0]))@SXZ.T

    # But X is endogenous...
    X = XZ[:,[0]] + gamma*u
    Z = XZ[:,1:]

    # Calculate y
    y = X*beta + u

    return y,X,Z

def hful(y,X,Z): 
    
    #Define matrices
    # y,X,Z = data 
    P = Z@inv(Z.T@Z)@Z.T
    Xbar = np.c_[y,X]
    Pxbarsum = 0
    for i in range(P.shape[0]):
        Pxbarsum= Pxbarsum + P[i,i]*(Xbar[i,]@Xbar[i,].T)
    
    Pxsum = 0
    for i in range(P.shape[0]):
        Pxsum= Pxsum + P[i,i]*(X[i,]@X[i,].T)
    
    Pxysum = 0
    for i in range(P.shape[0]):
        Pxysum = Pxysum + P[i,i]*(X[i,]@y[i,])
    
    #Find the minimum eigenvalue 
    eigenval, v = np.linalg.eig(inv(Xbar.T@Xbar)@(Xbar.T@P@Xbar - Pxbarsum))
    
    alpha = np.min(eigenval)    
    alpha_hat = (alpha - (1 - alpha)/y.shape[0])/(1 - (1 - alpha)/y.shape[0])

    b = float(inv(X.T@P@X - Pxsum - alpha_hat*X.T@X)@(X.T@P@y - Pxysum - alpha_hat*X.T@y))
    return b


def iv(y,X,Z):

    # y,X,Z = data
    
    return float(inv(X.T@Z@inv(Z.T@Z)@Z.T@X)@(X.T@Z@inv(Z.T@Z)@Z.T@y))
#     bfirststage = np.linalg.inv(Z.T @ Z) @ Z.T @ X
#     xhat = Z @ bfirststage
#     return float(np.linalg.pinv(xhat.T @ xhat) @ xhat.T @ y)
    

# N = 2000 # Sample size

# D = 1000 # Monte Carlo draws


# b_draws = []
# for d in range(D):
#     b = hful(dgp(N,*truth))
#     b_draws.append(b)

# _ = plt.hist(b_draws,bins=int(np.ceil(np.sqrt(1000))), color='blue', alpha=0.2)
# _ = plt.axvline(beta,color='r')

# b_draws = []
# for d in range(D):
#     b = iv(dgp(N,*truth))
#     b_draws.append(b)
# _ = plt.hist(b_draws,bins=int(np.ceil(np.sqrt(1000))), color='green', alpha=0.2)

sample_sizes = [100, 1000, 2000]

fig, ax = plt.subplots(3, 3, figsize=(15, 7))

rowcount = 0
colcount = 0

for n in sample_sizes:
    for z in [1, 2, 5]:
        k = 1 # number of explanatory variables
        k_i = z # number of instruments
        
        hful_est = []
        iv_est = []

        for i in range(1000):
            y = np.random.rand(n, 1)
            X = np.concatenate((np.vstack(np.ones(n)), np.random.rand(n, k)), axis = 1)
            Z = np.concatenate((np.vstack(np.ones(n)), np.random.rand(n, k_i)), axis = 1)
            
            hful_est.append(hful(y, X, Z))
            iv_est.append(iv(y, X, Z))
        print("IV range: %4.4f %4.4f" % (min(iv_est), max(iv_est)))
        print("HFUL range: %4.4f %4.4f" % (min(hful_est), max(hful_est)))
        ax[rowcount, colcount].hist(hful_est, alpha = 0.5, label = 'HFUL')
        ax[rowcount, colcount].hist(iv_est, alpha = 0.5, label = 'IV')
        ax[rowcount, colcount].set_title("N = "+str(n)+", Z = "+str(z))
        
        colcount += 1
        
    rowcount += 1
    colcount = 0
    print("Completed sample size " + str(n))
        

plt.legend()

