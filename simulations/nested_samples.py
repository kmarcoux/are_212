#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  9 16:40:04 2021

@author: kendramarcoux
"""
import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt
import statistics as stats

beta = 1 # "Coefficient of interest"
gamma = 1 # Governs effect of u on X
sigma_u = 1 # Note assumption of homoskedasticity
## Play with us!

# Let Z have order ell, and X order 1, with Var([X,Z]|u)=VXZ
ell = 1

# Arbitrary (but deterministic) choice for VXZ = [VX Cov(X,Z);
#                                                 Cov(Z,X) VZ]
# Pinned down by choice of a matrix A...
A = np.sqrt(1/np.arange(1,(ell+1)**2+1)).reshape((ell+1,ell+1))

## Below here we're less playful.

# Now Var([X,Z]|u) is constructed so guaranteed pos. def.
VXZ = A.T@A

Q = -VXZ[1:,[0]] # -EZX', or generally Edgj/db'

# Gimme some truth:
truth = (beta,gamma,sigma_u,VXZ)


## But play with Omega if you want to introduce heteroskedascity
Omega = (sigma_u**2)*VXZ[1:,1:] # E(Zu)(u'Z')

from scipy.stats import distributions as iid

def dgp(N1,N2,beta,gamma,sigma_u,VXZ):

    u2 = iid.norm.rvs(size=(N2,1))*sigma_u
    
    # "Square root" of VXZ via eigendecomposition
    lbda,v = np.linalg.eig(VXZ)
    SXZ = v@np.diag(np.sqrt(lbda))
    
    # Generate normal random variates [X*,Z]
    XZ = iid.norm.rvs(size=(N2,VXZ.shape[0]))@SXZ.T
    
    # But X is endogenous...
    X2 = XZ[:,[0]] + gamma*u2
    Z2 = XZ[:,1:]
    
    X1 = X2[:N1,]
    Z1 = Z2[:N1,]
    u1 = u2[:N1,]
    
    # Calculate y
    y = X1*beta + u1
    
    return y,X1,Z1,X2,Z2

def iv2(data):
    
    y,X1,Z1,X2,Z2 = data
    
    pi_hat = inv(Z2.T@Z2)@Z2.T@X2
    xhat = Z2@pi_hat
    
    xhat1 = xhat[:y.shape[0],]
    var_x1hat = np.var(xhat1)
    cov = np.cov(xhat1, y, rowvar=0)[0,1]
    
    beta_hat = inv(xhat1.T@xhat1)@(xhat1.T@y)
    
    return float(beta_hat),float(pi_hat),float(cov)

def iv1(data):
    
    y,X1,Z1,X2,Z2 = data
    
    pi_hat = inv(Z1.T@Z1)@Z1.T@X1
    xhat = Z1@pi_hat
    var_xhat =  np.var(xhat)
    cov = np.cov(xhat, y, rowvar=0)[0,1]
    
    beta_hat = inv(xhat.T@xhat)@(xhat.T@y)
    
    return float(beta_hat),float(pi_hat),float(cov)


D = 10000 # Monte Carlo draws
N1 = 400
N2 = 1000

b2_draws = []
pi2_draws = []
var_draws = []
for d in range(D):
    b2, pi2, var = iv2(dgp(N1,N2,*truth))
    b2_draws.append(b2)
    pi2_draws.append(pi2)
    var_draws.append(var)
    
_ = plt.hist(var_draws,bins=int(np.ceil(np.sqrt(1000))), color='blue', alpha=0.2)
_ = plt.axvline(beta,color='r')

b1_draws = []
pi1_draws = []
var_draws = []
for d in range(D):
    b1, pi1, var = iv1(dgp(N1,N2,*truth))
    b1_draws.append(b1)
    pi1_draws.append(pi1)
    var_draws.append(var)
    
_ = plt.hist(var_draws,bins=int(np.ceil(np.sqrt(1000))), color='green', alpha=0.2)


    
    
    
