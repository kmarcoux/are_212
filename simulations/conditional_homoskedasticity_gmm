#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 17:10:18 2021

@author: kendramarcoux
"""

import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt


## Play with us!
alpha = 2
beta = 1 # "Coefficient of interest"
gamma = 0 # Governs effect of u on X
sigma_u = 1 # Note assumption of homoskedasticity
## Play with us!

# Gimme some truth:
truth = (alpha,beta,gamma,sigma_u)

## But play with Omega if you want to introduce heteroskedascity
Omega = (sigma_u**2)


from scipy.stats import distributions as iid


def dgp(N,alpha,beta,gamma,sigma_u):
    """Generate a tuple of (y,X,Z).
      
    Satisfies model:
        y = a+X@beta + u
        E X'u = 0
        Var(u) = sigma^2
        Cov(X,u) = gamma*sigma_u^2
        u,X mean zero, Gaussian

    Each element of the tuple is an array of N observations.

    Inputs include
    - beta :: the coefficient of interest
    - gamma :: linear effect of disturbance on X
    - sigma_u :: Variance of disturbance
    """
    
    u = iid.norm.rvs(size=(N,1))*sigma_u
    
    # Generate normal random variates [X*,Z]
    XZ = iid.norm.rvs(size=(N,1))
    
    # But X is endogenous...
    X = XZ[:,[0]] + gamma*u
    
    # Calculate y
    y = alpha + X*beta + u
    
    return y,X

def gj(b,s,data):
    """Observations of g_j(b).
    
    This defines the deviations from the predictions of our model; i.e.,
    e_j = Z_ju_j, where EZ_ju_j=0.
    
    Can replace this function to testimate a different model.
    """
    y,X = data
    return np.hstack((X*(y - X*b), X*((y - X*b)**2)-X*(s**2)))

def gN(b,s,data):
    """Averages of g_j(b).
    
    This is generic for data, to be passed to gj.
    """
    e = gj(b,s,data)
    
    # Check to see more obs. than moments.
    assert e.shape[0] > e.shape[1]
    
    return e.mean(axis=0)

def Omegahat(b,s,data):
    y,X = data
    
# If we have that the optimal weighting matrix W=E[Z'Z]^{-1}\sigma^2 and Omegahat = W^{-1}
    e = gj(b,s,data)[:,:1]
    e = e - e.mean(axis=0)
    
    return np.vstack((np.hstack((e.T@e/(e.shape[0]), np.zeros((X.shape[1],X.shape[1])))),
                      np.hstack((np.zeros((X.shape[1],X.shape[1])), (e**2-s**2).T@(e**2-s**2)/e.shape[0]))))

N = 1000
data = dgp(N,*truth)
y,X = data
Winv = Omegahat(beta,sigma_u,data)

def J(b,s,W,data):
    
    m = gN(b,s,data) # Sample moments @ b, s
    N = data[0].shape[0]
    
    return N*m.T@W@m # Scale by sample size

# Limiting distribution under the null
limiting_J = iid.chi2(1*2-2)

import scipy.optimize as optimize

def two_step_gmm(data):
    
    # First step uses identity weighting matrix
    W1 = np.eye(gj(1,1,data).shape[1])
    x0 = [1,1]
    def J2(params):
        b,s = params
        return J(b,s,W1,data)
    
    result = optimize.minimize(J2, x0)
    b1, s1 = result.x
    
    # Construct 2nd step weighting matrix using
    # first step estimate of beta
    W2 = inv(Omegahat(b1,s1,data))
    
    def J3(params):
        b,s = params
        return J(b,s,W2,data)
    
    return optimize.minimize(J3, result.x)

soltn = two_step_gmm(data)
print("b=%f, s=%f, J=%f, Critical J=%f" % (soltn.x[0],soltn.x[1],soltn.fun,limiting_J.isf(0.05)))

D = 1000 # Monte Carlo draws

J_draws = []
for d in range(D):
    J_val = two_step_gmm(dgp(N,*truth)).fun
    J_draws.append(J_val)
    
_ = plt.hist(J_draws,bins=int(np.ceil(np.sqrt(1000))), color='green', alpha=0.2)

