import numpy as np
from numpy.linalg import inv
from scipy.stats import distributions as iid
from matplotlib import pyplot as plt
%matplotlib inline

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

def k_class(data, k_type): 
    
    #Define matrices
    y,X,Z = data 
    Mz = np.identity(N)-Z@inv(Z.T@Z)@Z.T
    W = np.c_[y,X]
    
    #Find the minimum eigenvalue - k=λ in LIML and k=1 in 2SLS
    eigenval, v = np.linalg.eig(W.T@Mz@W@inv(W.T@Mz@W))
    [i.real for i in eigenval]
    
    if k_type == "LIML":
        k = np.min([i.real for i in eigenval])    
    elif k_type=="OLS":
        k=0
    else:
        k = 1
    k_class_b = np.float(inv(X.T@(np.identity(N)-k*Mz)@X)@X.T@(np.identity(N)-k*Mz)@y)
    print(eigenval)
    return k_class_b

N = 100 # Sample size

D = 1000 # Monte Carlo draws

bias_draws = []
eigenvals = []
for d in range(D):
    k_class_b = k_class(dgp(N,*truth), "2SLS")
    bias = beta - k_class_b
    bias_draws.append(bias)

_ = plt.hist(bias_draws,bins=int(np.ceil(np.sqrt(1000))), color='blue', alpha=0.2,label = "2SLS")
_ = plt.axvline(0,color='r')

bias_draws = []
for d in range(D):
    k_class_b = k_class(dgp(N,*truth), "LIML")
    bias = beta - k_class_b
    bias_draws.append(bias)
_ = plt.hist(bias_draws,bins=int(np.ceil(np.sqrt(1000))), color='green', alpha=0.2,label = "LIML")
_ = plt.axvline(0,color='r')

bias_draws = []
for d in range(D):
    k_class_b = k_class(dgp(N,*truth), "OLS")
    bias = beta - k_class_b
    bias_draws.append(bias)
_ = plt.hist(bias_draws,bins=int(np.ceil(np.sqrt(1000))), color="red", alpha=0.2,label = "LIML")
_ = plt.axvline(0,color='r')