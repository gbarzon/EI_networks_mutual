import numpy as np
from numba import njit

@njit
def theo_mean(w, k, h):
    ### Compute expected average
    beta = 1 + w*(k-1)

    xplus = (h[0] + w*k*(h[0]-h[1])) / beta
    xminus = (h[1] + w*(h[0]-h[1])) / beta
    return np.array( [xplus, xminus] )

@njit
def theo_sigma(w, k):
    ### Compute expected covariance matrix
    tt = 2*(1+w*(k-1))*(2+w*(k-1))
    sigma = np.array([[2+w*(-1+k*(3 + 2*k*w)), w+k*w*(2*w-1)],[w+k*w*(2*w-1), 2+w*(-3+k+2*w)]])
    
    return sigma / tt

@njit
def theo_W_factor(w, k):
    return (2 + (-1 + k)*w) * (2 + (-1+3*k)*w + (1+k**2)*w**2) / (1 + (-1+k)*w) / (2 + 2*(-1+k)*w +(1+k**2)*w**2)

@njit
def tmp_factor(w, k, h1, h2):
    mu1 = theo_mean(w, k, h1)
    mu2 = theo_mean(w, k, h2)
    sigma = theo_sigma(w, k)
    
    return (mu1-mu2).T @ np.linalg.inv(sigma) @ (mu1 - mu2)

@njit
def kl(w, k, h1, h2):
    return tmp_factor(w, k, h1, h2) / 2

@njit
def ch(w, k, h1, h2):
    return tmp_factor(w, k, h1, h2) / 8

@njit
def theo_bound(w, k, pi, hs, which):
    '''
    WARNING: valid only for the case of input to inhibitory = 0
    '''
    if which == 'lb':
        factor = 1 / 8
    elif which == 'ub':
        factor = 1 / 2
        
    factor *= theo_W_factor(w, k)
    
    n_inputs = len(pi)
    
    ress = 0
    for idx_i in range(n_inputs):
        tmp = 0
        for idx_j in range(n_inputs):
            tmp += pi[idx_j] * np.exp( -factor*(hs[0,idx_i]-hs[0,idx_j])**2 )
        ress += pi[idx_i] * np.log( tmp )
    return - ress
    
def theo_lb(w, k, pi, hs):
    n_inputs = len(pi)
    
    ress = 0
    for idx_i in range(n_inputs):
        tmp = 0
        for idx_j in range(n_inputs):
            tmp += pi[idx_j] * np.exp( -ch(w, k, hs[:,idx_i],hs[:,idx_j]) )
        ress += pi[idx_i] * np.log( tmp )
    return - ress

def theo_ub(w, k, pi, hs):
    n_inputs = len(pi)

    ress = 0
    for idx_i in range(n_inputs):
        tmp = 0
        for idx_j in range(n_inputs):
            tmp += pi[idx_j] * np.exp( -kl(w, k, hs[:,idx_i],hs[:,idx_j]) )
        ress += pi[idx_i] * np.log( tmp )
    return - ress