import numpy as np
from numba import njit

@njit
def theo_mean(w, k, r, h):
    ### Compute expected average
    beta = r * (r + w*(k-1))

    xplus = (h[0]*r + w*k*(h[0]-h[1])) / beta
    xminus = (h[1]*(r-w) + w*h[0]) / beta
    return np.array( [xplus, xminus] )

@njit
def theo_sigma(w, k, r, D):
    ### Compute expected covariance matrix
    tt = (r+w*(k-1))*(2*r+w*(k-1))
    sigma = np.array([[2*r**2+(-1+3*k)*w*r+2*k**2*w**2, w*(r-k*r+2*k*w)],
                      [w*(r-k*r+2*k*w), 2*r**2+(k-3)*r*w+2*w**2]])
    
    return D * sigma / tt

@njit
def theo_W_factor(w, k, r, D):
    return (2*r + (-1 + k)*w) * (2*r**2 + (-1+3*k)*w*r + (1+k**2)*w**2) / (4*D*r) / (r + (-1+k)*w) / (2*r**2 + 2*(-1+k)*w*r +(1+k**2)*w**2)

@njit
def tmp_factor(w, k, r, D, h1, h2):
    mu1 = theo_mean(w, k, r, h1)
    mu2 = theo_mean(w, k, r, h2)
    sigma = theo_sigma(w, k, r, D)
    
    return (mu1-mu2).T @ np.linalg.inv(sigma) @ (mu1 - mu2)

@njit
def kl(w, k, r, D, h1, h2):
    return tmp_factor(w, k, r, D, h1, h2) / 2

@njit
def ch(w, k, h1, h2):
    return tmp_factor(w, k, r, D, h1, h2) / 8

@njit
def theo_bound(w, k, r, D, pi, hs, which):
    '''
    WARNING: valid only for the case of input to inhibitory = 0
    '''
    if which == 'lb':
        factor = 1 / 4
        
    factor *= theo_W_factor(w, k, r, D)
    
    n_inputs = len(pi)
    
    ress = 0
    for idx_i in range(n_inputs):
        tmp = 0
        for idx_j in range(n_inputs):
            tmp += pi[idx_j] * np.exp( -factor*(hs[0,idx_i]-hs[0,idx_j])**2 )
        ress += pi[idx_i] * np.log( tmp )
    return - ress
    
@njit
def mutual_time_dependent(w, k, sigmah, t):
    return ( 0.5*np.log((-4*np.exp((1 + k)*t*w)*(1 + k)*(1 + (-1 + k)*w)*(2 + (-1 + k)*w)*sigmah**2 + 
                    2*np.exp(2*t*w)*(2 + (-1 + k)*w)**2*sigmah**2 + 
                    np.exp(2*k*t*w)*(1 + k**2)*(1 + (-1 + k)*w)*(2 + (-1 + k)*w)**2*sigmah**2 + 
                    4*np.exp(t*(1 + w + k*w))*(-1 + k)*(2 + (-1 + k)*w)*(1 + k*w)*sigmah**2 - 
                    2*np.exp(t + 2*k*t*w)*(-1 + k)*(1 + (-1 + k)*w)*(2 + (-1 + k)*w)*(w + k*(2 + k*w))*sigmah**2 + 
                    np.exp(2*(t + k*t*w))*(-1 + k)**2*(2 + 4*sigmah**2 + (-1 + k)*(1 + k**2)*w**3*(1 + sigmah**2) +
                    4*w*(-1 + k + (-1 + 2*k)*sigmah**2) + w**2*(3 - 4*k + 3*k**2 + (3 + k*(-4 + 5*k))*sigmah**2)))/
                   (np.exp(2*(t + k*t*w))*(-1 + k)**2*(1 + (-1 + k)*w)*(2 + w*(-2 + w + k*(2 + k*w)))))
           )

@njit
def product_mutual_st_second_der(w, k, sigmah):
    return ( (0.5*(4. + w*(-8. + 1.*k**2*w + (7. - 2.*w)*w + k*(4. + w*(-4. + 2.*w))))*sigmah**2*
            (-np.log((1 + (-1 + k)*w)*(2 + w*(-2 + w + k*(2 + k*w)))) + 
            np.log(2 + 4*sigmah**2 + (-1 + k)*(1 + k**2)*w**3*(1 + sigmah**2) +
            4*w*(-1 + k + (-1 + 2*k)*sigmah**2) + 
            w**2*(3 - 4*k + 3*k**2 + (3 + k*(-4 + 5*k))*sigmah**2)))) /
            (2. + w*(-2. + w + k*(2. + k*w))) )

@njit
def mutual_second_der(w, k, sigmah):
    return  (4. + w*(-8. + 1.*k**2*w + (7. - 2.*w)*w + k*(4. + w*(-4. + 2.*w))))*sigmah**2 / (2. + w*(-2. + w + k*(2. + k*w)))

@njit
def mutual_stat_input(w, k, sigmah):
    return (0.5*
            (-np.log((1 + (-1 + k)*w)*(2 + w*(-2 + w + k*(2 + k*w)))) + 
            np.log(2 + 4*sigmah**2 + (-1 + k)*(1 + k**2)*w**3*(1 + sigmah**2) +
            4*w*(-1 + k + (-1 + 2*k)*sigmah**2) + 
            w**2*(3 - 4*k + 3*k**2 + (3 + k*(-4 + 5*k))*sigmah**2))))

@njit
def mutual_second_der_global(w, k, sigmah, r=1, D=1):
    return  (sigmah**2 / D) * (2 * r + (k - 1) * w)

@njit
def mutual_stat_input_global(w, k, sigmah, r=1, D=1):
    # First term
    term1 = -2 * np.log(D)
    
    # Second term
    term2 = -np.log(2 * r * (2 * r**2 + 2 * (k - 1) * r * w + (1 + k**2) * w**2))
    
    # Third term
    common_factor = 2 * r**2 + 2 * (k - 1) * r * w + (1 + k**2) * w**2
    
    term3 = np.log(
        2 * D**2 * r * common_factor + 
        (2 * D * (2 * r + (k - 1) * w) * common_factor * sigmah**2) / (r + (k - 1) * w) + 
        ((2 * r + (k - 1) * w)**2 * sigmah**4) / (r + (k - 1) * w)
    )
    
    # Final expression
    return 0.5 * (term1 + term2 + term3)
    
def theo_lb(w, k, r, D, pi, hs):
    n_inputs = len(pi)
    
    ress = 0
    for idx_i in range(n_inputs):
        tmp = 0
        for idx_j in range(n_inputs):
            tmp += pi[idx_j] * np.exp( -ch(w, k, r, D, hs[:,idx_i],hs[:,idx_j]) )
        ress += pi[idx_i] * np.log( tmp )
    return - ress

def theo_ub(w, k, r, D, pi, hs):
    n_inputs = len(pi)

    ress = 0
    for idx_i in range(n_inputs):
        tmp = 0
        for idx_j in range(n_inputs):
            tmp += pi[idx_j] * np.exp( -kl(w, k, r, D, hs[:,idx_i],hs[:,idx_j]) )
        ress += pi[idx_i] * np.log( tmp )
    return - ress