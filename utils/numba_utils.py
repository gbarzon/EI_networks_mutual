import numpy as np
from numba import njit, prange

@njit
def numba_masked_log(x):
    l = np.log(x)
    l[np.where(l == -np.inf)] = 0

    return l

@njit
def numba_determinant_2d(mat):
    return mat[0,0]*mat[1,1] - mat[0,1]*mat[1,0]

@njit
def numba_inverse_2d(mat):
    return np.array([ [mat[1,1], -mat[1,0]], [-mat[0,1], mat[0,0]] ]) / numba_determinant_2d(mat)

@njit
def numba_gaussian_2d(x, sigma_inv, sigma_det, mean):
    return np.exp(-0.5*(x-mean)@sigma_inv@(x-mean)) / (2*np.pi*sigma_det**(1/2))

@njit
def numba_gaussian_2d_multiple(x, sigma_inv, sigma_det, mean):
    nsamples = x.shape[0]
    ress = np.zeros(nsamples)
    
    for idx in range(nsamples):
        ress[idx] = numba_gaussian_2d(x[idx], sigma_inv, sigma_det, mean)
    
    return ress

@njit
def generate_2d_gaussian_samples(mean, cov_matrix, num_samples):
    """
    Generate random samples from a 2D Gaussian distribution using Numba.

    Parameters:
        mean (array_like): Mean vector of the distribution.
        cov_matrix (array_like): Covariance matrix of the distribution.
        num_samples (int): Number of samples to generate.

    Returns:
        array: Random samples from the 2D Gaussian distribution.
    """
    
    dim = len(mean)
    samples = np.empty((num_samples, dim), dtype=np.float64)
    L = np.linalg.cholesky(cov_matrix)
    random_numbers = np.empty(2)  # Preallocate array for random numbers
    
    for i in range(num_samples):
        for j in range(dim):
            random_numbers[j] = np.random.normal()
        z = random_numbers[:dim]
        x = mean + np.dot(L, z)
        samples[i] = x
        
    return samples

@njit
def numba_sample_discrete_distribution(probs, nsamples):
    cdf = np.cumsum(probs)
    
    sizes = np.zeros(probs.size, dtype=np.int_)
    rands = np.random.rand(nsamples)
    
    for idx in range(nsamples):
        sizes[np.where(rands[idx]<cdf)[0][0]] += 1
        
    return sizes