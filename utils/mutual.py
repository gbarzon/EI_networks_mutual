import numpy as np
from numba import njit
from scipy.special import rel_entr

from utils.analytical import *
from utils.numba_utils import *

################## COMPUTE MUTUAL FROM TRAJECTORIES ##################
@njit
def compute_bins(data, bins):
    x_lim, y_lim = (0, data[:,0].max()), (0, data[:,1].max())

    x_edges = np.linspace(x_lim[0], x_lim[1], bins + 1)
    y_edges = np.linspace(y_lim[0], y_lim[1], bins + 1)

    x_center = ( x_edges[1:]-x_edges[:-1] ) / 2
    y_center = ( y_edges[1:]-y_edges[:-1] ) / 2
    
    return (x_edges, y_edges), (x_center, y_center)

def compute_prob_2d(data, edges):
    return np.histogram2d(data[:,0],data[:,1], bins=(edges[0], edges[1]), density=True)[0]

def compute_mutual_information_from_trajectories(data, inputs, bins):
    ### Compute input statistics
    prob_input = np.unique(inputs, return_counts=True)[1]
    prob_input = prob_input / prob_input.sum()

    ### Compute bins
    edges, centers = compute_bins(data, bins)

    ### Compute bin size
    dx, dy = edges[0][1], edges[1][1]

    ### Compute complete state probability
    H_state = compute_prob_2d(data, edges=edges)

    ### Loop over input states
    kl_cond = np.zeros(prob_input.size)

    for idx in range(prob_input.size):
        ### Compute input times
        times_input = np.where(inputs == idx)[0]
    
        ### Compute conditional probabilities
        H_cond = compute_prob_2d(data[times_input], edges=edges)
    
        ### Compute KL
        kl_cond[idx] = rel_entr(H_cond, H_state).sum()
    
    ### Multiply KL by volume of integration elements
    kl_cond *= dx*dy

    ### Compute total mutual information
    return np.sum( prob_input * kl_cond )

################## COMPUTE MUTUAL WITH IMPORTANCE SAMPLING ##################
@njit
def system_probability_single(x, sigma_inv, sigma_det, x_means, p_inputs):
    pdf = 0
    
    for idx in range(x_means.shape[0]):
        pdf += p_inputs[idx] * numba_gaussian_2d(x, sigma_inv, sigma_det, x_means[idx])
    
    return pdf

@njit
def system_probability(x, sigma_inv, sigma_det, x_means, p_inputs):
    nsamples = x.shape[0]
    pdf = np.zeros((nsamples))
    
    for idx in range(nsamples):
        pdf[idx] = system_probability_single(x[idx], sigma_inv, sigma_det, x_means, p_inputs)
    
    return pdf

@njit
def MC_underhood(w, k, h_inputs, p_inputs, nsamples):
    ### Check stability
    if (k <= 1 - 1/w):
        return np.nan
    
    # Compute sigma stationary
    sigma_st = theo_sigma(w,k)
    sigma_inv = numba_inverse_2d(sigma_st)
    sigma_det = numba_determinant_2d(sigma_st)

    # Mean of ei system for different input values
    means = np.zeros(h_inputs.T.shape)
    for idx in range(h_inputs.shape[1]):
        means[idx] = theo_mean(w, k, h_inputs[:,idx])
    
    ### Generate samples from inputs
    nsamples_input = numba_sample_discrete_distribution(p_inputs, nsamples)
    
    ### Generate samples from system with specific inputs &&& compute joint pdf
    samples_system = np.zeros((nsamples, 2))
    pjoint = np.zeros(nsamples)
    pmarg = np.zeros(nsamples)
    
    idx_start = 0
    for idx_input in range(p_inputs.size):
        tmp_samples = nsamples_input[idx_input]
        
        # Generate samples
        tmp_samples_system = generate_2d_gaussian_samples(means[idx_input], sigma_st, tmp_samples)
        samples_system[idx_start:idx_start+tmp_samples] = tmp_samples_system
        
        # Compute joint pdf
        pjoint[idx_start:idx_start+tmp_samples] = nsamples_input[idx_input] / nsamples * numba_gaussian_2d_multiple(tmp_samples_system, sigma_inv, sigma_det, means[idx_input])
        
        # Compute system pdf
        pmarg[idx_start:idx_start+tmp_samples] = nsamples_input[idx_input] / nsamples * system_probability(tmp_samples_system, sigma_inv, sigma_det, means, p_inputs)
        
        idx_start += tmp_samples
    
    return np.sum(numba_masked_log(pjoint) - numba_masked_log(pmarg))/nsamples #, samples_system, pjoint, pmarg

@njit(parallel=False)
def mutual_information_slowjumps(w_list, k_list, h_inputs, p_inputs, nsamples = int(1e4)):
    mutual = np.empty((w_list.size, k_list.size), dtype = np.float64)

    for idx_w in prange(w_list.size):
        for idx_k in prange(k_list.size):
            mutual[idx_w,idx_k] = MC_underhood(w_list[idx_w], k_list[idx_k], h_inputs, p_inputs, nsamples)
                
    return mutual

@njit(parallel=False)
def mutual_information_slowjumps_fixedw(w, k_list, h_inputs, p_inputs, nsamples = int(1e4)):
    mutual = np.empty(k_list.size, dtype = np.float64)

    for idx_k in prange(k_list.size):
        mutual[idx_k] = MC_underhood(w, k_list[idx_k], h_inputs, p_inputs, nsamples)
                
    return mutual