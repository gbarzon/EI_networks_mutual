import numpy as np
from numba import njit
from scipy.special import rel_entr

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

def compute_mutual_information(data, inputs, bins):
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