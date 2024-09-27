import numpy as np

import sys
sys.path.append('../')

from utils.simulation import *
from utils.analytical import *

import utils.npeet as ee


def get_default_params_system():
    ### Info system
    N = 2
    w = 4.
    k = 1.1
    tau_ei = 1
    r = np.ones(N)
    D = 0.001
    
    return {'N': N, 'w': w, 'k': k, 'tau_ei': tau_ei, 'r': r, 'D': D}

def get_default_params_sims():
    ### Info simulation
    steps = int(1e6)
    dt = 0.005
    
    return {'steps': steps, 'dt': dt}



delta_h = 0.15
ks = np.linspace(0, 4, 7)
ws = [1, 4, 10]

n_runs = 5
NSteps = int(5e6)
dt = 0.01

wup = 0.0005
wdown = 0.0005
M = 3

hs = np.arange(M) * delta_h
hs = np.stack([hs, np.zeros(M)])
W = create_transition_matrix_star_graph(M, wup, wdown)
info_input = create_info_input(W, hs)

p_stat = np.array([wdown] + [wup] * (M-1))
p_stat = p_stat / p_stat.sum()

params_sims = get_default_params_sims()
info_simulation = create_info_simulation(**params_sims)

info_simulation['steps'] = NSteps
info_simulation['dt'] = dt

steps_skip = 10
initial_transient = int(1e4)

for w in ws:
    print(f'[*] w={w}')
    params = get_default_params_system()
    params['w'] = w

    res_linear = np.zeros((len(ks),n_runs))
    res_nonlinear = np.zeros((len(ks),n_runs))

    for idx_k, k in enumerate(ks):
        print(f'\t [*] i={idx_k+1}/{ks.size}')
        params['k'] = k
        info_system = create_info_system(**params)

        for idx_run in range(n_runs):
            print(f'\t \t [*] run={idx_run+1}/{n_runs}')
            inputs_slow, states_slow = simulate_coupled_system(info_system, info_input, info_simulation, seed=None)
            inputs_slow_nonlinear, states_slow_nonlinear = simulate_coupled_system(info_system, info_input, info_simulation, seed=None, linear=False)

            try:
                res_linear[idx_k,idx_run] = ee.micd(states_slow[initial_transient:][::steps_skip], inputs_slow[initial_transient:][::steps_skip].reshape(-1,1))
            except:
                res_linear[idx_k,idx_run] = np.nan
            try:
                res_nonlinear[idx_k,idx_run] = ee.micd(states_slow_nonlinear[initial_transient:][::steps_skip], inputs_slow_nonlinear[initial_transient:][::steps_skip].reshape(-1,1))
            except:
                res_nonlinear[idx_k,idx_run] = np.nan
    
    np.save(f'../data/nonlinear_new_sim/mutual_linear_w_{w}_wup_{wup}_dh_{delta_h}_k_{ks[0]}_{ks[-1]}_{ks.size}_NSteps_{NSteps}_dt_{dt}.npy', res_linear)
    np.save(f'../data/nonlinear_new_sim/mutual_nonlinear_w_{w}_wup_{wup}_dh_{delta_h}_k_{ks[0]}_{ks[-1]}_{ks.size}_NSteps_{NSteps}_dt_{dt}.npy', res_nonlinear)