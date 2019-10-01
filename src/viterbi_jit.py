
# Author: Zack Larsen
# Date: August 31, 2019
# Decode most likely sequence of hidden states

import numpy as np
from numpy import NINF, inf
import json
import pickle
from collections import namedtuple
import os
import sys
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm

homedir = '/Users/zacklarsen/Zack_Master/Projects/Work Projects/hmm'
data_dir = os.path.join(homedir, 'data')
sys.path.append(os.path.join(homedir, 'src/'))
from hmm import *


# Handle the input sequences:
with open(os.path.join(data_dir, 'test_tuples.pickle'), 'rb') as f:
    test_sequences = pickle.load(f)

mid = [tup[0] for tup in test_sequences]
test_mids = list(set(mid))
obs = [tup[1] for tup in test_sequences]
hidden_states = [tup[2] for tup in test_sequences]

# Load integer maps:
with open(os.path.join(data_dir, 'observation_map.json'), 'r') as fp:
    observation_map = json.load(fp)

with open(os.path.join(data_dir, 'state_map.json'), 'r') as fp:
    state_map = json.load(fp)

# Load log10 probability matrices:
log_matrices = np.load(os.path.join(data_dir, 'log_matrices.npz'))
transitions_matrix_log = log_matrices['transitions']
emissions_matrix_log = log_matrices['emissions']
Pi_log = log_matrices['Pi']





@jit(nopython=True)
def viterbi_pyjit(observations, transitions_matrix, emissions_matrix, Pi):
    '''
    Compute the optimal sequence of hidden states

    :param observations: List of tokens in sequence
    :param states: List of tags in sequence
    :param transitions_matrix: Matrix of probability of tag given previous tag
    :param emissions_matrix: Matrix of probability of tag given token
    :param Pi: Initial probability vector

    :return: bestpath, bestpathprob
    '''

    # Initialization
    N = emissions_matrix.shape[0] # Number of states (hidden states)
    T = len(observations) # Number of observations (observations)
    viterbi_trellis = np.ones((N, T)) * np.NINF
    backpointer = np.zeros_like(viterbi_trellis)
    for s in range(0, N):
        viterbi_trellis[s, 0] = Pi[s] + emissions_matrix[s, observations[0]]

    # Recursion
    for time_step in range(1, T):
        for current_state in range(0, N):
            priors = np.zeros((N, 2))
            for previous_state in range(0, N):
                priors[previous_state, 0] = previous_state
                priors[previous_state, 1] = viterbi_trellis[previous_state, time_step - 1] + \
                                            transitions_matrix[previous_state, current_state] + \
                                            emissions_matrix[current_state, observations[time_step]] # Previously, we were using timestep-1 here
            viterbi_trellis[current_state, time_step] = np.amax(priors[:, 1])
            #viterbi_trellis[current_state, time_step] = max(priors[:, 1])
            backpointer[current_state, time_step] = np.argmax(priors[:, 1])
            #backpointer[current_state, time_step] = max(zip(priors[:, 1], range(len(priors[:, 1]))))[1]

    # Termination
    bestpathprob = np.amax(viterbi_trellis[:, -1])
    bestpathpointer = np.argmax(viterbi_trellis[:, -1])
    #print(bestpathprob, bestpathpointer)

    viterbi_cell_idx = np.zeros_like(viterbi_trellis[0,:])
    for i in range(0, viterbi_trellis.shape[1]):
        viterbi_cell_idx[i] = np.argmax(viterbi_trellis[:,i])
        #print(viterbi_cell_idx[i])

    viterbi_hidden_states = []
    for ix, i in enumerate(viterbi_cell_idx):
        viterbi_hidden_states.append(backpointer[i, ix])
    viterbi_hidden_states = np.delete(viterbi_hidden_states, 0)  # We don't need the first entry - it is always zero
    viterbi_hidden_states = np.append(viterbi_hidden_states, bestpathpointer) # Add the bestpathpointer to the last entry

    return bestpathprob, viterbi_hidden_states



if __name__ == '__main__':
    test_tuple = namedtuple('test_tuple', 'seq_id kappa quad_kappa')
    results = []

    # Initialize viterbi_jit
    test_sequence = [tup[1] for tup in test_sequences if tup[0] == test_mids[0]]
    hidden_states = [tup[2] for tup in test_sequences if tup[0] == test_mids[0]]
    viterbi_pyjit(test_sequence, transitions_matrix_log, emissions_matrix_log, Pi_log)

    print("Decoding test sequences...")
    for mid in tqdm(test_mids):
        test_sequence = [tup[1] for tup in test_sequences if tup[0]==mid]
        hidden_states = [tup[2] for tup in test_sequences if tup[0]==mid]

        #bestpathprob, viterbi_hidden_states = viterbi(test_sequence, transitions_matrix_log, emissions_matrix_log, Pi_log)
        bestpathprob, viterbi_hidden_states = viterbi_pyjit(test_sequence, transitions_matrix_log, emissions_matrix_log, Pi_log)
        #viterbi_pyjit(test_sequence, transitions_matrix_log, emissions_matrix_log, Pi_log)

        #kappa_score = cohen_kappa_score(hidden_states, viterbi_hidden_states)
        #quadratic_kappa_score = cohen_kappa_score(hidden_states, viterbi_hidden_states, weights='quadratic')
        #results.append(test_tuple(mid, kappa_score, quadratic_kappa_score))

    # print("Results being saved to", os.path.join(data_dir, 'results.pickle'))
    # with open(os.path.join(data_dir, 'results.pickle'), 'wb') as f:
    #     pickle.dump(results, f)
    #
    # avg_kappas = []
    # avg_quad_kappas = []
    # for tup in results:
    #     avg_kappas.append(tup.kappa)
    #     avg_quad_kappas.append(tup.quad_kappa)
    # print("Average kappa score is:", np.mean(avg_kappas))
    # print("Average quadratic-weighted kappa score is:", np.mean(avg_quad_kappas))