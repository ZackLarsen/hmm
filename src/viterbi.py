
import numpy as np
from numpy import NINF, inf
import json
from collections import namedtuple
import os
import sys
from sklearn.metrics import cohen_kappa_score

data_dir = '/Users/zacklarsen/Zack_master/Projects/Work Projects/hmm/data'

# Handle the input sequences:
with open(os.path.join(data_dir, 'test_sequences.pickle'), 'rb') as f:
    test_sequences = pickle.load(f)

mid = [tup[0] for tup in test_sequences]
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


if __name__ == '__main__':
    for test_sequence in test_sequences:
        bestpathprob, viterbi_hidden_states = viterbi(test_sequence, transitions_matrix_log, emissions_matrix_log, Pi_log)
        kappa_score = cohen_kappa_score(hidden_states, viterbi_hidden_states)