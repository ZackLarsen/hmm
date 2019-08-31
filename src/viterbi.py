
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
sys.path.append(os.path.join(homedir, 'src/'))
from hmm import *

data_dir = '/Users/zacklarsen/Zack_master/Projects/Work Projects/hmm/data'

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


if __name__ == '__main__':
    test_tuple = namedtuple('test_tuple', 'seq_id kappa')
    results = []

    print("Decoding test sequences...")
    for mid in tqdm(test_mids):
        test_sequence = [tup[1] for tup in test_sequences if tup[0]==mid]
        hidden_states = [tup[2] for tup in test_sequences if tup[0]==mid]
        bestpathprob, viterbi_hidden_states = viterbi(test_sequence, transitions_matrix_log, emissions_matrix_log, Pi_log)
        kappa_score = cohen_kappa_score(hidden_states, viterbi_hidden_states, )
        results.append(test_tuple(mid, kappa_score))

    print("Results being saved to", os.path.join(data_dir, 'results.pickle'))
    with open(os.path.join(data_dir, 'results.pickle'), 'wb') as f:
        pickle.dump(results, f)

    avg_kappas = []
    for tup in results:
        avg_kappas.append(tup.kappa)
    print("Average kappa score is:", np.mean(avg_kappas))