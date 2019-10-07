
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

# Import the cython function we need for speedup
import pyximport
pyximport.install()
from viterbi import cython_viterbi
import viterbi
viterbi.cython_viterbi?






data_dir = '/Users/zacklarsen/Zack_Master/Projects/Work Projects/hmm/data'
homedir = '/Users/zacklarsen/Zack_Master/Projects/Work Projects/hmm'
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







test_tuple = namedtuple('test_tuple', 'seq_id kappa quad_kappa')
results = []

for mid in tqdm(test_mids):
    test_sequence = [tup[1] for tup in test_sequences if tup[0] == mid]
    hidden_states = [tup[2] for tup in test_sequences if tup[0] == mid]

    bestpathprob, viterbi_hidden_states = cython_viterbi(test_sequence, transitions_matrix_log, emissions_matrix_log, Pi_log)
    kappa_score = cohen_kappa_score(hidden_states, viterbi_hidden_states)
    quadratic_kappa_score = cohen_kappa_score(hidden_states, viterbi_hidden_states, weights='quadratic')
    results.append(test_tuple(mid, kappa_score, quadratic_kappa_score))

    print("Results being saved to", os.path.join(data_dir, 'results.pickle'))
    with open(os.path.join(data_dir, 'results.pickle'), 'wb') as f:
        pickle.dump(results, f)

    avg_kappas = []
    avg_quad_kappas = []
    for tup in results:
        avg_kappas.append(tup.kappa)
        avg_quad_kappas.append(tup.quad_kappa)
    print("Average kappa score is:", np.mean(avg_kappas))
    print("Average quadratic-weighted kappa score is:", np.mean(avg_quad_kappas))