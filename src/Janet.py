
## Project: Hidden Markov Models for pos tagging
## Author: Zack Larsen
## Date: July 31, 2019
## Script purpose: Create POS tagger using hidden markov model (HMM)
##      and apply it to the Janet example from Speech & Language Processing textbook

import numpy as np
from numba import jit
import time


@jit(nopython=True)
def viterbi(observations, states):
    '''
    Compute the optimal sequence of hidden states
    :param observations:
    :param states:
    :return: bestpath, bestpathprob
    '''
    N = emissions.shape[0]
    T = len(observations)

    # Initialize
    viterbi_trellis = np.zeros([N, T])
    backpointer = np.zeros([N, T])
    for s in range(0, N):
        viterbi_trellis[s, 0] = Pi[s] * emissions[s, observations[0]]
        backpointer[s, 0] = 0

    # Recursion
    for time_step in range(1, T):
        for cs in range(0, N):
            priors = np.zeros([N, 2])
            for ps in range(0, N):
                priors[ps, 0] = ps
                priors[ps, 1] = viterbi_trellis[ps, time_step - 1] * \
                                transitions[ps, cs] * \
                                emissions[cs, observations[time_step] - 1]
            # print(priors)
            viterbi_trellis[cs, time_step] = np.amax(priors[:, 1], axis=0)
            backpointer[cs, time_step] = np.argmax(priors[:, 1], axis=0)

    # Termination
    bestpathprob = np.amax(viterbi_trellis[:,-1])
    bestpathpointer = np.argmax(viterbi_trellis[:,-1])
    bestpath = '''the path starting at state bestpathpointer that traverses backpointer to 
        states back in time'''

    return bestpath, bestpathprob


def token_to_int(tokens):
    '''
    Take a list of tokens and map them to integers
    :param token_list: List of tokens
    :return: Dictionary mapping each token to a unique integer
    '''
    unique_token_list = np.unique(tokens)
    unique_token_list.sort(kind='quicksort')
    token_map = {token: value for token, value in zip(unique_token_list, range(0,len(unique_token_list)))}
    return token_map


def reverse_token_map(token_map):
    '''
    Reverse the order of the dictionary for the tokens
    :param token_map: The result of the token_to_int(tokens) function
    :return: A reversed version of the token map dictionary
    '''
    reversed_dictionary = {value: key for key, value in token_map.items()}
    return reversed_dictionary








# Transition probability matrix
transitions = np.array([[0.2767,0.0006,0.0031,0.0453,0.0449,0.051,0.2026],
    [0.3777,0.011,0.0009,0.0084,0.0584,0.009,0.0025],
    [0.0008,0.0002,0.7968,0.0005,0.0008,0.1698,0.0041],
    [0.0322,0.0005,0.005,0.0837,0.0615,0.0514,0.2231],
    [0.0366,0.0004,0.0001,0.0733,0.4509,0.0036,0.0036],
    [0.0096,0.0176,0.0014,0.0086,0.1216,0.0177,0.0068],
    [0.0068,0.0102,0.1011,0.1012,0.012,0.0728,0.0479],
    [0.1147,0.0021,0.0002,0.2157,0.4744,0.0102,0.0017]])

transitions.shape
transitions

# Emission probability matrix
emissions = np.array([[0.000032,0,0,0.000048,0],
    [0,0.308431,0,0,0],
    [0,0.000028,0.000672,0,0.000028],
    [0,0,0.00034,0,0],
    [0,0.0002,0.000223,0,0.002337],
    [0,0,0.010446,0,0],
    [0,0,0,0.506099,0]
    ])

emissions.shape
emissions

# Initial probabilities
Pi = np.array([0.2767,
    0.0006,
    0.0031,
    0.0453,
    0.0449,
    0.051,
    0.2026
    ])

Pi.shape
Pi



token_to_int(['Janet','will','back','the','bill'])
token_to_int(['NNP','MD','VB','DT','NN'])



states = [value for value in token_to_int(['NNP','MD','VB','DT','NN']).values()]
states

observations = [value for value in token_to_int(['Janet','will','back','the','bill']).values()]
observations



N = emissions.shape[0]
T = len(observations)





# Initialize
viterbi_trellis = np.zeros([N, T])
backpointer = np.zeros([N, T])
for s in range(0,N):
    viterbi_trellis[s, 0] = Pi[s] * emissions[s, observations[0]]
    backpointer[s, 0] = 0

# Recursion
all_priors = np.zeros([T-1,N,N,2])
for time_step in range(1, T):
    for cs in range(0, N):
        priors = np.zeros([N, 2])
        for ps in range(0, N):
            priors[ps, 0] = ps
            priors[ps, 1] = viterbi_trellis[ps, time_step - 1] * \
                            transitions[ps, cs] * \
                            emissions[cs, observations[time_step] - 1]
        viterbi_trellis[cs, time_step] = np.amax(priors[:,1], axis=0)
        backpointer[cs, time_step] = np.argmax(priors[:,1], axis=0)
        all_priors[time_step - 1] = priors
# Termination
bestpathprob = np.amax(viterbi_trellis[:,-1])
bestpathpointer = np.argmax(viterbi_trellis[:,-1])





viterbi_trellis
backpointer

bestpathprob
bestpathpointer

all_priors



bestpath = '''the path starting at state bestpathpointer that traverses backpointer to 
        states back in time'''











all_priors = np.zeros([T-1,N,N,2])
for time_step in range(1, T):
    for cs in range(0, N):
        priors = np.ones([N, 2])
        for ps in range(0, N):
            priors[ps, 0] = ps
            priors[ps, 1] = viterbi_trellis[ps, time_step - 1] * \
                            transitions[ps, cs] * \
                            emissions[cs, observations[time_step] - 1]
        all_priors[time_step-1, cs] = priors



all_priors.shape

all_priors

all_priors[0].shape
all_priors[0][0].shape

all_priors[3][6]
all_priors[3][6][:,1]

np.amax(all_priors[3][6][:,1], axis=0)
np.argmax(all_priors[3][6][:,1], axis=0)


