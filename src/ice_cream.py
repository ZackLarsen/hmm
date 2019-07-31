
## Project: Hidden Markov Models for pos tagging
## Author: Zack Larsen
## Date: July 20, 2019
## Script purpose: Create POS tagger using hidden markov model (HMM)
##      and apply it to Jason Eisner's ice cream example

import numpy as np
from numba import jit
import time


def forward(observations, states):
    '''
    Note: This algorithm is intractible due to the number of calculations that
    must be performed. The standard way to find the optimal path of a hidden markov
    model is the Viterbi decoding algorithm, defined in a function viterbi() below.

    Given a sequence of observations q, calculate the forward probabilities
    of each state in the sequence. Return the most probable path.
    Consists of three major steps:
        1. Initialization
        2. Recursion
        3. Termination

    :param observations: Length T
    :param states: Length N
    :return Forward probability for path
    '''
    # Initialization step: create forward probability matrix [N,T]
    # and fill the first column with the initial probabilities
    N = len(observations)
    T = len(states)
    forward_trellis = np.zeros([N,T])
    # For each state s from 1 to N:
    #   forward[s,1] ← πs ∗ bs(o1)
    #   the forward matrix entry at [s,1] will be equal to
    #   Pi(s) * b[s](o1), which means the initial probability
    #   of state s (P(s|START)) times the emission probability
    #   matrix at entry [s,observation 1] for P(s| observation 1)
    for s, state in enumerate(states):
        forward_trellis[s, 1] = Pi[s-1] * emissions[s, observations[1]]

    # Recursion step: fill the remaining columns in the trellis
    # for each time step t from 2 to T:
        # for each state s from 1 to N:
            # forward_trellis[s, t] =
            # Sum from s'=1 to N of forward_trellis[s′, t−1] ∗ a[s', s] ∗ b[s, (ot)]
    for t in range(2,T):
        for s in range(1,N):
            forward_trellis[s, t] =

    # Termination step:
    # sum from s=1 to N of forward_trellis[s, T]
    forwardprob = forward_trellis[:, T]
    return forwardprob


@jit(nopython=True)
def viterbi(observations, states):
    '''
    Compute the optimal sequence of hidden states
    :param observations:
    :param states:
    :return: bestpath, bestpathprob
    '''
    T = len(observations)
    N = len(states)
    viterbi_trellis = np.empty([N, T])
    backpointer = np.empty([N, T])

    # Initialize
    for s, state in enumerate(states):
        viterbi_trellis[s, 0] = Pi[s] * emissions[observations[0] - 1, s]
        backpointer[s, 0] = 0

    # Recursion
    for time_step in range(1, T):
        for cs, current_state in enumerate(states):
            priors = np.zeros([N, N])
            for ps, previous_state in enumerate(states):
                # print(time_step, cs, current_state, ps, previous_state)
                priors[ps, cs] = viterbi_trellis[ps, time_step - 1] * \
                                 transitions[cs, ps] * \
                                 emissions[observations[time_step] - 1, cs]
            # print(priors)
            viterbi_trellis[cs, time_step] = np.amax(priors[:, 0], axis=0)
            backpointer[cs, time_step] = np.argmax(priors[:, 0], axis=0)

    # Termination
    bestpathprob = np.amax(viterbi_trellis[:,-1])
    bestpathpointer = np.argmax(viterbi_trellis[:,-1])
    bestpath = '''the path starting at state bestpathpointer that traverses backpointer to 
        states back in time'''

    return bestpath, bestpathprob






# Transition probability matrix
transitions = np.array([[0.5,0.3,0.2],
              [0.4,0.6,0.8],
              [0.1,0.1,0]])
transitions

# Emission probability matrix
emissions = np.array([[0.5,0.2],
              [0.4,0.4],
              [0.1,0.4]])
emissions

# Initial probabilities
Pi = np.array([0.2, 0.8])
Pi

states = ['Cold','Hot']
observations = [3,1,3]
N = len(states)
T = len(observations)


viterbi_trellis = np.zeros([N, T])
backpointer = np.zeros([N, T])

for s, state in enumerate(states):
    print(s, state)
# 0 Cold
# 1 Hot



# s = 0, state = Cold
viterbi_trellis[s, 0] = P(s|START) * emissions(observation, state)
viterbi_trellis[0, 0] = Pi[s] * emissions[observations[0], 'Cold']
viterbi_trellis[0, 0] = Pi[s] * emissions['3', 'Cold']
viterbi_trellis[0, 0] = Pi[s] * emissions['3', s]
viterbi_trellis[0, 0] = Pi[0] * emissions[2, 0]

# s = 1, state = Hot
viterbi_trellis[s, 0] = P(s|START) * emissions(observation, state)
viterbi_trellis[1, 0] = Pi[s] * emissions[observations[0], 'Hot']
viterbi_trellis[1, 0] = Pi[s] * emissions['3', 'Hot']
viterbi_trellis[1, 0] = Pi[s] * emissions['3', s]
viterbi_trellis[1, 0] = Pi[1] * emissions[2, 1]



# Initialize
for s, state in enumerate(states):
    viterbi_trellis[s, 0] = Pi[s] * emissions[observations[0]-1, s]
    backpointer[s, 0] = 0


viterbi_trellis
backpointer






# Recursion
for time_step in range(1, T):
    for cs, current_state in enumerate(states):
        priors = np.zeros([N,N])
        for ps, previous_state in enumerate(states):
            #print(time_step, cs, current_state, ps, previous_state)
            priors[ps, cs] = viterbi_trellis[ps, time_step - 1] * \
                             transitions[cs, ps] * \
                             emissions[observations[time_step]-1, cs]
        #print(priors)
        viterbi_trellis[cs, time_step] = np.amax(priors[:,0], axis=0)
        backpointer[cs, time_step] = np.argmax(priors[:,0], axis=0)


viterbi_trellis
backpointer



# Termination
bestpathprob = np.amax(viterbi_trellis[:,-1])
bestpathpointer = np.argmax(viterbi_trellis[:,-1])






# Find best path:
bestpathprob
bestpathpointer
















def viterbi(y, A, B, Pi=None):
    """
    https://stackoverflow.com/questions/9729968/python-implementation-of-viterbi-algorithm
    Return the MAP estimate of state trajectory of Hidden Markov Model.

    Parameters
    ----------
    y : array (T,)
        Observation state sequence. int dtype.
    A : array (K, K)
        State transition matrix. See HiddenMarkovModel.state_transition  for
        details.
    B : array (K, M)
        Emission matrix. See HiddenMarkovModel.emission for details.
    Pi: optional, (K,)
        Initial state probabilities: Pi[i] is the probability x[0] == i. If
        None, uniform initial distribution is assumed (Pi[:] == 1/K).

    Returns
    -------
    x : array (T,)
        Maximum a posteriori probability estimate of hidden state trajectory,
        conditioned on observation sequence y under the model parameters A, B,
        Pi.
    T1: array (K, T)
        the probability of the most likely path so far
    T2: array (K, T)
        the x_j-1 of the most likely path so far
    """
    # Cardinality of the state space
    K = A.shape[0]
    # Initialize the priors with default (uniform dist) if not given by caller
    Pi = Pi if Pi is not None else np.full(K, 1 / K)
    T = len(y)
    T1 = np.empty((K, T), 'd')
    T2 = np.empty((K, T), 'B')

    # Initialize the tracking tables from first observation
    T1[:, 0] = Pi * B[:, y[0]]
    T2[:, 0] = 0

    # Iterate through the observations updating the tracking tables
    for i in range(1, T):
        T1[:, i] = np.max(T1[:, i - 1] * A.T * B[np.newaxis, :, y[i]].T, 1)
        T2[:, i] = np.argmax(T1[:, i - 1] * A.T, 1)

    # Build the output, optimal model trajectory
    x = np.empty(T, 'B')
    x[-1] = np.argmax(T1[:, T - 1])
    for i in reversed(range(1, T)):
        x[i - 1] = T2[x[i], i]

    return x, T1, T2






# Observations to decode:
y = np.array([3, 1, 3])
y


viterbi(y, A, B)

'''
y : array (T,)
    Observation state sequence. int dtype.
A : array (K, K)
    State transition matrix. See HiddenMarkovModel.state_transition  for
    details.
B : array (K, M)
    Emission matrix. See HiddenMarkovModel.emission for details.
Pi: optional, (K,)
    Initial state probabilities: Pi[i] is the probability x[0] == i. If
    None, uniform initial distribution is assumed (Pi[:] == 1/K).

'''





x = np.arange(100).reshape(10, 10)

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def go_fast(a): # Function is compiled to machine code when called the first time
    trace = 0
    for i in range(a.shape[0]):   # Numba likes loops
        trace += np.tanh(a[i, i]) # Numba likes NumPy functions
    return a + trace              # Numba likes NumPy broadcasting

# DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
start = time.time()
go_fast(x)
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))

# NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
start = time.time()
go_fast(x)
end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))


