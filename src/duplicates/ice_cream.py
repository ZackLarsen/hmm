
## Project: Hidden Markov Models for pos tagging
## Author: Zack Larsen
## Date: July 20, 2019
## Script purpose: Create POS tagger using hidden markov model (HMM)
##      and apply it to Jason Eisner's ice cream example

import numpy as np
from numba import jit
import time


def forward(observations, states):
    """
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
    """
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
    """
    Compute the optimal sequence of hidden states
    :param observations:
    :param states:
    :return: bestpath, bestpathprob
    """

    T = len(observations)
    N = len(states)
    viterbi_trellis = np.empty([N, T])
    backpointer = np.empty([N, T])

    # Initialize
    for s, state in enumerate(states):
        viterbi_trellis[t, 0] = Pi[t] * emissions[observations[0] - 1, t]
        backpointer[t, 0] = 0

    # Recursion
    for time_step in range(1, T):
        for current_state in range(0, N):
            s_prime = np.zeros([N, N])
            for previous_state in range(0, N):
                previous_state_prob = viterbi_trellis[previous_state, time_step - 1] * \
                                      transitions[current_state, previous_state] * \
                                      emissions[observations[time_step - 1] - 1, current_state]
                s_prime[previous_state] = [previous_state, previous_state_prob]
            # print(s_prime)
            argmax_state, maxprob = np.amax(s_prime, axis=0)
            print(argmax_state, maxprob)
            viterbi_trellis[current_state, time_step] = maxprob
            backpointer[current_state, time_step] = argmax_state

    # Termination
    bestpathprob = np.amax(viterbi_trellis, axis=0)
    bestpathpointer = np.argmax(viterbi_trellis[:,N], axis=0)
    bestpath = '''the path starting at state bestpathpointer that traverses backpointer to 
        states back in time'''

    return bestpath, bestpathprob





# Transition probability matrix
transitions = np.array([[0.5,0.4,0.2],
              [0.5,0.6,0.8],
              [0.1,0.1,0]])
transitions.shape
transitions


# Emission probability matrix
emissions = np.array([[0.5,0.2],
              [0.4,0.4],
              [0.1,0.4]])
emissions.shape
emissions

emissions[2]
emissions[2,0]
emissions[0,1]


# Initial probabilities
Pi = np.array([0.2, 0.8])
Pi.shape
Pi




states = ['Cold','Hot']
observations = [3,1,3]

N = len(states)
T = len(observations)

viterbi_trellis = np.zeros([N,T])
# viterbi_trellis.shape
# viterbi_trellis

backpointer = np.zeros([N, T])
# backpointer.shape
# backpointer

for t, state in enumerate(states):
    print(t, state)
#0 Cold
#1 Hot





# Initialize
for t, state in enumerate(states):
    viterbi_trellis[t, 0] = Pi[t] * emissions[observations[0]-1, t]
    backpointer[t, 0] = 0

viterbi_trellis
backpointer


for time_step in range(1, T):
    for current_state in range(0, N):
        for previous_state in range(0, N):
            print(time_step, current_state, previous_state)
# 1 0 0
# 1 0 1
# 1 1 0
# 1 1 1
# 2 0 0
# 2 0 1
# 2 1 0
# 2 1 1

# Walking through one iteration:
time_step = 2
current_state = 1
previous_state = 1
viterbi_trellis[previous_state, time_step - 1]
transitions[current_state, previous_state]
emissions[observations[time_step-1]-1, current_state]

np.amax(s_prime, axis=0)
viterbi_trellis[1,2]

# Recursion
for time_step in range(1, T):
    for current_state in range(0, N):
        s_prime = np.zeros([N,N])
        for previous_state in range(0, N):
            previous_state_prob = viterbi_trellis[previous_state, time_step-1] * \
                                  transitions[current_state, previous_state] * \
                                  emissions[observations[time_step-1]-1, current_state]
            s_prime[previous_state] = [previous_state, previous_state_prob]
        #print(s_prime)
        argmax_state, maxprob = np.amax(s_prime, axis=0)
        #print(argmax_state, maxprob)
        viterbi_trellis[current_state, time_step] = maxprob
        backpointer[current_state, time_step] = argmax_state

# Termination
bestpathprob = np.amax(viterbi_trellis, axis=0)
bestpathpointer = np.argmax(viterbi_trellis[:,N], axis=0)

bestpathprob
bestpathpointer


viterbi_trellis
backpointer




x = np.zeros(T, 'B')
x
x[-1] = np.argmax(backpointer[:, T - 1])
x
for i in reversed(range(1, T)):
    x[i - 1] = backpointer[x[i], i]
x


































#viterbi_trellis[current_state, time_step] = np.amax(s_prime, axis=0)[1]
#backpointer[current_state, time_step] = np.amax(s_prime, axis=0)[0]


s_prime
s_prime[1,0]
s_prime[1,1]
s_prime[0]

np.amax(s_prime, axis=0)[0]
np.amax(s_prime, axis=0)[1]


previous_array = np.array([[0,0.0010000000000000002],
    [1,0.012800000000000004],
    [0,0.004000000000000001],
    [1,0.07680000000000002]])

previous_array.shape
previous_array
previous_array[0]


np.amax(previous_array, axis=0)[0]
np.amax(previous_array, axis=0)[1]





























# tuple_list = [tuple([0,.2]),tuple([1,.4]),tuple([2,.6])]
# tuple_list
#
# for tup in tuple_list:
#     print(tup[1])






# Recursion
for time_step in range(1, T+1):
    for current_state in range(0, N):
        previous_viterbi_states = np.zeros([T])
        for previous_state in range(0, N):
            previous_viterbi_states[previous_state] = viterbi[previous_state, time_step - 1] * \
                                                      transitions[current_state, previous_state] * \
                                                      emissions[current_state, time_step]
        viterbi[current_state, time_step] = np.amax(previous_viterbi_states, axis=0)
        backpointer[current_state, time_step] = np.argmax(previous_viterbi_states, axis=0)























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

