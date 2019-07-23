
## Project: Hidden Markov Models for pos tagging
## Author: Zack Larsen
## Date: July 20, 2019
## Script purpose: Create POS tagger using hidden markov model (HMM)
##      and apply it to Jason Eisner's ice cream example

import numpy as np
from numba import jit
import time


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





A = np.array([0.6,0.4,
              0.4,0.6])
A


B = np.array([0.5,0.2,
              0.4,0.4,
              0.1,0.4])
B


Pi = np.array([])


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







## Exactly as it appears in the Excel workbook example:

A = np.array([0.5,0.3,0.2,
              0.4,0.6,0.8,
              0.1,0.1,0])
A


B = np.array([0.5,0.2,
              0.4,0.4,
              0.1,0.4])
B


Pi


y = np.array([3, 1, 3])
y