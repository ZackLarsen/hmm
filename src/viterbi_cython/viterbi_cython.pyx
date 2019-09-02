cpdef int test(int x):
    cdef int y = 1
    cdef int i
    for i in range(1, x+1):
        y *= i
    return y

cpdef viterbi():
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