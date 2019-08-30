
test_sequences = ...

transitions_matrix_log
emissions_matrix_log
Pi_log


if __name__ == '__main__':
    viterbi(sequence, transitions_matrix_log, emissions_matrix_log, Pi_log)