
import numpy as np
from numba import jit
import math
import re, sys, datetime, os
from collections import defaultdict, Counter
from scipy.sparse import csr_matrix

def unigram(tag_list):
    '''
    Construct unigram model with LaPlace smoothing
    :param tag_list: A list of pos tags
    :return: A default dictionary of pos tag counts and
    a default dictionary of pos tag counts smoothed by LaPlace smoothing
    '''
    counts_dd = defaultdict(int)
    for tag in tag_list:
        counts_dd[tag] += 1

    model = counts_dd.copy()
    for word in counts_dd:
        #model[word] = model[word]/float(len(counts_dd))
        model[word] = model[word]/float(sum(counts_dd.values()))

    return counts_dd, model


def find_ngrams(input_list, n):
    '''
    Generate ngrams
    :param input_list: List of tokens in a sequence
    :param n: Number of tokens to slide over, e.g. 1 for unigram,
    2 for bigram, 3 for trigram, etc.
    :return: A generator object with ngrams
    '''
    return zip(*[input_list[i:] for i in range(n)])


def integer_map(obs):
    '''
    Take a list of tokens and map them to integers
    :param obs: List of observations
    :return: Dictionary mapping each observation to a unique integer
    '''
    unique_obs_list = set(obs)
    unique_obs_list.sort(kind='quicksort')
    integer_map = {token: value for token, value in zip(unique_obs_list, range(0,len(unique_obs_list)))}
    return integer_map


def reverse_integer_map(integer_map):
    '''
    Reverse the order of the dictionary for the tokens
    :param integer_map: The result of the integer_map(obs) function
    :return: A reversed version of the observation map dictionary
    '''
    reversed_integer_map = {value: key for key, value in integer_map.items()}
    return reversed_integer_map


def file_prep(filename, nrows = 100, lowercase = False):
    '''
    Read file, create a list of tokens, a list of parts-of-speech
    (pos), and a data dictionary of the token: tag co-occurrences
    :param filename: Name of the file being read
    :param nrows The number of rows to read in
    :param lowercase Whether or not to lowercase all the tokens
    :return: token_list, pos_list, data
    '''
    token_list = []
    pos_list = []
    token_pos_list = []
    sentences = 0
    with open(filename) as infile:
        head = [next(infile) for x in range(nrows)]
        token_list.append('<START>')
        pos_list.append('<START>')
        token_pos_list.append(tuple(('<START>','<START>')))
        for line in head:
            line = line.strip('\n')
            chars = line.split(' ')
            if len(chars) == 3:
                token = chars[0]
                pos = chars[1]
                if lowercase:
                    token_list.append(token.lower())
                    token_pos_list.append(tuple((token.lower(), pos)))
                elif not lowercase:
                    token_list.append(token)
                    token_pos_list.append(tuple((token, pos)))
                pos_list.append(pos)
            elif len(chars) != 3:
                sentences += 1
                token_list.append('<STOP>')
                token_list.append('<START>')
                pos_list.append('<STOP>')
                pos_list.append('<START>')
                token_pos_list.append(tuple(('<STOP>', '<STOP>')))
                token_pos_list.append(tuple(('<START>', '<START>')))
        token_list.append('<STOP>')
        pos_list.append('<STOP>')
        token_pos_list.append(tuple(('<STOP>', '<STOP>')))
    print(sentences, "sentences read in.")
    return token_list, pos_list, token_pos_list


def create_transitions(tags, bigram_counts, tag_counts, n_tags, smoothing_rate = 0.01):
    """
    Create the transitions matrix representing the probability of a tag given
    the previous tag
    :param tags: list of unique tags
    :param bigram_counts: Number of times a tag co-occurs in training corpus with
    the previous tag
    :param tag_counts: Number of times a tag occurs in training corpus
    :param n_tags: Number of unique tags
    :param smoothing_rate: Number by which to inflate zero-probability tags
    :return: transition_matrix
    """
    transition_matrix = np.zeros((n_tags, n_tags))
    for i, tag_i in enumerate(tags):
        for j, tag_j in enumerate(tags):
            bigram = tuple((tag_i, tag_j))
            bigram_count = bigram_counts[bigram]
            unigram_count = tag_counts[tag_i]
            a = bigram_count / (unigram_count + smoothing_rate)
            transition_matrix[i, j] = a
    # Verify that the start tag probabilities sum to 1:
    # start_index = tags.index('<START>')
    # assert(transition_matrix[start_index,:].sum() == 1)
    return transition_matrix


def create_emissions(tokens, tags, tuple_counts, tag_counts, n_tokens, n_tags):
    """
    Create the emissions matrix representing the probability of a token given a tag.
    The MLE of the emission probability is P(token_i|tag_i) = C(tag_i,token_i) / C(tag_i).
    :param tokens: list of unique tokens
    :param tags: list of unique tags
    :param tuple_counts: tuple([token, tag]) counts
    :param tag_counts: POS tag counts
    :param n_tokens: Number of unique tokens
    :param n_tags: Number of unique tags
    :return: emissions_matrix of shape (n_tags, n_tokens)
    """
    emissions_matrix = np.zeros([n_tags, n_tokens])
    for i, tag in enumerate(tags):
        for j, token in enumerate(tokens):
            tuple_count = tuple_counts[tuple((token, tag))]
            tag_count = tag_counts[tags[i]]
            b = tuple_count / tag_count
            emissions_matrix[i, j] = b
    return emissions_matrix


def create_pi(start_tag, emissions_matrix, token_map):
    """
    Create the initial probability distribution Pi, representing the
     probability of the HMM starting in state i
    :param start_tag: Which tag in the vocabulary corresponds to starting a sequence
    :param emissions_matrix: The matrix of probabilities of tokens given a tag
    :return: Pi, the initial probability distribution
    """
    start_index = token_map[start_tag]
    Pi = emissions_matrix[:, start_index]
    assert Pi.sum() == 1, "Does not form a valid probability distribution."
    return Pi


@jit(nopython=True)
def jit_viterbi(observations, states, transitions_matrix, emissions_matrix, Pi):
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
            viterbi_trellis[current_state, time_step] = np.amax(priors[:, 1], axis=0)
            backpointer[current_state, time_step] = np.argmax(priors[:, 1], axis=0)

    # Termination
    bestpathprob = np.amax(viterbi_trellis[:, -1])
    bestpathpointer = np.argmax(viterbi_trellis[:, -1])
    viterbi_cell_idx = np.argmax(viterbi_trellis, axis=0)
    viterbi_hidden_states = []
    for i, ix in enumerate(viterbi_cell_idx):
        viterbi_hidden_states.append(backpointer[ix, i])
    viterbi_hidden_states = np.delete(viterbi_hidden_states, 0)  # We don't need the first entry - it is always zero
    viterbi_hidden_states = np.append(viterbi_hidden_states, bestpathpointer) # Add the bestpathpointer to the last entry

    return bestpathprob, viterbi_hidden_states


def viterbi(observations, transitions_matrix, emissions_matrix, Pi):
    """
    Compute the optimal sequence of hidden states
    Note that -inf (or np.NINF) for negative infinity is being used as a default
    value to represent zero probabilities because we are computing in log10 space

    :param observations:
    :param transitions_matrix:
    :param emissions_matrix:
    :param Pi:
    :return:
    """

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
            viterbi_trellis[current_state, time_step] = np.amax(priors[:, 1], axis=0)
            backpointer[current_state, time_step] = np.argmax(priors[:, 1], axis=0)

    # Termination
    bestpathprob = np.amax(viterbi_trellis[:, -1])
    bestpathpointer = np.argmax(viterbi_trellis[:, -1])
    viterbi_cell_idx = np.argmax(viterbi_trellis, axis=0)
    viterbi_hidden_states = []
    for i, ix in enumerate(viterbi_cell_idx):
        viterbi_hidden_states.append(backpointer[ix, i])
    viterbi_hidden_states = np.delete(viterbi_hidden_states, 0)  # We don't need the first entry - it is always zero
    viterbi_hidden_states = np.append(viterbi_hidden_states, bestpathpointer) # Add the bestpathpointer to the last entry

    return bestpathprob, viterbi_hidden_states
