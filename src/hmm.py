
import numpy as np
from numba import jit
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