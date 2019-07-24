
## Project: Hidden Markov Models for pos tagging
## Author: Zack Larsen
## Date: July 20, 2019
## Script purpose: create POS tagger for WSJ articles


## Outline:
'''
Step 1:
    Read in the data from either a text file or nltk corpus
Step 2:
    Create list of tokens
    Create list of POS tags
    Create list of (token, pos) tuples
Step 3:
    Create list of unique tokens
    Create list of unique pos tags
    Create list of unique pos tags but with <START> added to
        beginning of it as a tag
Step 4:
    Create a default dictionary with {tag: count} values
    for all tags
Step 5:
    Create a default dictionary with {bigram: count} values
    for all bigrams of POS tags
Step 6: (optional)
    Create a default dictionary with {trigram: count}
    value for all trigrams of POS tags
Step 7:
    Create matrix_a for transition probabilities from state i to
    state j
    Populate the cells of matrix_a with the corresponding probabilities
Step 8:
    Create matrix_b for emission probabilities
    Populate the cells of matrix_b with the corresponding probabilities
Step 9:
    Create matrix_v for Viterbi lattice
    Populate the cells of matrix_v with the corresponding probabilities
'''


## Main data structures:
'''
tag_counts
bigram_counts
unique_pos
start_pos
matrix_A (transitions) (n_unique_tags + 1, n_unique_tags)
matrix_B (emissions) (n_unique_tokens * n_unique_tags)
matrix_V (Viterbi lattice) (n_tags * n_tokens)
'''


## Main variables
'''
n_tokens
n_unique_tokens
n_tags
n_unique_tags
n_bigrams
n_unique_bigrams
'''


import re, pprint, sys, datetime, os
from collections import defaultdict, Counter
import nltk
from nltk import bigrams, trigrams, word_tokenize, sent_tokenize
from nltk import corpus
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from numba import jit
#import shelve


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
    token_map = {token: value for token, value in zip(unique_token_list, range(1,len(unique_token_list)))}
    return token_map


def viterbi_decoder(sequence):
    '''
    Take a sequence of tokens and generate associated states by
    computing the most probable path
    :param sequence: Sequence of tokens/words/observations
    :return: sequence of tags/states
    '''



def file_prep(filename, nrows = 100, lowercase = False):
    '''
    Read file, create a list of tokens, a list of parts-of-speech
    (pos), and a data dictionary of the token: tag co-occurrences
    :param filename: Name of the file being read
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


# https://github.com/ssghule/Hidden-Markov-Models-and-Viterbi-in-Natural-Language-Processing/blob/master/pos_solver.py
def hmm_viterbi(self, sentence):
    """Returns most likely POS tags of words in a sentence
       by performing Viterbi algorithm
    :param sentence: List of words (string)
    :return: List of tags
    """
    tag_list = list(tag_set)  # Converting tag_set to a list to have indexes to refer
    rows = len(tag_list)
    cols = len(sentence)
    compatibility_matrix = [[None] * cols for i in range(rows)]

    # Storing a tuple in each cell (index of the previous cell, probability of the current cell)
    for col_index, curr_word in enumerate(sentence):
        curr_emission_probs = self.get_emission_probs(curr_word)
        for row_index, curr_tag in enumerate(tag_list):
            # Computing the probabilities for the first column
            if col_index == 0:
                init_prob = self.init_prob[curr_tag] if curr_word in self.init_prob else max_val
                compatibility_matrix[row_index][col_index] = (-1, curr_emission_probs[curr_tag] + init_prob)
            # Computing the probabilities of the other columns
            else:
                best_prob_tuple = (-1, max_val)
                for prev_row_index, prev_tag in enumerate(tag_list):
                    prev_prob = compatibility_matrix[prev_row_index][col_index - 1][1]
                    curr_prob = prev_prob + curr_emission_probs[curr_tag] + self.trans_prob[prev_tag][curr_tag]
                    if curr_prob < best_prob_tuple[1]:
                        best_prob_tuple = (prev_row_index, curr_prob)
                compatibility_matrix[row_index][col_index] = best_prob_tuple

    # Backtracking to fetch the best path
    # Finding the cell with the max probability from the last column
    (max_index, max_prob) = (-1, max_val)
    for row in range(rows):
        curr_prob = compatibility_matrix[row][cols - 1][1]
        if curr_prob < max_prob:
            (max_index, max_prob) = (row, curr_prob)

    output_tag_list = list()  # List to store the output tags
    # Adding the best path to output list
    for col in range(cols - 1, 0, -1):
        output_tag_list.insert(0, tag_list[max_index])
        max_index = compatibility_matrix[max_index][col][0]
    output_tag_list.insert(0, tag_list[max_index])
    return output_tag_list



## Step 1

# POS tag file from text file:
dirname = os.path.dirname('/Users/zacklarsen/Zack_Master/Projects/Work Projects/hmm/')
WSJ_train = os.path.join(dirname, 'data/WSJ-train.txt')
WSJ_test = os.path.join(dirname, 'data/WSJ-test.txt')

# POS tag file from nltk.corpus:
# nltk.download('treebank')
# nltk.download('universal_tagset')
# WSJ = corpus.treebank.tagged_words(tagset='universal')
# WSJ[:100]
# len(WSJ) # 100,676


## Step 2

token_list, pos_list, token_pos_list = file_prep(WSJ_train, nrows=10000, lowercase=True)
token_list
pos_list
token_pos_list

## (Optional): After creating token_list, we can map tokens to integers:
token_map = token_to_int(token_list)
token_map
len(token_map)
## Reversing dictionary from above:
int_to_token = dict(map(reversed, token_map.items()))
int_to_token


## Step 3

unique_tokens = np.unique(token_list)
n_unique_tokens = len(unique_tokens)

unique_tags = np.unique(pos_list)
unique_tags = list(unique_tags)
n_unique_tags = len(unique_tags)

## Add in start tags
#start_pos = unique_pos.copy()
#start_pos.insert(0,'<START>')


## Step 4

## Calculate number of times each tag occurs:
tag_counts, tag_model = unigram(pos_list)
## Add in the count for <START> tags:
#start_count = token_list.count('<START>')
#tag_counts['<START>'] = start_count


## Step 5

## Create bigrams of tags and then count them:
bigrams = find_ngrams(pos_list, 2)
bigram_counts = defaultdict(int)
for bigram in bigrams:
    bigram_counts[bigram] += 1

n_unique_bigrams = len(bigram_counts.keys())


## Step 6 (optional):
# trigrams = find_ngrams(pos_list, 3)
# trigram_counts = defaultdict(int)
# for trigram in trigrams:
#     trigram_counts[trigram] += 1


## Step 7

## Create transition probability matrix A (dimensions (n_unique_pos * n_unique_pos):
matrix_a = np.zeros((n_unique_tags, n_unique_tags))

## Calculate probability of tag t given that it is
## preceded by tag t-1 (P(t|(t,t-1))
## Do this by dividing the bigram count by the unigram count:
## C(ti−1,ti) / C(ti−1)

## Populating cells of matrix_a:
for i, tag_i in enumerate(unique_tags):
    for j, tag_j in enumerate(unique_tags):
        bigram = tuple((tag_i, tag_j))
        bigram_count = bigram_counts[bigram]
        unigram_count = tag_counts[tag_i]
        a = bigram_count / unigram_count
        matrix_a[i,j] = a
## Verify that the start tag probabilities sum to 1:
start_index = unique_tags.index('<START>')
if matrix_a[start_index,:].sum() != 1:
    print('ERROR: <START> tag probabilities do not form a valid probability distribution')


## Step 8

## Matrix B (emission probabilities, P(wi|ti)), represents
## the probability, given a tag, that it will be associated
## with a given word. The MLE of the emission probability is
## P(wi|ti) = C(ti,wi) / C(ti).

## Calculate co-occurrences of tokens and tags (C(ti,wi)):
token_tag_counts = defaultdict(int)
for pairing in token_pos_list:
    token_tag_counts[pairing] += 1

## Construct empty matrix_b
matrix_b = np.zeros((n_unique_tokens, n_unique_tags))

## Populate cells of matrix_b:
for i, token in enumerate(unique_tokens):
    for j, tag in enumerate(unique_tags):
        token_tag_tuple = tuple((unique_tokens[i],unique_tags[j]))
        token_tag_count = token_tag_counts[token_tag_tuple]
        tag_count = tag_counts[unique_tags[j]]
        b = token_tag_count / tag_count
        matrix_b[i,j] = b


## Step 9

## Create viterbi probability trellis / lattice of dimensions
## n_unique_tags * n_unique_tokens
matrix_v = np.zeros((n_unique_tags, n_unique_tokens))

matrix_v.shape
matrix_a.shape
matrix_b.shape

matrix_b.T.shape
























## Extras
'''

## Sparse matrix implementation:
#csr_matrix((3, 4), dtype=np.int8).toarray()

sent_tokenize_list = sent_tokenize(text)

## Tokenize
tokens = []
for sentence in sent_tokenize_list:
    token_list=word_tokenize(sentence)
    tokens.append(token_list)

## Add in the <s> and <\s> to the sentence list
for sentence in tokens:
    sentence.insert(0,'<s>')
    sentence.append('</s>')


counts = {}
for sentence in tokens:
    for token in sentence:
        counts[token] = counts.get(token,0)+1
total = sum(counts.values())
print('Total number of tokens = ',total)

token_list = []
for sentence in tokens:
    for token in sentence:
        token_list.append(token)






unigrams = unigram(token_list)






bigrams = find_ngrams(token_list,2)

bigramcounts = {}
for bigram in bigrams:
    bigramcounts[bigram] = bigramcounts.get(bigram,0)+1
bigramtotal = sum(bigramcounts.values())

bigrams = find_ngrams(token_list,2)


print('Total number of unique bigrams = ',len(bigramcounts))







## Store the unigram and bigram probabilities persistently in databases
bigrams = bigram_probs
d = shelve.open('bigrams.db')
d['bigrams'] = bigrams
d.close()


d = shelve.open('unigrams.db')
d['unigrams'] = unigrams
d.close()
'''
