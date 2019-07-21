
## Project: Hidden Markov Models for pos tagging
## Author: Zack Larsen
## Date: July 20, 2019


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
Step 8:
    Create matrix_b for emission probabilities
Step 9:
    Create matrix_v for Viterbi lattice
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
from scipy.sparse import csr_matrix
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


## Step 3

unique_tokens = np.unique(token_list)
n_unique_tokens = len(unique_tokens)

unique_pos = np.unique(pos_list)
unique_pos = list(unique_pos)
n_unique_tags = len(unique_pos)

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

## Create transition probability matrix A (dimensions (N + 1) * N):
N1 = len(start_pos)
N2 = len(unique_pos)
matrix_a = np.zeros((N1,N2))

## Calculate probability of tag t given that it is
## preceded by tag t-1 (P(t|(t,t-1))
## Do this by dividing the bigram count by the unigram count:
## C(ti−1,ti) / C(ti−1)

## Populating cells of matrix_a:
for i, tag_i in enumerate(start_pos):
    for j, tag_j in enumerate(unique_pos):
        bigram = tuple((tag_i, tag_j))
        bigram_count = bigram_counts[bigram]
        unigram_count = tag_counts[tag_i]
        a = bigram_count / unigram_count
        matrix_a[i,j] = a
## Verify that the start tag probabilities sum to 1:
if matrix_a[0,:].sum() != 1:
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
    for j, tag in enumerate(unique_pos):
        token_tag_tuple = tuple((unique_tokens[i],unique_pos[j]))
        token_tag_count = token_tag_dd[token_tag_tuple]
        tag_count = tag_counts[unique_pos[j]]
        b = token_tag_count / tag_count
        matrix_b[i,j] = b


## Step 9

## Create viterbi probability trellis / lattice of dimensions
## n_unique_tags * n_unique_tokens
matrix_v = np.zeros((n_unique_tags, n_unique_tokens))




















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