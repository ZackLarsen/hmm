

## Project: Hidden Markov Models for pos tagging
## Author: Zack Larsen
## Date: July 20, 2019


#from __future__ import division
import re, pprint, sys, datetime, os
from collections import defaultdict, Counter
#from nltk import bigrams, trigrams, word_tokenize, sent_tokenize
#import shelve
import numpy as np
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


def file_prep(filename, lowercase = False):
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
        read_data = infile.readlines()
        token_list.append('<START>')
        for line in read_data:
            line = line.strip('\n')
            chars = line.split(' ')
            if len(chars) == 3:
                if lowercase:
                    token_list.append(chars[0].lower()) # Token, lowercased
                    token_pos_list.append(tuple((chars[0].lower(),chars[1])))
                elif not lowercase:
                    token_list.append(chars[0]) # Token
                    token_pos_list.append(tuple((chars[0], chars[1])))
                pos_list.append(chars[1]) # pos
            elif len(chars) != 3:
                sentences += 1
                token_list.append('<STOP>')
                token_list.append('<START>')
                pos_list.append('<STOP>')
                pos_list.append('<START>')
        token_list.append('<STOP>')
    print(sentences, "sentences read in.")
    return token_list, pos_list, token_pos_list





token_list, pos_list, token_pos_list = file_prep('WSJ_head.txt', lowercase=True)

token_list
pos_list
token_pos_list









## Create numpy ndarray and then list of UNIQUE pos tags:
unique_pos = np.unique(pos_list)
unique_pos = list(unique_pos)

## Add in start and stop tags
## Do we need to do below? Or is it okay to have '.' as the
## STOP equivalent for the end of sentence tag? And then we
## would have no START tag.
#unique_pos.insert(0,'<START>')
#unique_pos
#unique_pos.insert(len(unique_pos),'<STOP>')

#unique_pos
#len(unique_pos) # 32




#len(token_list) # 518
#len(pos_list) # 484
## Difference in length here is due to START and STOP tokens
## not having a corresponding pos tag








## Create transition probability matrix A (dimensions N*N):


## First, calculate number of times each tag occurs:
tag_counts, tag_model = unigram(pos_list)
#tag_counts
#tag_model

#len(tag_counts)
#sum(tag_counts.values())


## Second, create bigrams of tags and then count them:
bigrams = find_ngrams(pos_list, 2)
#for bigram in bigrams:
#    print(list(bigram))

bigram_dd = defaultdict(int)
for bigram in bigrams:
    bigram_dd[bigram] += 1
#bigram_dd

#bigram_sample = ('VBZ', 'VBN')
#type(bigram_sample)
#bigram_sample[0]


## Thirdly, calculate probability of tag t given that it is
## preceded by tag t-1 (P(t|(t,t-1))
## Do this by dividing the bigram count by the unigram count:
## C(ti−1,ti) / C(ti−1)

# First, create empty matrix of correct dimensions N * N:
N = len(unique_pos)
matrix_a = np.zeros((N,N))

## Populating cells of matrix_a:
for i, tag_i in enumerate(unique_pos):
    for j, tag_j in enumerate(unique_pos):
        bigram = tuple((unique_pos[i],unique_pos[j]))
        bigram_count = bigram_dd[bigram]
        unigram_count = tag_counts[unique_pos[i]]
        a = bigram_count / unigram_count
        matrix_a[i,j] = a

matrix_a

matrix_a[12,35] # 0.75 - this value is very high!
unique_pos[12] # 'EX'
unique_pos[35] # 'VBZ'








## Matrix B (emission probabilities, P(wi|ti)), represents
## the probability, given a tag, that it will be associated
## with a given word. The MLE of the emission probability is
## P(wi|ti) = C(ti,wi) / C(ti).

## Step 1: calculate co-occurrences of tokens and tags (C(ti,wi)):

token_tag_dd = defaultdict(int)
for pairing in token_pos_list:
    token_tag_dd[pairing] += 1

token_tag_dd

len(token_list) # 2595
len(np.unique(token_list)) # 754, 712 after lowercasing


## Step 2: Constructing empty matrix_b
## Might need to use sparse implementation:
#csr_matrix((3, 4), dtype=np.int8).toarray()

unique_tokens = np.unique(token_list)
Nw = len(unique_tokens)
Nt = len(unique_pos)
matrix_b = np.zeros((Nw,Nt))
matrix_b
matrix_b.shape # (712, 40), used to be (754, 40) before lowercasing the tokens


## Step 3: Populating cells of matrix_b:

for i, token in enumerate(unique_tokens):
    for j, tag in enumerate(unique_pos):
        token_tag_tuple = tuple((unique_tokens[i],unique_pos[j]))
        token_tag_count = token_tag_dd[token_tag_tuple]
        tag_count = tag_counts[unique_pos[j]]
        b = token_tag_count / tag_count
        matrix_b[i,j] = b

matrix_b





























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
