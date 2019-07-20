
# Hidden markov model project with Wall Street Journal
# POS tagger data

from __future__ import division
import re, pprint, sys, datetime, os
from collections import defaultdict, Counter
from nltk import bigrams, trigrams, word_tokenize, sent_tokenize
import shelve


def unigram(tag_list):
    '''
    Construct unigram model with LaPlace smoothing
    :param tag_list: A list of POS tags
    :return: A default dictionary of POS tag counts and
    a default dictionary of POS tag counts smoothed by LaPlace smoothing
    '''
    counts_dd = defaultdict(int)
    for tag in tag_list:
        counts_dd[tag] += 1

    model = counts_dd.copy()
    for word in counts_dd:
        model[word] = model[word]/float(len(counts_dd))

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





tokenlist = []
POSlist = []
data = {}
sentences = 0
with open('WSJ_head.txt') as infile:
    read_data = infile.readlines()
    tokenlist.append('<START>')
    for line in read_data:
        line=line.strip('\n')
        chars = line.split(' ')
        if len(chars) == 3:
            tokenlist.append(chars[0]) # Token
            POSlist.append(chars[1])  # POS
            data[chars[0]] = chars[1] # {Token : POS}
            #PI = chars[2] # Not needed now
        else:
            sentences += 1
            tokenlist.append('<STOP>')
            tokenlist.append('<START>')
    tokenlist.append('<STOP>')

tokenlist
POSlist
data
sentences






## Create transition probability matrix A:

## First, calculate number of times each tag occurs:
tag_counts, tag_model = unigram(POSlist)
tag_counts
tag_model

## Second, create bigrams of tags and then count them:
bigrams = find_ngrams(POSlist, 2)

#for bigram in bigrams:
#    print(list(bigram))

bigram_dd = defaultdict(int)

for bigram in bigrams:
    bigram_dd[bigram] += 1

bigram_dd


bigram_sample = ('VBZ', 'VBN')
type(bigram_sample)
bigram_sample[0]










## Thirdly, calculate probability of tag t given that it is
## preceded by tag t-1 (P(t|(t,t-1))
## Do this by dividing the bigram count by the unigram count:
## C(ti−1,ti) / C(ti−1)









































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