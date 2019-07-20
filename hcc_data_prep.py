
# Hidden markov model project with Wall Street Journal
# POS tagger data

from __future__ import division
import re, pprint, sys, datetime, os, collections
from nltk import bigrams, trigrams, word_tokenize, sent_tokenize
import shelve



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
            token = chars[0]
            POS = chars[1]
            PI = chars[2]
            data[token] = POS
            tokenlist.append(token)
            POSlist.append(POS)
        else:
            sentences += 1
            tokenlist.append('<STOP>')
            tokenlist.append('<START>')
    tokenlist.append('<STOP>')

tokenlist
POSlist
data
sentences

















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





def unigram(tokens):
    '''
    Construct unigram model with LaPlace smoothing
    :param tokens:
    :return:
    '''
    model = {}
    for f in tokens:
        try:
            model[f] += 1
        except KeyError:
            model[f] = 1
            continue
    for word in model:
        model[word] = model[word]/float(len(model))
    return model
unigrams = unigram(token_list)




def find_ngrams(input_list, n):
    '''
    Generate ngrams
    :param input_list:
    :param n:
    :return:
    '''
    return zip(*[input_list[i:] for i in range(n)])

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