
# Project: Hidden Markov Models for pos tagging
# Author: Zack Larsen
# Date: July 20, 2019


# Outline:
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
    Create list of unique pos tags but with <START> added to the
        beginning of it as a tag
Step 4:
    Create an integer mapping between tokens and tags, starting at zero
    for both. Then, use these two dictionaries to convert the token list,
    tag list, and tuple(token, tag) list to integers. This will make it
    easier to identify the correct indices of the transition and emission
    and initial probability matrices.
Step 5:
    Create a default dictionary with {tag: count} values
    for all tags
Step 6:
    Create a default dictionary with {bigram: count} values
    for all bigrams of POS tags
Step 7: (optional)
    Create a default dictionary with {trigram: count}
    value for all trigrams of POS tags
Step 8:
    Create transition_matrix for transition probabilities from state i to
    state j
    Populate the cells of transition_matrix with the corresponding probabilities
Step 9:
    Create emission_matrix for emission probabilities
    Populate the cells of emission_matrix with the corresponding probabilities
Step 10:
    Create initial probability matrix Pi
    Populate Pi by grabbing the row from emissions corresponding to the
    start tag. Assert that Pi sums to 1 to form a valid probability distribution
Step 11:
    Run Viterbi decoding algorithm on the input matrices
'''


## Main data structures:
'''
tag_counts
bigram_counts
unique_pos
start_pos
transitions_matrix (n_unique_tags + 1, n_unique_tags)
emissions_matrix (n_unique_tokens * n_unique_tags)
viterbi_trellis (n_tags * n_tokens)
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

import numpy as np
import numba
from numba import jit
import re, pprint, sys, datetime, os
import sys
import importlib
from collections import defaultdict, Counter
import nltk
from nltk import bigrams, trigrams, word_tokenize, sent_tokenize
from nltk import corpus
from scipy.sparse import csr_matrix
#import shelve






sys.path

# Windows:
sys.path.append('C:\\Users\\U383387\\Zack_master\\ProgramEvaluations\\Team Projects\\HCC\\src')

# Mac OS X:
sys.path.append(os.path.join(homedir, 'src/'))




import hmm
from hmm import *

importlib.reload(hmm)





# POS tag file from text file:

# Windows:
homedir = os.path.dirname('C:\\Users\\U383387\\Zack_master\\ProgramEvaluations\\Team Projects\\HCC\\')
WSJ_head = os.path.join(homedir, 'data\\WSJ_head.txt')

# Mac OS X:
homedir = os.path.dirname('/Users/zacklarsen/Zack_master/Projects/Work Projects/hmm/')
WSJ_head = os.path.join(homedir, 'data/WSJ_head.txt')
WSJ_train = os.path.join(homedir, 'data/WSJ-train.txt')
WSJ_test = os.path.join(homedir, 'data/WSJ-test.txt')




# POS tag file from nltk.corpus:
#nltk.download('treebank')
#nltk.download('universal_tagset')
WSJ = corpus.treebank.tagged_words(tagset='universal')






# From text file
token_list, tag_list, token_tag_list = file_prep(WSJ_train,
                                                 nrows=4000,
                                                 lowercase=True)



# From nltk corpus
token_list = [token[0].lower() for token in WSJ]
tag_list = [tag[1] for tag in WSJ]
#token_tag_list = WSJ
token_tag_list = [tuple([token.lower(), tag]) for token, tag in WSJ]





unique_tokens = np.unique(token_list)
n_tokens = len(unique_tokens)

unique_tags = np.unique(tag_list)
n_tags = len(unique_tags)

token_map = token_to_int(unique_tokens)
token_map_reversed = reverse_token_map(token_map)
integer_token_list = [token_map[token] for token in token_list]

tag_map = token_to_int(unique_tags)
tag_map_reversed = reverse_token_map(tag_map)
integer_tag_list = [tag_map[tag] for tag in tag_list]

integer_tuple_list = [tuple([a,b]) for a,b in zip(integer_token_list, integer_tag_list)]

unique_integer_tokens = list(np.unique(integer_token_list))
unique_integer_tags = list(np.unique(integer_tag_list))

token_counts = Counter(integer_token_list)
tag_counts = Counter(integer_tag_list)
token_tag_counts = Counter(integer_tuple_list)

bigrams = find_ngrams(integer_tag_list, 2)
bigram_counts = Counter(bigrams)
n_bigrams = len(bigram_counts.keys())






transition_matrix = create_transitions(unique_integer_tags, bigram_counts,
                                       tag_counts, n_tags)

emissions_matrix = create_emissions(unique_integer_tokens, unique_integer_tags,
                                    token_tag_counts, tag_counts, n_tokens,
                                    n_tags)

Pi = create_pi('<START>', emissions_matrix, token_map)

transition_matrix.shape # 42, 42
emissions_matrix.shape # 42, 1248
Pi.shape # 42,









nouns = {key: value for key, value in token_tag_counts.items() if key[1] == tag_map['NNP']}
nouns

nouns.keys()
nouns.values()

sum(nouns.values())
tag_count[tag_map['NNP']]



for tag in unique_integer_tags:
    print(tag)






emissions_matrix[:, tag_map['<START>']].sum()
emissions_matrix[tag_map['<START>'],:].sum()
emissions_matrix[:, :].sum()















# Extras

'''

#tag_counts, tag_model = unigram(integer_tag_list)

## Step 6 (optional):
# trigrams = find_ngrams(pos_list, 3)
# trigram_counts = defaultdict(int)
# for trigram in trigrams:
#     trigram_counts[trigram] += 1




## Look at tags - are START or STOP included here?
# unique_integer_tags
# ## If not, add them:
# unique_tags.insert(0,'<START>')
# unique_tags.insert(len(unique_tags),'<STOP>')



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
