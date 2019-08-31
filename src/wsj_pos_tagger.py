
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

#import pandas as pd
import numpy as np
import numba
from numba import jit
import re, pprint, sys, datetime, os
import importlib
from collections import defaultdict, Counter, namedtuple
#import nltk
#from nltk import bigrams, trigrams, word_tokenize, sent_tokenize
#from nltk import corpus
from scipy.sparse import csr_matrix
from sklearn.metrics import cohen_kappa_score
#import shelve
import json



# Windows:
#sys.path.append('C:\\Users\\U383387\\Zack_master\\ProgramEvaluations\\Team Projects\\HCC\\src')

# Mac OS X:
homedir = '/Users/zacklarsen/Zack_Master/Projects/Work Projects/hmm'
sys.path.append(os.path.join(homedir, 'src/'))

from hmm import *
#importlib.reload(hmm)




# POS tag file from text file:
# Windows:
#homedir = os.path.dirname('C:\\Users\\U383387\\Zack_master\\ProgramEvaluations\\Team Projects\\HCC\\')
#WSJ_head = os.path.join(homedir, 'data\\WSJ_head.txt')

# Mac OS X:
homedir = os.path.dirname('/Users/zacklarsen/Zack_master/Projects/Work Projects/hmm/')
WSJ_head = os.path.join(homedir, 'data/WSJ_head.txt')
WSJ_train = os.path.join(homedir, 'data/WSJ-train.txt')
WSJ_test = os.path.join(homedir, 'data/WSJ-test.txt')

# From text file
observation_list, state_list, observation_state_list = file_prep(WSJ_train,
                                                 nrows=150000,
                                                 lowercase=True)


observation_list
state_list
observation_state_list

observation_state_list[-10:]









observation_state_list = []
sentence_count = 1
nrows = 220663
lowercase = True


with open(WSJ_train) as infile:
    head = [next(infile) for x in range(nrows)]
    observation_state_list.append((sentence_count, '<START>', '<START>'))
    for line in head:
        line = line.strip('\n')
        chars = line.split(' ')
        if len(chars) == 3:
            observation = chars[0]
            state = chars[1]
            if lowercase:
                observation_state_list.append((sentence_count, observation.lower(), state))
            elif not lowercase:
                observation_state_list.append((sentence_count, observation, state))
        elif len(chars) != 3:
            observation_state_list.append((sentence_count, '<STOP>', '<STOP>'))
            sentence_count += 1
            observation_state_list.append((sentence_count, '<START>', '<START>'))
    observation_state_list.append((sentence_count, '<STOP>', '<STOP>'))

observation_state_list
observation_state_list[-25:]



len(set([tup[0] for tup in observation_state_list]))
# 8,937 sentences in total. Last one is junk so get rid of it:
observation_state_list = [tup for tup in observation_state_list if tup[0]!=8937]
observation_state_list
observation_state_list[-25:]
num_sequences = len(set([tup[0] for tup in observation_state_list]))


# Split into train/test (6255 is about 70% of all sentences):
train = [tup for tup in observation_state_list if tup[0]<=round(0.7*num_sequences)]
test = [tup for tup in observation_state_list if tup[0]>round(0.7*num_sequences)]


train_gen = (tup for tup in train)
next(train_gen)






# Close the vocabulary so anything with a low occurrence frequency gets the
# out-of-vocabulary state:
'<OOV>'

observations = [tup[1] for tup in train]
states = [tup[2] for tup in train]



Counter([tup[1] for tup in train])
Counter([tup[2] for tup in train])
set(Counter([tup[1] for tup in train]).values())

total_observations = sum(Counter([tup[1] for tup in train]).values())
total_observations

# Get the percent of the vocabulary that each observation represents:
obs_dict = {k: (v/total_observations) for k,v in Counter([tup[1] for tup in train]).items()}
obs_dict

# Common observations:
common_obs = {k:v for k,v in obs_dict.items() if v >= 0.0001}.keys()

# Rare observations:
rare_obs = {k:v for k,v in obs_dict.items() if v <= 0.0001}.keys()

# Replace rare observations with the out-of-vocabulary observation:
oov_train = []
for tup in train:
    if tup[1] in rare_obs:
        oov_train.append((tup[0], '<OOV>', '<OOV>'))
    else:
        oov_train.append(tup)

oov_train












prep_tuple = hmm_prep(oov_train)
#prep_tuple._asdict()
#prep_tuple._fields

n_observations = prep_tuple.n_observations
unique_integer_observations = prep_tuple.unique_integer_observations
n_states = prep_tuple.n_states
unique_integer_states = prep_tuple.unique_integer_states
observation_map = prep_tuple.observation_map
state_map = prep_tuple.state_map
state_counts = prep_tuple.state_counts
observation_state_counts = prep_tuple.observation_state_counts
bigram_counts = prep_tuple.bigram_counts
integer_tuple_list = prep_tuple.integer_tuple_list



# Save integer maps
with open('/Users/zacklarsen/Zack_master/Projects/Work Projects/hmm/data/observation_map.json', 'w') as fp:
    json.dump(observation_map, fp)

with open('/Users/zacklarsen/Zack_master/Projects/Work Projects/hmm/data/state_map.json', 'w') as fp:
    json.dump(state_map, fp)





transitions_matrix = create_transitions(unique_integer_states, bigram_counts,
                                       state_counts, n_states)

emissions_matrix = create_emissions(unique_integer_observations, unique_integer_states,
                                    observation_state_counts, state_counts, n_observations,
                                    n_states)

#observation_map['<START>'] # 4618
Pi = create_pi('<START>', emissions_matrix, observation_map)
# Assert that all start probabilities sum to 1:
#emissions_matrix[state_map['<START>'],:].sum()
#emissions_matrix[:,observation_map['<START>']].sum()

#transitions_matrix.shape
#emissions_matrix.shape
#Pi.shape








# Computing log probabilities first so we can avoid underflow:
transitions_matrix_log = np.log10(transitions_matrix)
emissions_matrix_log = np.log10(emissions_matrix)
Pi_log = np.log10(Pi)

# Save the matrices as a zipped archive of numpy arrays:
np.savez_compressed('/Users/zacklarsen/Zack_master/Projects/Work Projects/hmm/data/log_matrices',
                    transitions=transitions_matrix_log,
                    emissions=emissions_matrix_log,
                    Pi=Pi_log)











test




# Prepare the test sequences to feed to the decoder program:
test_sequences = []
for tup in test:
    if tup[1] in observation_map.keys():
        test_sequences.append((tup[0], observation_map[tup[1]], state_map[tup[2]]))
    else:
        test_sequences.append((tup[0], observation_map['<OOV>'], state_map['<OOV>']))


test_sequences
















test_sequence = namedtuple('test_sequence', 'seq_id kappa')
p1 = test_sequence('6298', 0.998)
p1
p1.seq_id
p1.kappa
p1._asdict()


















# Extras

'''



# Example WSJ sentence from test set to tag:

Mr. NNP
Carlucci NNP
, ,
59 CD
years NNS
old JJ
, ,
served VBN
as IN
defense NN
secretary NN
in IN
the DT
Reagan NNP
administration NN
. .


test_sentence = ['Mr.','Carlucci','','59','years','old','','served','as','defense','secretary','in','the','Reagan','administration','.']
test_sentence
[word.lower() for word in test_sentence]



correct_hidden_states = ['NNP','NNP',',','CD','NNS','JJ',',','VBN','IN','NN','IN','DT','NNP','NN']







# POS tag file from nltk.corpus:
#nltk.download('treebank')
#nltk.download('universal_tagset')
# WSJ = corpus.treebank.tagged_words(tagset='universal')

# From nltk corpus
# token_list = [token[0].lower() for token in WSJ]
# tag_list = [tag[1] for tag in WSJ]
# token_tag_list = [tuple([token.lower(), tag]) for token, tag in WSJ] # Or, #token_tag_list = WSJ



nouns = {key: value for key, value in token_tag_counts.items() if key[1] == tag_map['NNP']}
nouns

nouns.keys()
nouns.values()

sum(nouns.values())
tag_count[tag_map['NNP']]


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
