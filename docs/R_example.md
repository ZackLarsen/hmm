## HMM

#### Here is an example of a hidden markov model in R for part-of-speech (POS) tagging from Jurafsky & Martin's [Speech & Language Processing](https://web.stanford.edu/~jurafsky/slp3/) book:

```r

library(pacman)
library(tidyverse)
library(magrittr)

p_load(HMM, TraMineR, seqHMM, data.table, here, DataCombine,
       esquisse, ggthemr, hrbrthemes, reticulate, ggthemes)


transitions <- matrix(
  c(
    0.3777, 8e-04, 0.0322, 0.0366, 0.0096, 0.0068, 0.1147,
    0.011, 2e-04, 5e-04, 4e-04, 0.0176, 0.0102, 0.0021,
    9e-04, 0.7968, 0.005, 1e-04, 0.0014, 0.1011, 2e-04,
    0.0084, 5e-04, 0.0837, 0.0733, 0.0086, 0.1012, 0.2157,
    0.0584, 8e-04, 0.0615, 0.4509, 0.1216, 0.012, 0.4744,
    0.009, 0.1698, 0.0514, 0.0036, 0.0177, 0.0728, 0.0102,
    0.0025, 0.0041, 0.2231, 0.0036, 0.0068, 0.0479, 0.0017
  ),
  nrow=7, # number of rows
  ncol=7, # number of columns
  byrow = TRUE
) %>%
  t()

transition_labels <- c("NNP", "MD", "VB", "JJ", "NN", "RB", "DT")


emissions <- matrix(
  c(
    0.000032,0,0,0.000048,0,
    0,0.308431,0,0,0,
    0,0.000028,0.000672,0,0.000028,
    0,0,0.00034,0,0,
    0,0.0002,0.000223,0,0.002337,
    0,0,0.010446,0,0,
    0,0,0,0.506099,0
  ),
  nrow=7, # number of rows
  ncol=5, # number of columns
  byrow = TRUE
)

emissions_labels <- c("Janet", "will", "back", "the", "bill")


Pi <- c(0.2767, 6e-04, 0.0031, 0.0453, 0.0449, 0.051, 0.2026)


# This function initialises a general discrete time and discrete space
# Hidden Markov Model (HMM). A HMM consists of an alphabet of states and
# emission symbols. A HMM assumes that the states are hidden from the observer,
# while only the emissions of the states are observable. The HMM is designed
# to make inference on the states through the observation of emissions.
# The stochastics of the HMM is fully described by the initial starting
# probabilities of the states, the transition probabilities between states
# and the emission probabilities of the states.
#
# States
#   Vector with the names of the states.
# Symbols
#   Vector with the names of the symbols.
# startProbs
#   Vector with the starting probabilities of the states.
# transProbs
#   Stochastic matrix containing the transition probabilities between the states.
# emissionProbs
#   Stochastic matrix containing the emission probabilities of the states


# These are our observations:
Symbols <- emissions_labels

# These are our hidden states:
States <- transition_labels

hmm <- initHMM(States, # Hidden States
               Symbols, # Symbols, or observations
               transProbs = transitions,
               emissionProbs = emissions,
               startProbs = Pi)

hmm$States
hmm$Symbols
hmm$startProbs
hmm$transProbs
hmm$emissionProbs


# Calculate Viterbi path
viterbi_hidden_states <- viterbi(hmm, c('Janet', 'will', 'back', 'the', 'bill'))
print(viterbi_hidden_states)

# Expected output:
c('NNP', 'MD', 'VB', 'DT', 'NN')


# Evaluate viterbi-predicted hidden state sequence against actual hidden states
x <- data.frame(
  viterbi_hidden_states = viterbi_hidden_states, 
  actual_hidden_states = c('NNP', 'MD', 'VB', 'DT', 'NN')
)

# Calculate cohen's kappa statistic, which is an inter-rater agreement score between two sequences:
psych::cohen.kappa(x, w=NULL, n.obs=NULL, alpha=.05, levels=NULL) 


# Simulate from the HMM
simHMM(hmm, 4)

```
