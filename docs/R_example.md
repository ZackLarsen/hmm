## HMM R

#### Here is an example of a hidden markov model in R for part-of-speech (POS) tagging from Jurafsky & Martin's [Speech & Language Processing](https://web.stanford.edu/~jurafsky/slp3/) book:

```r

library(pacman)
library(tidyverse)
library(magrittr)

p_load(HMM, TraMineR, seqHMM, data.table, psych, progress, glue, scales)


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

#### Here is an example of POS tagging with hand-tagged data from Wall Street Journal articles:

#### Preprocessed WSJ sequence data looks like:

```r

         token     tag
 1:    <START> <START>
 2:       most     JJS
 3:    banking      NN
 4:     issues     NNS
 5:      <OOV>   <OOV>
 6:      after      IN
 7:          a      DT
 8:     sector      NN
 9:  downgrade      NN
10:         by      IN
11:      <OOV>   <OOV>
12: securities     NNP
13:          ,       ,
14:   although      IN
15:   national     NNP
16:      <OOV>   <OOV>
17:     showed     VBD
18:   strength      NN
19:         on      IN
20:   positive      JJ
21:   comments     NNS
22:       from      IN
23:  brokerage      NN
24:      firms     NNS
25:      about      IN
26:        its    PRP$
27:  long-term      JJ
28:  prospects     NNS
29:          .       .
30:      <EOS>   <EOS>

```

#### Start by reading in preprocessed CSV files:

```r

wsj_train <- fread('wsj_train.csv', col.names = c('token','tag'), header = FALSE, skip = 1)
wsj_test <- fread('wsj_test.csv', col.names = c('token','tag'), header = FALSE, skip = 1)

n_tokens <- wsj_train$token %>% n_distinct()

# Handle OOV words:
token_freq <- wsj_train %>%
  group_by(token) %>%
  tally() %>%
  mutate(freq = n/sum(n)) %>%
  arrange(-n)


# Using proportion of tokens:
vocab <- token_freq %>% head(round(0.9*n_tokens)) %>% select(token)
vocab_size <- length(vocab$token)
#vocab_size # 12,816

oov_tokens <- token_freq %>% tail(n_tokens-round(0.9*n_tokens)) %>% select(token)
#oov_tokens
#length(oov_tokens$token) # 1,424

# Replace rare observation tokens with out-of-vocabulary token:
wsj_train_closed <- wsj_train
wsj_train_closed[wsj_train_closed$token %in% oov_tokens$token] <- '<OOV>'

# Do the same for test:
wsj_test_closed <- wsj_test
wsj_test_closed[!wsj_test_closed$token %in% vocab$token] <- '<OOV>'



# First steps are to create transitions, emissions,
# and starting/initial probability matrices:


# Transitions probabilities (from one state to the next state):
transitions <- wsj_train_closed %>%
  select(tag) %>%
  mutate(next_tag = lead(tag)) %>%
  table()

transitions_probs <- transitions / rowSums(transitions)
rm(transitions) # Garbage collection - counts no longer needed


# Emission probabilities (probability of state given observation):
emissions <- wsj_train_closed %>%
  select(token, tag) %>%
  na.omit() %>%
  table()

emissions_probs <- emissions / rowSums(emissions)
rm(emissions) # Garbage collection - counts no longer needed


# Initial probabilities
Pi <- transitions_probs['<START>',]


# Do the probability matrices sum to 1 per column/row?
#rowSums(emissions_probs) # all 1's
#rowSums(transitions_probs) # all 1's
#sum(Pi) # 1


# Initialise HMM
#initHMM(States, Symbols, startProbs=NULL, transProbs=NULL, emissionProbs=NULL)

# These are our observations:
Symbols <- row.names(emissions_probs)

# These are our hidden states:
States <- colnames(transitions_probs)

# This is our parameterized hidden markov model:
hmm <- initHMM(States, # Hidden States
               Symbols, # Symbols, or observations
               transProbs = transitions_probs,
               emissionProbs = emissions_probs %>% t(),
               startProbs = Pi)

#print(hmm)
hmm$States
hmm$Symbols
hmm$startProbs
hmm$transProbs
row.names(hmm$transProbs)
hmm$emissionProbs
row.names(hmm$emissionProbs)

# Simulate from the HMM
simHMM(hmm, 10)


# Going through one sample sequence tagging procedure:

# Sequence of observations
observations <- wsj_test_closed[2:30,]$token
actual_hidden_states <- wsj_test_closed[2:30,]$tag

# Calculate Viterbi path
viterbi_hidden_states <- viterbi(hmm, observations)
print(viterbi_hidden_states)

# Evaluate viterbi-predicted hidden state sequence against actual hidden states
x <- data.frame(
  viterbi_hidden_states = viterbi_hidden_states,
  actual_hidden_states = actual_hidden_states
)

# Calculating cohen's kappa score for inter-rater agreement to assess performance of the model:
kappa <- psych::cohen.kappa(x, w=NULL, n.obs=NULL, alpha=.05, levels=NULL) 

kappa$kappa
kappa$weighted.kappa



# Going through multiple test set sequences:

# First, split test set into sequences, removing the <START> and <END>
# tokens/tags:

test_sequences <- wsj_test_closed
test_sequences$start <- ifelse(test_sequences$token == '<START>', 1, 0)
test_sequences$seq_id <- cumsum(test_sequences$start)
test_sequences %<>% 
  filter(
    token %nin% c('<START>', '<EOS>')
  ) %>% 
  select(-start)


# With the prepared test sequences, loop through them and add the kappa scores to a list:

options(show.error.messages = FALSE)
k <- max(test_sequences$seq_id)
kappas <- list(length = k)
successes <- 0
pb <- progress_bar$new(format = "[:bar] :current/:total (:percent)", total = k)
for(i in 1:k){
  pb$tick()
  try(
    {
    sequences <- test_sequences %>% filter(seq_id == i)
    actual_observations <- sequences$token
    actual_hidden_states <-  sequences$tag
    viterbi_hidden_states <- HMM::viterbi(hmm, actual_observations)
    x <- data.frame(
      viterbi_hidden_states = viterbi_hidden_states,
      actual_hidden_states = actual_hidden_states
    )
    kappa <- psych::cohen.kappa(x, w=NULL, n.obs=NULL, alpha=.05, levels=NULL)
    kappas[[i]] <- kappa$weighted.kappa
    successes <- successes + 1
    }
    ,silent = TRUE
  )
}
options(show.error.messages = TRUE)


successes

kappas
avg_kappa <- mean(unlist(kappas))

print(glue("Average kappa score for hidden markov model was: {percent(round(avg_kappa,4))}"))

```
