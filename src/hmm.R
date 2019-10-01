##################################################
## Project: Hidden Markov Models
## Script purpose: Create HMM for POS tagging example
## Date: July 21, 2019
## Author: Zack Larsen
##################################################

library(pacman)
library(tidyverse)
library(magrittr)

p_load(HMM, TraMineR, seqHMM, data.table, here, DataCombine,
       esquisse, ggthemr, hrbrthemes, reticulate, ggthemes,
       Hmisc, conflicted, progress, glue, scales)

conflicted::conflict_prefer("filter", "dplyr")
conflicted::conflict_prefer("lag", "dplyr")
conflict_prefer("%>%", "magrittr")
conflict_prefer("select", "dplyr")

# Reticulate --------------------------------------------------------------

library(reticulate)
use_python("/Users/zacklarsen/anaconda3/bin/python")
#use_condaenv("conda_master")
use_condaenv(condaenv = "conda_master", conda = "/Users/zacklarsen/anaconda3/bin/conda")






py$dictionary

py$dictionary$alpha





# Read in text ------------------------------------------------------------

wsj_train <- fread(here("data",'WSJ-train.txt'), 
             fill = TRUE, 
             header = FALSE, 
             col.names = c('token','tag','disregard'))

wsj_train
glimpse(wsj_train)


wsj_test <- fread(here("data",'WSJ-test.txt'), 
             fill = TRUE, 
             header = FALSE, 
             col.names = c('token','tag','disregard'))

wsj_test
glimpse(wsj_test)







wsj_train %>% 
  select(token, tag) %>% 
  head(40)




# Add <START> and <EOS> tags
wsj_train %>% 
  select(token, tag) %>% 
  slice(38)

wsj_train %>% 
  select(token, tag) %>% 
  mutate(lag = lag(token)) %>% 
  filter(lag == '.')

wsj_train %>% 
  select(token, tag) %>% 
  filter(token == '') %>% 
  head()



start_tags <- data.frame(token = "<START>", tag = "<START>", disregard = "<START>")
start_tags

end_tags <- data.frame(token = "<EOS>", tag = "<EOS>", disregard = "<EOS>")
end_tags





wsj_train <- rbind(start_tags, wsj_train)
wsj_train

wsj_train <- rbind(wsj_train, end_tags)
wsj_train


wsj_train %>% 
  select(token, tag, disregard) %>% 
  mutate(index = ifelse(token == '', 1, 0)) %>% 
  mutate(
    new_token = ifelse(index == 1, "<START>", token),
    new_tag = ifelse(index == 1, "<START>", tag),
    new_disregard = ifelse(index == 1, "<START>", disregard)
  ) %>% 
  mutate(
    new_token = ifelse(index == 1, "<START>", token),
    new_tag = ifelse(index == 1, "<START>", tag),
    new_disregard = ifelse(index == 1, "<START>", disregard)
  )






token_tag_list <- list(length=length(wsj_train$token))
token_tag_list
token_tag_list[[1]]


num_lines <- 0
for(i in 0:length(wsj_train$token)){
  if(wsj_train[i,1] == ''){
    print("I found a blank line")
    token_tag_list[[num_lines]] <- wsj_train[i,]
    token_tag_list[[num_lines]] <- wsj_train[i,]
    num_lines <- num_lines + 2
  }else{
    token_tag_list[[num_lines]] <- c("<START>","<START>")
  }
  num_lines <- num_lines + 1
}


wsj_train[39,1]








# ICD codes ---------------------------------------------------------------

icd <- fread('/Users/zacklarsen/Zack_Master/Projects/Work Projects/hmm/data/current_lcd/current_lcd_csv/lcd_x_icd10_support.csv')

# icd %>% 
#   select(icd10_code_id, icd10_code_version, description) %>% 
#   head() %>% 
#   View("ICD")

icd %<>% 
  select(icd10_code_id, description) %>% 
  separate(icd10_code_id, c("first", "second"), remove = FALSE)




install.packages("collapsibleTree")
#https://adeelk93.github.io/collapsibleTree/








install.packages("visNetwork")
install.packages("rpart")
#https://datastorm-open.github.io/visNetwork/tree.html












#https://cran.r-project.org/web/packages/data.tree/vignettes/data.tree.html

#Let’s convert that into a data.tree structure! We start by defining a pathString. The pathString describes the hierarchy by defining a path from the root to each leaf. In this example, the hierarchy comes very naturally:
icd$pathString <- paste("world", 
                        icd$first, 
                        icd$second, 
                        sep = "/")

icd


# Remove any missing values:
icd %<>% 
  na.omit()

icd






# Make a data.tree out of icd:
population <- as.Node(icd)
population


print(population, "first", "second", "description", limit = 20)

plot(population)






library(data.tree)

SetGraphStyle(acme, rankdir = "TB")
SetEdgeStyle(acme, arrowhead = "vee", color = "grey35", penwidth = 2)
SetNodeStyle(acme, style = "filled,rounded", shape = "box", fillcolor = "GreenYellow", 
             fontname = "helvetica", tooltip = GetDefaultTooltip)
SetNodeStyle(acme$IT, fillcolor = "LightBlue", penwidth = "5px")
plot(acme)








library(networkD3)

#convert to Node
useRtree <- as.Node(useRdf, pathDelimiter = "|")

#plot with networkD3
useRtreeList <- ToListExplicit(useRtree, unname = TRUE)
radialNetwork(useRtreeList)
















# Using XML from website ftp archive:

# install.packages("xml2")
# library(xml2)
# x <- read_xml(
#   'ftp://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/ICD10CM/2019/icd10cm_tabular_2019.xml'
# )
# x
# xml_name(x)
# xml_children(x)
# 
# # Find all name nodes anywhere in the document
# name <- xml_find_all(x, ".//name")
# name
# 
# # Find all diag nodes anywhere in the document
# diag <- xml_find_all(x, ".//diag")
# diag




# HCPCS CPT codes ---------------------------------------------------------

#https://www.cms.gov/Medicare/Coding/HCPCSReleaseCodeSets/Alpha-Numeric-HCPCS-Items/2019-Alpha-Numeric-HCPCS-File.html?DLPage=1&DLEntries=10&DLSort=0&DLSortDir=descending





# Synthetic Data ----------------------------------------------------------













# Synthpop ----------------------------------------------------------------

#http://gradientdescending.com/generating-synthetic-data-sets-with-synthpop-in-r/
#http://gradientdescending.com/synthesising-multiple-linked-data-sets-in-r/

suppressPackageStartupMessages(library(synthpop))
suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(sampling))
suppressPackageStartupMessages(library(partykit))

mycols <- c("darkmagenta", "turquoise")
options(xtable.floating = FALSE)
options(xtable.timestamp = "")
myseed <- 20190110


# filtering the dataset
original.df <- SD2011 %>% dplyr::select(sex, age, socprof, income, marital, depress, sport, nofriend, smoke, nociga, alcabuse, bmi)
head(original.df)


# setting continuous variable NA list
cont.na.list <- list(income = c(NA, -8), nofriend = c(NA, -8), nociga = c(NA, -8))

# apply rules to ensure consistency
rules.list <- list(
  marital = "age < 18", 
  nociga = "smoke == 'NO'")

rules.value.list <- list(
  marital = "SINGLE", 
  nociga = -8)


# This person has a BMI over 450!
SD2011[which.max(SD2011$bmi),]
# getting around the error: synthesis needs to occur before the rules are applied
original.df$bmi <- ifelse(original.df$bmi > 75, NA, original.df$bmi)


# synthesise data
synth.obj <- syn(original.df, cont.na = cont.na.list, rules = rules.list, rvalues = rules.value.list, seed = myseed)
synth.obj


# compare the synthetic and original data frames
compare(synth.obj, original.df, nrow = 3, ncol = 4, cols = mycols)$plot


# checking rules worked
table(synth.obj$syn[,c("smoke", "nociga")])
# They did. All non-smokers have missing values for 
# the number of cigarettes consumed.



# ------------ MODEL COMPARISON
glm1 <- glm.synds(ifelse(depress > 7, 1, 0) ~ sex + age + log(income) + sport + nofriend + smoke + alcabuse + bmi, data = synth.obj, family = "binomial")

summary(glm1)

# compare to the original data set
compare(glm1, original.df, lcol = mycols)


# Read in csv --------------------------------------------------------------

wsj_train <- fread('wsj_train.csv', col.names = c('token','tag'), header = FALSE, skip = 1)
wsj_test <- fread('wsj_test.csv', col.names = c('token','tag'), header = FALSE, skip = 1)

# WSJ ---------------------------------------------------------------------

wsj_train
wsj_test



# Look at probability of all words and determine a cutoff. Treat
# anything below that point as OOV:

# Vocab size:
V <- wsj_train$token %>% n_distinct()


word_freq <- wsj_train %>% 
  group_by(token) %>% 
  tally() %>%
  mutate(freq = n / sum(n)) %>% 
  arrange(-n)

word_freq


ggplot(word_freq, aes(x=n)) +
  geom_density(color="darkblue", fill="lightblue") + 
  theme_gdocs() + 
  ggtitle("Word frequency")

ggplot(word_freq, aes(x=freq)) +
  geom_density(color="darkblue", fill="lightblue") + 
  theme_gdocs() + 
  ggtitle("Word frequency density")




vocab <- wsj_train %>% 
  group_by(token) %>% 
  tally() %>%
  mutate(freq = n / sum(n)) %>% 
  filter(freq > 0.00001) %>% 
  select(token)

vocab$token


OOV_tokens <- wsj_train %>% 
  group_by(token) %>% 
  tally() %>%
  mutate(freq = n / sum(n)) %>% 
  filter(freq <= 0.00001) %>% 
  select(token)

OOV_tokens$token


# Replace any OOV tokens with '<OOV>' string in train data frame:
wsj_train_closed <- wsj_train
wsj_train_closed$token %in% OOV_tokens$token

wsj_train_closed[wsj_train_closed$token %in% OOV_tokens$token, "token"] <- '<OOV>'
wsj_train_closed[wsj_train_closed$token == '<OOV>', "tag"] <- '<OOV>'

wsj_train_closed





# Replace OOV tokens in the test data frame:
wsj_test_closed <- wsj_test
wsj_test_closed[wsj_test_closed$token %in% OOV_tokens$token, "token"] <- '<OOV>'
wsj_test_closed[wsj_test_closed$token == '<OOV>', "tag"] <- '<OOV>'

wsj_test_closed %>% 
  head(50)













# First steps are to create transitions, emissions, 
# and starting/initial probability matrices:




# Transitions:
transitions <- wsj_train_closed %>% 
        select(tag) %>% 
        mutate(next_tag = lead(tag)) %>% 
        table()

transitions

transitions_probs <- transitions / rowSums(transitions)
transitions_probs

rm(transitions)







# Emission probabilities
emissions <- wsj_train_closed %>%
        select(token, tag) %>%
        na.omit() %>%
        table()

emissions

emissions_probs <- emissions / rowSums(emissions)
emissions_probs
rm(emissions)








# Initial probabilities
sum(transitions_probs['<START>',])
Pi <- transitions_probs['<START>',]
Pi
sum(Pi)
#View(Pi)



# Do the probability matrices sum to 1 per column/row?
rowSums(emissions_probs) # all 1's
colSums(emissions_probs) %>% sum()

rowSums(transitions_probs) # all 1's
colSums(transitions_probs) %>% sum()

sum(Pi)











# Save matrix column and row names before using sapply for log10 function:
row.names(emissions_probs)
colnames(emissions_probs)

row.names(transitions_probs)
colnames(transitions_probs)

State_names <- colnames(transitions_probs)
Symbol_names <- row.names(emissions_probs)






# Convert all probabilitiy matrices to log10 to avoid numerical underflow:
transitions_probs_log10 <- transitions_probs %>% 
  sapply(function(x) log10(x))

emissions_probs_log10 <- emissions_probs %>% 
  sapply(function(x) log10(x))

Pi_log10 <- Pi %>% 
  sapply(function(x) log10(x))



# Using purrr instead of sapply:
transitions_probs_log10 <- modify(transitions_probs, log10)
emissions_probs_log10 <- modify(emissions_probs, log10)
Pi_log10 <- modify(Pi, log10)





# Initialise HMM
#initHMM(States, Symbols, startProbs=NULL, transProbs=NULL, emissionProbs=NULL)

hmm <- initHMM(State_names,
               Symbol_names, # Symbols
               transProbs = transitions_probs,
               emissionProbs = emissions_probs,
               startProbs = Pi)

# hmm <- initHMM(State_names,
#                Symbol_names, # Symbols
#                transProbs = transitions_probs_log10,
#                emissionProbs = emissions_probs_log10,
#                startProbs = Pi_log10)

hmm$States
hmm$Symbols
hmm$startProbs
hmm$transProbs
hmm$emissionProbs



transitions_probs['<START>', 'NN']
emissions_probs['confidence', 'NN']
Pi['NN']


# Sequence of observations
observations <- c('confidence','in','the','pound')
observations

# Verify words are in vocab:
for(i in observations){
  if(!i %in% vocab$token){
    print("Out of vocabulary")
  }
}

'confidence' %in% vocab$token
'in' %in% vocab$token
'the' %in% vocab$token
'pound' %in% vocab$token






# Calculate Viterbi path
viterbi <- viterbi(hmm, observations)
print(viterbi)


# Simulate from the HMM
simHMM(hmm, 100)






# WSJ from work ---------------------------------------------------------------

n_tokens <- wsj_train$token %>% n_distinct()

# Handle OOV words:
token_freq <- wsj_train %>%
  group_by(token) %>%
  tally() %>%
  mutate(freq = n/sum(n)) %>%
  arrange(-n)

token_freq



# Using proportion of tokens:
vocab <- token_freq %>% head(round(0.9*n_tokens)) %>% select(token)
vocab_size <- length(vocab$token)
#vocab_size # 12,816


oov_tokens <- token_freq %>% tail(n_tokens-round(0.9*n_tokens)) %>% select(token)
#oov_tokens
#length(oov_tokens$token) # 1,424



# Using frequency threshold:
# freq_threshold <- 0.0001

# token_freq %>%
#   filter(freq > freq_threshold) %>%
#   tally()

# token_freq %>%
#   filter(freq <= freq_threshold) %>%
#   tally()

# vocab <- wsj_train %>%
#   group_by(token) %>%
#   tally() %>%
#   mutate(freq = n/sum(n)) %>%
#   filter(freq <= freq_threshold) %>%
#   select(token)

# vocab$token

# oov_words <- wsj_train %>%
#   group_by(token) %>%
#   tally() %>%
#   mutate(freq = n/sum(n)) %>%
#   filter(freq <= freq_threshold) %>%
#   select(token)

# length(oov_words$token)
# oov_words$token




wsj_train_closed <- wsj_train
wsj_train_closed[wsj_train_closed$token %in% oov_tokens$token] <- '<OOV>'

wsj_train_closed %>% 
  filter(token == '<OOV>')





# Do the same for test:
#wsj_test[!wsj_test$token %in% vocab$token]
#wsj_test[wsj_test$token %in% vocab$token]

wsj_test_closed <- wsj_test
wsj_test_closed[!wsj_test_closed$token %in% vocab$token] <- '<OOV>'

wsj_test_closed %>% 
  head(50)









# First steps are to create transitions, emissions,
# and starting/initial probability matrices:






# Transitions probabilities (from one state to the next state):
transitions <- wsj_train_closed %>%
  select(tag) %>%
  mutate(next_tag = lead(tag)) %>%
  table()

row.names(transitions)
colnames(transitions)
row.names(transitions) == colnames(transitions)

transitions

transitions_probs <- transitions / rowSums(transitions)
#transitions_probs %>% View()

rm(transitions)


transitions_probs %>%
  as_tibble() %>%
  pivot_wider(names_from = next_tag, values_from = n) %>%
  column_to_rownames("tag") %>%
  View()

dim(transitions_probs)

transitions_probs %>% sum()
transitions_probs %>% colSums()
transitions_probs %>% rowSums()







# Emission probabilities (probability of state given observation):
emissions <- wsj_train_closed %>%
  select(token, tag) %>%
  na.omit() %>%
  table()

emissions

emissions_probs <- emissions / rowSums(emissions)
emissions_probs

rm(emissions)

dim(emissions_probs)
row.names(emissions_probs)




#emissions_probs['<START>',]
#transitions_probs['<START>',]
#sum(transitions_probs['<START>',])






# Initial probabilities
Pi <- transitions_probs['<START>',]
Pi

sum(Pi)
length(Pi)




# Do the probability matrices sum to 1 per column/row?
#rowSums(emissions_probs) # all 1's
#rowSums(emissions_probs)
#colSums(emissions_probs) %>% sum()


#rowSums(transitions_probs) # all 1's
#colSums(transitions_probs)
#colSums(transitions_probs) %>% sum()

#sum(Pi) # 1




# Initialise HMM

#initHMM(States, Symbols, startProbs=NULL, transProbs=NULL, emissionProbs=NULL)

#row.names(emissions_probs)
#colnames(emissions_probs)

#row.names(transitions_probs)
#colnames(transitions_probs)






# This function initializes a general discrete time and discrete space
# Hidden Markov Model (HMM). A HMM consists of an alphabet of states and
# emission symbols. A HMM assumes that the states are hidden from the observer,
# while only the emissions of the states are observable. The HMM is designed
# to make inference on the states through the observation of emissions.
# The stochastics of the HMM is fully described by the initial starting
# probabilities of the states, the transition probabilities between states
# and the emission probabilities of the states.

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
Symbols <- row.names(emissions_probs)
Symbols


# These are our hidden states:
States <- colnames(transitions_probs)
States


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
simHMM(hmm, 100)




# Going through one sample sequence tagging procedure:

# Sequence of observations
#observations <- c("<START>","confidence","in","the","<OOV>")
observations <- wsj_test_closed[2:30,]$token
observations

actual_hidden_states <- wsj_test_closed[2:30,]$tag
actual_hidden_states

# Calculate Viterbi path
viterbi_hidden_states <- viterbi(hmm, observations)
print(viterbi_hidden_states)

# Evaluate viterbi-predicted hidden state sequence against actual hidden states
x <- data.frame(
  viterbi_hidden_states = viterbi_hidden_states,
  actual_hidden_states = actual_hidden_states
)

kappa <- psych::cohen.kappa(x, w=NULL, n.obs=NULL, alpha=.05, levels=NULL) 

kappa$kappa

kappa$weighted.kappa









# Going through all test set sequences:

# First, split test set into sequences, removing the <START> and <END>
# tokens/tags:

wsj_test_closed

test_sequences <- wsj_test_closed
test_sequences$start <- ifelse(test_sequences$token == '<START>', 1, 0)
test_sequences$seq_id <- cumsum(test_sequences$start)
test_sequences %<>% 
  filter(
    token %nin% c('<START>', '<EOS>')
  ) %>% 
  select(-start)

# test_sequences %>%
#   filter(seq_id <= 10) %>% 
#   View()


max(test_sequences$seq_id)

test_sequences %>% filter(seq_id == max(test_sequences$seq_id))



test_sequences %>% filter(seq_id == 86)








options(show.error.messages = FALSE)
k <- 100
#k <- max(test_sequences$seq_id)
viterbi_hidden_states_list <- list(length = k)
kappas <- list(length = k)
successes <- 0
#pb <- progress_bar$new(total = k)
pb <- progress_bar$new(format = "[:bar] :current/:total (:percent)", total = k)
for(i in 1:k){
  pb$tick()
  try(
    {
    sequences <- test_sequences %>% filter(seq_id == i)
    actual_observations <- sequences$token
    actual_hidden_states <-  sequences$tag
    viterbi_hidden_states <- HMM::viterbi(hmm, actual_observations)
    viterbi_hidden_states_list[[i]] <- viterbi_hidden_states
    x <- data.frame(
      viterbi_hidden_states = viterbi_hidden_states,
      actual_hidden_states = actual_hidden_states
    )
    kappa <- psych::cohen.kappa(x, w=NULL, n.obs=NULL, alpha=.05, levels=NULL)
    kappas[[i]] <- kappa$weighted.kappa
    #print(kappa$weighted.kappa)
    successes <- successes + 1
    }
    ,silent = TRUE
  )
}
options(show.error.messages = TRUE)


successes

# Compare actual hidden states to the model's predictions:
viterbi_hidden_states_list[[1]]
test_sequences %>% 
  filter(seq_id == 1) %>% 
  select(tag)



kappas
mean(unlist(kappas))

avg_kappa <- mean(unlist(kappas))
avg_kappa

print(glue("Average kappa score for hidden markov model was: {percent(round(avg_kappa,4))}"))






# Janet example -----------------------------------------------------------

# Example from Jurafsky & Martin Speech & Language Processing book:

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

transitions

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

emissions

emissions_labels <- c("Janet", "will", "back", "the", "bill")



Pi <- c(0.2767, 6e-04, 0.0031, 0.0453, 0.0449, 0.051, 0.2026)
Pi



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
#Symbols <- row.names(emissions)
Symbols <- emissions_labels
Symbols


# These are our hidden states:
#States <- colnames(transitions)
States <- transition_labels
States


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
viterbi <- viterbi(hmm, c('Janet', 'will', 'back', 'the', 'bill'))
print(viterbi)

# Expected output:
c('NNP', 'MD', 'VB', 'DT', 'NN')



# Evaluate viterbi-predicted hidden state sequence against actual hidden states
x <- data.frame(
  viterbi_sequence = viterbi, 
  actual_sequence = c('NNP', 'MD', 'VB', 'DT', 'NN')
)

psych::cohen.kappa(x, w=NULL, n.obs=NULL, alpha=.05, levels=NULL) 


# Simulate from the HMM
simHMM(hmm, 4)






# HMM example -------------------------------------------------------------


# Initialise HMM
hmm = initHMM(c("A","B"), # States
              c("L","R"), # Symbols
              transProbs=matrix(c(.8,.2,.2,.8),2),
              emissionProbs=matrix(c(.6,.4,.4,.6),2),
              startProbs=matrix(c(0.9,0.1)))

print(hmm)

# Sequence of observations
observations = c("L","L","R","R")

# Calculate Viterbi path
viterbi = viterbi(hmm,observations)
print(viterbi)



# Calculate backward probablities
logBackwardProbabilities = backward(hmm,observations)
print(exp(logBackwardProbabilities))

viterbiTraining(hmm, observations, maxIterations=100, delta=1E-9, pseudoCount=0)


# biofam example ----------------------------------------------------------


# Build model
sc_initmod <- build_hmm(observations = wsj, initial_probs = initial_probs,
                        transition_probs = transitions_matrix, emission_probs = emissions_matrix)

# Fit model
sc_fit <- fit_model(sc_initmod)






#https://arxiv.org/pdf/1704.00543.pdf

data("biofam", package = "TraMineR")
biofam_seq <- seqdef(biofam[, 10:25], start = 15, 
                     labels = c("parent","left", "married", "left+marr",
                                "child", "left+child", "left+marr+ch",
                                "divorced"))

data("biofam3c")
marr_seq <- seqdef(biofam3c$married, start = 15, 
                   alphabet = c("single","married", "divorced"))
child_seq <- seqdef(biofam3c$children, start = 15,
                    alphabet = c("childless", "children"))
left_seq <- seqdef(biofam3c$left, start = 15, 
                   alphabet = c("with parents","left home"))


sc_init <- c(0.9, 0.06, 0.02, 0.01, 0.01)
sc_trans <- matrix(c(0.80, 0.10, 0.05, 0.03, 0.02, 
                     0.02, 0.80, 0.10, 0.05, 0.03,
                     0.02, 0.03, 0.80, 0.10, 0.05, 
                     0.02, 0.03, 0.05, 0.80, 0.10,
                     0.02, 0.03, 0.05, 0.05, 0.85), 
                   nrow = 5, ncol = 5, byrow = TRUE)

sc_emiss <- matrix(NA, nrow = 5, ncol = 8)
sc_emiss[1,] <- seqstatf(biofam_seq[, 1:4])[, 2] + 0.1
sc_emiss[2,] <- seqstatf(biofam_seq[, 5:7])[, 2] + 0.1
sc_emiss[3,] <- seqstatf(biofam_seq[, 8:10])[, 2] + 0.1
sc_emiss[4,] <- seqstatf(biofam_seq[, 11:13])[, 2] + 0.1
sc_emiss[5,] <- seqstatf(biofam_seq[, 14:16])[, 2] + 0.1
sc_emiss <- sc_emiss / rowSums(sc_emiss)
rownames(sc_trans) <- colnames(sc_trans) <- rownames(sc_emiss) <- paste("State", 1:5)
colnames(sc_emiss) <- attr(biofam_seq, "labels")

sc_trans
round(sc_emiss, 3)


sc_initmod <- build_hmm(observations = biofam_seq, initial_probs = sc_init,
                        transition_probs = sc_trans, emission_probs = sc_emiss)

# Estimate parameters
sc_fit <- fit_model(sc_initmod)
sc_fit$logLik
sc_fit$model




mc_init <- c(0.9, 0.05, 0.02, 0.02, 0.01)
mc_trans <- matrix(c(0.80, 0.10, 0.05, 0.03, 0.02, 
                     0, 0.90, 0.05, 0.03, 0.02, 
                     0, 0, 0.90, 0.07, 0.03,
                     0, 0, 0, 0.90, 0.10,
                     0, 0, 0, 0, 1),
                   nrow = 5, ncol = 5, byrow = TRUE)

mc_emiss_marr <- matrix(c(0.90, 0.05, 0.05, 0.90, 0.05,
                          0.05, 0.05, 0.90, 0.05, 0.05,
                          0.90, 0.05, 0.30, 0.30, 0.40),
                        nrow = 5, ncol = 3,byrow = TRUE)
mc_emiss_child <- matrix(c(0.9, 0.1, 0.9, 0.1, 0.1,
                           0.9, 0.1, 0.9, 0.5,0.5),
                         nrow = 5, ncol = 2, byrow = TRUE)
mc_emiss_left <- matrix(c(0.9, 0.1, 0.1, 0.9, 0.1,
                          0.9, 0.1, 0.9, 0.5,0.5),
                        nrow = 5, ncol = 2, byrow = TRUE)
mc_obs <- list(marr_seq, child_seq, left_seq)
mc_emiss <- list(mc_emiss_marr, mc_emiss_child, mc_emiss_left)

mc_initmod <- build_hmm(observations = mc_obs, initial_probs = mc_init,
                        transition_probs = mc_trans, emission_probs = mc_emiss,
                        channel_names = c("Marriage", "Parenthood", "Residence"))

mc_initmod



mc_fit <- fit_model(mc_initmod, em_step = FALSE, local_step = TRUE,
                    threads = 4)

hmm_biofam <- mc_fit$model
BIC(hmm_biofam)


plot(hmm_biofam)

plot(hmm_biofam, vertex.size = 50, vertex.label.dist = 1.5,
     edge.curved = c(0, 0.6, -0.8, 0.6, 0, 0.6, 0), legend.prop = 0.3,
     combined.slice.label = "States with prob. < 0.05")


vertex_layout <- matrix(c(1, 2, 2, 3, 1, 0, 0.5, -0.5, 0, -1), ncol = 2)
plot(hmm_biofam, layout = vertex_layout, xlim = c(0.5, 3.5),
     ylim = c(-1.5, 1), rescale = FALSE, vertex.size = 50, 
     vertex.label.pos = c("left", "top", "bottom", "right", "left"),
     edge.curved = FALSE, edge.width = 1, edge.arrow.size = 1,
     with.legend = "left", legend.prop = 0.4, label.signif = 1,
     combine.slices = 0, cpal = colorpalette[[30]][c(14:5)])


ssplot(hmm_biofam, plots = "both", type = "I", sortv = "mds.hidden",
       title = "Observed and hidden state sequences", xtlab = 15:30,
       xlab = "Age")


