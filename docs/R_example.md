## Here is an example of a hidden markov model in R for part-of-speech (POS) tagging:

```r

library(pacman)
library(tidyverse)
library(magrittr)

p_load(HMM, TraMineR, seqHMM, data.table, here, DataCombine,
       esquisse, ggthemr, hrbrthemes, reticulate, ggthemes)

wsj_train <- fread('wsj_train.csv', col.names = c('token','tag'), header = FALSE, skip = 1)
wsj_test <- fread('wsj_test.csv', col.names = c('token','tag'), header = FALSE, skip = 1)

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


```
