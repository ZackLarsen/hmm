##################################################
## Project: Hidden Markov Models
## Script purpose: Create HMM for POS tagging example
## Date: July 21, 2019
## Author: Zack Larsen
##################################################

library(pacman)
library(tidyverse)
library(magrittr)

p_load(HMM, TraMineR, seqHMM, data.table, here)






fread(here("data",'WSJ_head.txt'), fill = TRUE, header = FALSE)






