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





# Read in data ------------------------------------------------------------

wsj <- fread(here("data",'WSJ_head.txt'), fill = TRUE, header = FALSE, col.names = c('token','tag','disregard'))
wsj










# Initial states
initial_probs







# Matrix_a, transition probabilities
transitions_matrix








# Matrix_b, emission probabilities
emissions_matrix








# Build model
sc_initmod <- build_hmm(observations = wsj, initial_probs = initial_probs,
                        transition_probs = transitions_matrix, emission_probs = emissions_matrix)

# Fit model
sc_fit <- fit_model(sc_initmod)



# biofam example ----------------------------------------------------------

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


