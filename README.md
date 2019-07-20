# Hidden Markov Models (HMM)


#### According to Dan Jurafsky and James Martin's book Speech and Language Processing https://web.stanford.edu/~jurafsky/slp3/8.pdf, a hidden markov model (HMM) is comprised of the following 5 components:
  * Q: a set of N **states**
  * A: a **transition probability matrix**, with each element aij representing the probability of transitioning from state i to state j, subject to the constraint that the sum of these elements is 1 (forming a proper probability distribution)
  * O: a set of T **observations**, each one drawn from a vocabulary V
  * B: a set of observation likelihoods, also called **emission probabilities**, each expressing the probability of an observation ot being generated from a state i
  * pi: an **initial probability distribution** over states. pi i is the probability that the markov chain will start in state i. Some states j may have pi j = 0, meaning they cannot be initial states. As with the transition probabilities, all must sum to one to form a valid probability distribution.
