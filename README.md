# Hidden Markov Models (HMM)


#### According to Dan Jurafsky and James Martin's book Speech and Language Processing https://web.stanford.edu/~jurafsky/slp3/8.pdf, a hidden markov model (HMM) is comprised of the following 5 components:
  * Q: a set of N **states**.
  * A: a **transition probability matrix**, with each element aij representing the probability of transitioning from state i to state j, subject to the constraint that the sum of these elements is 1 (forming a proper probability distribution)
  * O: a set of T **observations**, each one drawn from a vocabulary V.
  * B: a set of observation likelihoods, also called **emission probabilities**, each expressing the probability of an observation ot being generated from a state i.
  * pi: an **initial probability distribution** over states. pi i is the probability that the markov chain will start in state i. Some states j may have pi j = 0, meaning they cannot be initial states. As with the transition probabilities, all must sum to one to form a valid probability distribution.

#### A  first-order  hidden  Markov  model  instantiates  two  simplifying  assumptions:
   * First, as with a first-order Markov chain, the probability of a particular state depends only on the previous state: **Markov Assumption**: P(qi|q1...qi−1) = P(qi|qi−1).
   * Second, the probability of an output observation oi depends only on the state that produced the observation qi and not on any other states or any other observations: **Output Independence**: P(oi|q1...qi,...,qT,o1,...,oi,...,oT) = P(oi|qi).

#### An HMM has two probability matrices, A and B:
   * **Matrix A** contains the tag transition probabilities P(ti|ti−1) which represent the probability of a tag occurring given the previous tag. We compute the maximum likelihood estimate of this transition probability by counting, out of the times we see the first tag in a labeled corpus, how often the first tag is followed by the second: P(ti|ti−1) = C(ti−1,ti) / C(ti−1). This matrix will have dimensions (N * N), where N is the number of tags.
   * **Matrix B** (emission probabilities, P(wi|ti)), represents the probability, given a tag, that it will be associated with a given word. The MLE of the emission probability is P(wi|ti) = C(ti,wi) / C(ti).

#### The  goal  of  HMM  decoding:
   * Given an HMM lambda = (A,B), and a sequence of observations O, find the most probable sequence of states Q:
     * <img src="https://render.githubusercontent.com/render/math?math=t_1^{n} = argmax P(t_1^{n}_ | w_1^{n})">
   * The way we would do this in the context of an HMM is to use Bayes' rule:
     * <img src="https://render.githubusercontent.com/render/math?math=t_1^{n} = argmax \frac{P(w_1^{n}_ | t_1^{n}_) P(t_1^n)}{ P(w_1^n)}">     
   * We can simplify a bit by dropping the denominator:
     * <img src="https://render.githubusercontent.com/render/math?math=t_1^{n} = argmax P(w_1^{n}_ | t_1^{n}_)P(t_1^n)">


#### HMM taggers make two further simplifying assumptions:
   * The first is that the probability of a word appearing depends only on its own tag and is independent of neighboring words and tags:
     * <img src="https://render.githubusercontent.com/render/math?math=P(w_1^{n}_|t_1^{n}) \approx \Pi_{i=1}^{n} P(w_i|t_i)">
   * The second assumption, the **bigram assumption**, is that the probability of a tag is dependent only on the previous tag, rather than the entire tag sequence:
     * <img src="https://render.githubusercontent.com/render/math?math=P(t_1^{n}) \approx \Pi_{i=1}^{n} P(t_{i} | t_{i-1})">

#### With our two simplifying assumptions, the equation for the most probable tag sequence simplifies to:
   * <img src="https://render.githubusercontent.com/render/math?math=t_1^{n} = argmax P(t_1^{n}_ | w_1^{n}) \approx argmax \Pi_{i=1}^{n} P(w_i|t_i) P(t_{i} | t_{i-1})">
     * <img src="https://render.githubusercontent.com/render/math?math=P(w_i|t_i)"> Corresponds to our **emission** probability matrix.
     * <img src="https://render.githubusercontent.com/render/math?math=P(t_{i} | t_{i-1})"> Corresponds to our **transmission** probability matrix.

