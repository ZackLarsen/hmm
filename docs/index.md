## Hello!

This is a site about work I have done related to Hidden Markov Models (HMM) and part-of-speech tagging as well as some applications in healthcare, specifically electronic health record (EHR) data and high-cost claimant analysis.

[Background](https://zacklarsen.github.io/hmm/Background.html)
This covers some of the statistical background information for part-of-speech tagging applications of hidden markov models, as outlined in the excellent (and free!) [Speech & Language Processing](https://web.stanford.edu/~jurafsky/slp3/) book by Jurafsky and Martin.

[Model workflow](https://zacklarsen.github.io/hmm/Model_workflow.html)
This page covers from a high level the steps that need to be taken to go from data to working HMM.

[R Examples](https://zacklarsen.github.io/hmm/R_example.html)
Making heavy use of the HMM package, I will show two examples of HMM's as POS taggers, and possibly later I will add examples from other data like medical insurance claims (synthetic, of course!).

[Python Examples](https://zacklarsen.github.io/hmm/Python_example.html)
Pomegranate is a Python package that offers an easy interface to build HMM's, and I may include an example on that, but my main focus for this page will be creating a HMM from scratch in python using numpy ndarrays, including the use of log10 transformations to model longer sequences without running into numerical underflow problems.

### HMM Applications

#### Natural language processing:
1. Part-of-speech (POS) tagging
1. Speech recognition
1. [Sign language recognition](https://machinelearnings.co/sign-language-recognition-with-hmms-504b86a2acde)

#### Healthcare:
1. Disease progression / phenotyping
1. Behavior modeling
  * [Alcoholism](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2757318/)
  * [Readmissions](https://pdfs.semanticscholar.org/77a5/bb8d5d0dd68ed03499f29be3acde7b2eea9c.pdf)

#### Scientific Research
1. [Biological sequence analysis](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2766791/)
1. [Parkinson's Disease Detection](https://www.researchgate.net/profile/Abed_Khorasani/publication/267728153_HMM_for_Classification_of_Parkinson's_Disease_Based_on_the_Raw_Gait_Data/links/5598a41508ae21086d2371fd/HMM-for-Classification-of-Parkinsons-Disease-Based-on-the-Raw-Gait-Data.pdf) Using Gait Analysis

#### Finance
1. Time series prediction
