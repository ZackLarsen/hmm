
# Train hidden markov model on sample data from SLP textbook Janet example

import numpy as np
from pomegranate import *





# Transition probability matrix
transitions = np.array([[0.2767,0.0006,0.0031,0.0453,0.0449,0.051,0.2026],
    [0.3777,0.011,0.0009,0.0084,0.0584,0.009,0.0025],
    [0.0008,0.0002,0.7968,0.0005,0.0008,0.1698,0.0041],
    [0.0322,0.0005,0.005,0.0837,0.0615,0.0514,0.2231],
    [0.0366,0.0004,0.0001,0.0733,0.4509,0.0036,0.0036],
    [0.0096,0.0176,0.0014,0.0086,0.1216,0.0177,0.0068],
    [0.0068,0.0102,0.1011,0.1012,0.012,0.0728,0.0479],
    [0.1147,0.0021,0.0002,0.2157,0.4744,0.0102,0.0017]])

transitions.shape
transitions

# Emission probability matrix
emissions = np.array([[0.000032,0,0,0.000048,0],
    [0,0.308431,0,0,0],
    [0,0.000028,0.000672,0,0.000028],
    [0,0,0.00034,0,0],
    [0,0.0002,0.000223,0,0.002337],
    [0,0,0.010446,0,0],
    [0,0,0,0.506099,0]
    ])

emissions.shape
emissions

# Initial probabilities
Pi = np.array([0.2767,
    0.0006,
    0.0031,
    0.0453,
    0.0449,
    0.051,
    0.2026
    ])

Pi.shape
Pi











dists = [NormalDistribution(5, 1), NormalDistribution(1, 7), NormalDistribution(8,2)]
trans_mat = transitions
starts = Pi
model = HiddenMarkovModel.from_matrix(trans_mat, starts)


model





















d1 = DiscreteDistribution({'A' : 0.35, 'C' : 0.20, 'G' : 0.05, 'T' : 0.40})
d2 = DiscreteDistribution({'A' : 0.25, 'C' : 0.25, 'G' : 0.25, 'T' : 0.25})
d3 = DiscreteDistribution({'A' : 0.10, 'C' : 0.40, 'G' : 0.40, 'T' : 0.10})



s1 = State(d1, name="s1")
s2 = State(d2, name="s2")
s3 = State(d3, name="s3")




model = HiddenMarkovModel('example')
model.add_states([s1, s2, s3])






# Initial probabilities
Pi = np.array([0.2767,
    0.0006,
    0.0031,
    0.0453,
    0.0449,
    0.051,
    0.2026
    ])
model.add_transition(model.start, s1, 0.2767)
model.add_transition(model.start, s2, 0.0006)
model.add_transition(model.start, s3, 0.0031)
model.add_transition(model.start, s4, 0.0453)
model.add_transition(model.start, s5, 0.0449)
model.add_transition(model.start, s6, 0.051)
model.add_transition(model.start, s7, 0.2026)





model.add_transition(s1, s1, 0.80)
model.add_transition(s1, s2, 0.20)
model.add_transition(s2, s2, 0.90)
model.add_transition(s2, s3, 0.10)
model.add_transition(s3, s3, 0.70)

model.add_transition(s3, model.end, 0.30)

model.bake()

model

print(model.log_probability(list('ACGACTATTCGAT')))

print(", ".join(state.name for i, state in model.viterbi(list('ACGACTATTCGAT'))[1]))

