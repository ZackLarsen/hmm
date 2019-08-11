from pomegranate import *
model = HiddenMarkovModel()




dists = [NormalDistribution(5, 1), NormalDistribution(1, 7), NormalDistribution(8,2)]
trans_mat = numpy.array([[0.7, 0.3, 0.0],
                        [0.0, 0.8, 0.2],
                        [0.0, 0.0, 0.9]])
starts = numpy.array([1.0, 0.0, 0.0])
ends = numpy.array([0.0, 0.0, 0.1])
model = HiddenMarkovModel.from_matrix(trans_mat, dists, starts, ends)


model








d1 = DiscreteDistribution({'A' : 0.35, 'C' : 0.20, 'G' : 0.05, 'T' : 0.40})
d2 = DiscreteDistribution({'A' : 0.25, 'C' : 0.25, 'G' : 0.25, 'T' : 0.25})
d3 = DiscreteDistribution({'A' : 0.10, 'C' : 0.40, 'G' : 0.40, 'T' : 0.10})

s1 = State(d1, name="s1")
s2 = State(d2, name="s2")
s3 = State(d3, name="s3")

model = HiddenMarkovModel('example')
model.add_states([s1, s2, s3])
model.add_transition(model.start, s1, 0.90)
model.add_transition(model.start, s2, 0.10)
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





