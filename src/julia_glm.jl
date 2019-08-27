
using Pkg
using RCall
using RDatasets
using DataFrames

using GLM, RDatasets, DataFrames, Distributions, PyPlot, Random, LinearAlgebra

Random.seed!(0)

#=
Pkg.add("RDatsets")
Pkg.add("RCall")
Pkg.add("GLM")
=#


RDatasets.datasets()
RDatasets.datasets("Zelig")
RDatasets.datasets("MASS")

Chem97 = dataset("mlmRev", "Chem97")
print(names(Chem97))
describe(Chem97[:Score])

boston = dataset("MASS", "boston")
boston
names(boston)
boston[:5]



mtcars = dataset("datasets", "mtcars");
mtcars

#=
train = readtable("train.csv")
size(train)
names(train)
head(train, 10)
=#




n = size(Chem97)
df = Chem97[shuffle(1:n),:]
pTrain = 0.29
lastTindex = Int(floor(n*(1-pTrain)))
numTest = n - lastTindex
train = df[1:lastTindex,:]
test = df[lastTindex+1:n,:]



formula = @formula(GCSEScore~Gender+Age+School+Lea)
model2 = glm(formula, train, Poisson(), LogLink())
model3 = glm(formula, train, Gamma(),  InverseLink())

coef(model2)

coef(model3)
