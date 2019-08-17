
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

mtcars = dataset("datasets", "mtcars");
mtcars

#=
train = readtable("train.csv")
size(train)
names(train)
head(train, 10)
=#




formula = @formula(Perf~CycT+MMin+MMax+Cach+ChMin+ChMax)
model2 = glm(formula, train, Poisson(), LogLink())
model3 = glm(formula, train, Gamma(),  InverseLink())

coef(model2)

coef(model3)
