

Pkg.add("Plots")
Pkg.add("StatPlots")

Pkg.add("PyPlot")
Pkg.add("ScikitLearn")
Pkg.add("PyCall")

Pkg.add("RDatsets")
Pkg.add("RCall")

using RCall, RDatasets

using PyCall

using DataFrames
using SparseArrays



train = readtable(“train.csv”)
size(train)
names(train)

head(train, 10)

describe(train[:LoanAmount])

countmap(train[:Property_Area])








@pyimport pandas as pd
df = pd.read_csv("train.csv")



mtcars = datasets("datasets", "mtcars");
library(ggplot2)
ggplot($mtcars, aes(x = WT, y=MPG)) + geom_point()












A = rand(4,4)
println(A[1,1])

A[1:3,1:3]

B = randn(100,100)





function testloops()
        b = rand(1000,1000)
        c = 0
        @time for i in bidx
                c+=b[i]
        end
end
testloops()




B = view(A,1:3,1:3) # Doesn't create a copy
C = reshape(A,8,2)



A = sparse([1;2;3],[2;2;1],[3;4;5])

# Convert into dense array:
Array(A)
