include("factorizatons.jl")

using LinearAlgebra, Permutations, Random

A = rand(5,5)
As1 = A*A'
As2 = [[6,12,3.0,-6] [12,-8,-13,4] [3.0,-13,-7,1] [-6,4,1,6]]
As3 = Matrix(Symmetric(rand(MersenneTwister(0),4,4)))
allMatrix = (As1, As2, As3)
for i in 1:3
    As = allMatrix[i]
    # Bunch Parlett
    L, D, perms, loc = bunch_parlett(As)
    P = Matrix(Permutation(perms))
    Asx = P*L*D*L'*P'
    println("Problem $i: BP , error = ", norm(As-Asx))
    # Bunch Kaufmann
    L, D, perms, loc = bunch_kaufmann(As)
    P = Matrix(Permutation(perms))
    Asx = P*L*D*L'*P'
    println("Problem $i: BK , error = ", norm(As-Asx))
    # bounded Bunch Kaufmann
    L, D, perms, loc = bounded_bunch_kaufmann(As)
    P = Matrix(Permutation(perms))
    Asx = P*L*D*L'*P'
    println("Problem $i: BBK , error = ", norm(As-Asx))
    # Aasen
    L,T,perms = aasen(As)
    P = Matrix(Permutation(perms))
    Asx = P*L*T*L'*P'
    println("Problem $i: Aasen , error = ", norm(As-Asx))
end