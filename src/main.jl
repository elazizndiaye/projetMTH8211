include("factorizatons.jl")

using LinearAlgebra, Permutations, Random
using QPSReader, QuadraticModels
using BenchmarkTools

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

# Test Solveur
As0 = [[6,12,3.0,-6] [12,-8,-13,4] [3.0,-13,-7,1] [-6,4,1,6]]
b0 = rand(4)
L1, D1, perms1, loc1 = bunch_kaufmann(As0)
xbk = solve(L1, D1, perms1, loc1, b0)
L2,T2,perms2 = aasen(As0)
xaas = solve(L2,T2,perms2,b0)
x0 = As0\b0
println("error solveur BK: ", norm(xbk-x0))
println("error solveur Aasen: ", norm(xaas-x0))

# Application à l'optimisation quadratique
# mm_path = fetch_mm()
# qm = QuadraticModel(readqps(joinpath(mm_path, "AUG2D.SIF")));

# Comparaison avec Lapack
n = 10
A_bm = rand(n,n)
syst = A_bm*A_bm'
# Bunch Kaufmann
println("Benchmark: Bunch-Kaufmann Lapack vs Projet")
@benchmark bunchkaufman(Symmetric(syst), false)
@benchmark bunch_kaufmann(syst)
println("Benchmark: Bounded-Bunch-Kaufmann (rook pivoting) Lapack vs Projet")
@benchmark bunchkaufman(Symmetric(syst), true)
@benchmark bounded_bunch_kaufmann(syst)