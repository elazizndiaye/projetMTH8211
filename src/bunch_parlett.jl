using LinearAlgebra, Permutations

alpha = (1+sqrt(17))/8

function bp_pivot(A::AbstractMatrix)
    mu_0, pq = findmax(abs,A)
    mu_1, r = findmax(abs,diag(A))
    if mu_1>=alpha*mu_0
        pivot = A[r,r]
        s = 1
        perm = ((1,r),)
    else
        p = pq[1]
        q = pq[2]
        if p>q
            p,q = q,p
        end
        pivot = [[A[p,p],A[q,p]] [A[q,p], A[q,q]]]
        s = 2
        perm = ((1,p), (2,q))
    end
    return pivot, s, perm
end
function bp_swap!(A::AbstractMatrix,i,j)
    temp = A[:,i]
    A[:,i] = A[:,j]
    A[:,j] = temp
    temp = A[i,:]
    A[i,:] = A[j,:]
    A[j,:] = temp
end

function bp_fact(A::AbstractMatrix)
    n, _ = size(A)
    L = UnitLowerTriangular(zeros(n,n))
    D = Tridiagonal(zeros(n,n))
    perms = Vector(1:1:n)
    k = 1
    Ak = copy(A)
    block_loc = zeros(Int64,n)
    while k<=n
        pivot, s, perm = bp_pivot(Ak)
        if s==1
            temp = perms[k-1+perm[1][1]]
            perms[k-1+perm[1][1]] = perms[k-1+perm[1][2]] 
            perms[k-1+perm[1][2]] = temp
            bp_swap!(Ak, perm[1][1], perm[1][2])
            invE = 1/pivot
            C = Ak[2:end,1]
            B = Ak[2:end,2:end]
            Ak = B-C*invE*C'
            L[k+1:end,k] = C*invE
            D[k,k] = pivot
            block_loc[k] = s
            k = k + 1
        else
            temp = perms[k-1+perm[1][1]]
            perms[k-1+perm[1][1]] = perms[k-1+perm[1][2]]
            perms[k-1+perm[1][2]] = temp
            temp = perms[k-1+perm[2][1]]
            perms[k-1+perm[2][1]] = perms[k-1+perm[2][2]]
            perms[k-1+perm[2][2]] = temp
            bp_swap!(Ak, perm[1][1], perm[1][2])
            bp_swap!(Ak, perm[2][1], perm[2][2])
            invE = inv(pivot)
            C = Ak[3:end,1:2]
            B = Ak[3:end,3:end]
            Ak = B-C*invE*C'
            L[k+2:end,k:k+1] = C*invE
            D[k:k+1,k:k+1] = pivot
            block_loc[k] = s
            k = k + 2
        end
    end
    return L, D, perms, block_loc
end

A = rand(5,5)
As = A*A'

L,D,perms,loc = bp_fact(As)
P = Matrix(Permutation(perms))
As2 = P'*L*D*L'*P
norm(As-As2)