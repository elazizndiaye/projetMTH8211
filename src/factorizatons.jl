using LinearAlgebra

"""
Ensemble de fonctions qui implémentent les factorisation d'une matrice symmétrique indéfinie.
Les factorisations implémentées sont listées ci-dessous:
    * Bunch-Parlett (complet pivoting, PAP' = LDL')
    * Bunch-Kaufmann (partial pivoting, PAP' = LDL')
    * Bounded-Bunch-Kaufmann (rook pivoting, PAP' = LDL')
    * Aasen (PAP' = LTL')
"""

__ALPHA = (1+sqrt(17))/8


"""
Swap lines and columns of the given indices
"""
function _swap!(A::AbstractMatrix,i,j)
    temp = A[:,i]
    A[:,i] = A[:,j]
    A[:,j] = temp
    temp = A[i,:]
    A[i,:] = A[j,:]
    A[j,:] = temp
end

"""
Swap two line of the triangular matrix
Note: i1 <= i2
"""
function _swapL!(L::UnitLowerTriangular,i1,i2)
    temp = L[i1,1:i1-1]
    L[i1,1:i1-1] = L[i2,1:i1-1]
    L[i2,1:i1-1] = temp
end

"""
Find the pivot for Bunch-Parlett
"""
function _bp_pivot(A::AbstractMatrix)
    mu_0, pq = findmax(abs,A)
    mu_1, r = findmax(abs,diag(A))
    if mu_1>=__ALPHA*mu_0
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

"""
Bunch-Parlett factorization
"""
function bunch_parlett(A::AbstractMatrix)
    n, _ = size(A)
    L = UnitLowerTriangular(zeros(n,n))
    D = Tridiagonal(zeros(n,n))
    perms = Vector(1:1:n)
    k = 1
    Ak = copy(A)
    block_loc = zeros(Int64,n)
    while k<=n
        pivot, s, perm = _bp_pivot(Ak)
        if s==1
            temp = perms[k-1+perm[1][1]]
            perms[k-1+perm[1][1]] = perms[k-1+perm[1][2]] 
            perms[k-1+perm[1][2]] = temp
            _swap!(Ak, perm[1][1], perm[1][2])
            _swapL!(L,k-1+perm[1][1],k-1+perm[1][2])
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
            _swap!(Ak, perm[1][1], perm[1][2])
            _swap!(Ak, perm[2][1], perm[2][2])
            _swapL!(L,k-1+perm[1][1],k-1+perm[1][2])
            _swapL!(L,k-1+perm[2][1],k-1+perm[2][2])
            invE = inv(pivot) # TODO: use exact formula
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

"""
Find the pivot for Bunch-Kaufmann
"""
function _bk_pivot(A::AbstractMatrix)
    n,_ = size(A)
    if n<=1
        return A[1,1],-1,((1,1),1)
    end
    w1, r = findmax(abs,A[2:end,1])
    r = r+1;
    if abs(A[1,1])>=__ALPHA*w1
        pivot = A[1,1]
        s = 1
        perm = ((1,1),)
    else
        wr, _ = findmax(abs,[A[1:r-1,r]; A[r+1:end,r]])
        if abs(A[1,1])*wr>=__ALPHA*w1^2
            pivot = A[1,1]
            s = 1
            perm = ((1,1),)
        elseif abs(A[r,r])>=__ALPHA*wr
            pivot = A[r,r]
            s = 1
            perm = ((1,r),)
        else
            pivot = [[A[1,1],A[r,1]] [A[r,1], A[r,r]]]
            s = 2
            perm = ((2,r),)
        end
    end
    return pivot, s, perm
end

"""
Bunch-Kaufmann factorization
"""
function bunch_kaufmann(A::AbstractMatrix)
    n, _ = size(A)
    L = UnitLowerTriangular(zeros(n,n))
    D = Tridiagonal(zeros(n,n))
    perms = Vector(1:1:n)
    k = 1
    Ak = copy(A)
    block_loc = zeros(Int64,n)
    while k<=n
        pivot, s, perm = _bk_pivot(Ak)
        if s==1
            temp = perms[k-1+perm[1][1]]
            perms[k-1+perm[1][1]] = perms[k-1+perm[1][2]] 
            perms[k-1+perm[1][2]] = temp
            _swap!(Ak, perm[1][1], perm[1][2])
            _swapL!(L,k-1+perm[1][1],k-1+perm[1][2])
            invE = 1/pivot
            C = Ak[2:end,1]
            B = Ak[2:end,2:end]
            Ak = B-C*invE*C'
            L[k+1:end,k] = C*invE
            D[k,k] = pivot
            block_loc[k] = s
            k = k + 1
        elseif s==2
            temp = perms[k-1+perm[1][1]]
            perms[k-1+perm[1][1]] = perms[k-1+perm[1][2]]
            perms[k-1+perm[1][2]] = temp
            _swap!(Ak, perm[1][1], perm[1][2])
            _swapL!(L,k-1+perm[1][1],k-1+perm[1][2])
            invE = inv(pivot)
            C = Ak[3:end,1:2]
            B = Ak[3:end,3:end]
            Ak = B-C*invE*C'
            L[k+2:end,k:k+1] = C*invE
            D[k:k+1,k:k+1] = pivot
            block_loc[k] = s
            k = k + 2
        else
            D[k,k] = pivot
            block_loc[k] = 1
            k = k + 1
        end
    end
    return L, D, perms, block_loc
end

"""
Find the pivot for Bounded_Bunch-Kaufmann
"""
function _bbk_pivot(A::AbstractMatrix)
    n,_ = size(A)
    if n<=1
        return A[1,1],-1,((1,1),)
    end
    w1, r = findmax(abs,A[2:end,1])
    r = r+1;
    if w1==0.0
        return 0.0, 0, ()
    end
    if abs(A[1,1])>=__ALPHA*w1
        pivot = A[1,1]
        s = 1
        perm = ((1,1),)
    else
        i = 1
        wi = w1
        pivotFound = false
        while !pivotFound
            wr, _ = findmax(abs,[A[1:r-1,r]; A[r+1:end,r]])
            if abs(A[r,r])>=__ALPHA*wr
                pivot = A[r,r]
                s = 1
                perm = ((1,r),)
                pivotFound = true
            elseif wr == wi
                pivot = [[A[i,i],A[r,i]] [A[r,i], A[r,r]]]
                s = 2
                perm = ((1,i),(2,r),)
                pivotFound = true
            else
                i = r;
                wi = wr;
                _, r = findmax(abs,A[i+1:end,i])
                r = r+i
            end
        end
    end
    return pivot, s, perm
end

"""
Bounded-Bunch-Kaufmann factorization
"""
function bounded_bunch_kaufmann(A::AbstractMatrix)
    n, _ = size(A)
    L = UnitLowerTriangular(zeros(n,n))
    D = Tridiagonal(zeros(n,n))
    perms = Vector(1:1:n)
    k = 1
    Ak = copy(A)
    block_loc = zeros(Int64,n)
    while k<=n
        pivot, s, perm = _bbk_pivot(Ak)
        if s==1
            temp = perms[k-1+perm[1][1]]
            perms[k-1+perm[1][1]] = perms[k-1+perm[1][2]] 
            perms[k-1+perm[1][2]] = temp
            _swap!(Ak, perm[1][1], perm[1][2])
            _swapL!(L,k-1+perm[1][1],k-1+perm[1][2])
            invE = 1/pivot
            C = Ak[2:end,1]
            B = Ak[2:end,2:end]
            Ak = B-C*invE*C'
            L[k+1:end,k] = C*invE
            D[k,k] = pivot
            block_loc[k] = s
            k = k + 1
        elseif s==2
            temp = perms[k-1+perm[1][1]]
            perms[k-1+perm[1][1]] = perms[k-1+perm[1][2]]
            perms[k-1+perm[1][2]] = temp
            temp = perms[k-1+perm[2][1]]
            perms[k-1+perm[2][1]] = perms[k-1+perm[2][2]]
            perms[k-1+perm[2][2]] = temp
            _swap!(Ak, perm[1][1], perm[1][2])
            _swap!(Ak, perm[2][1], perm[2][2])
            _swapL!(L,k-1+perm[1][1],k-1+perm[1][2])
            _swapL!(L,k-1+perm[2][1],k-1+perm[2][2])
            invE = inv(pivot)
            C = Ak[3:end,1:2]
            B = Ak[3:end,3:end]
            Ak = B-C*invE*C'
            L[k+2:end,k:k+1] = C*invE
            D[k:k+1,k:k+1] = pivot
            block_loc[k] = s
            k = k + 2
        elseif s==-1
            D[k,k] = pivot
            block_loc[k] = 1
            k = k + 1
        else
            return L, D, perms, block_loc
        end
    end
    return L, D, perms, block_loc
end

"""
Aasen factorization
"""
function aasen(Ain::AbstractMatrix)
    A = copy(Ain)
    n, _ = size(A)
    L = UnitLowerTriangular(zeros(n,n))
    alpha = zeros(n)
    beta = zeros(n-1)
    h = zeros(n)
    v = zeros(n)
    perms = Vector(1:1:n)
    # Init
    alpha[1] = A[1,1]
    h[1] = alpha[1]
    # Iter
    for i in 2:n
        # Permutation
        for k in i:n
            v[k] = A[k,i-1] - sum([L[k,j]*h[j] for j in 1:i-1])
        end
        vr, r = findmax(abs,v[i:n])
        r = r + i - 1
        temp = v[r]
        v[r] = v[i]
        v[i] = temp
        _swap!(A,i,r)
        _swapL!(L,i,r)
        temp = perms[r]
        perms[r] = perms[i]
        perms[i] = temp
        # Finding column i of L and T
        h[i] = v[i] 
        beta[i-1] = h[i]
        for k in i+1:n
            L[k,i] = v[k]/h[i]
        end
        h[1] = alpha[1]*L[i,1] + beta[1]*L[i,2]
        for j in 2:i-1
            h[j] = beta[j-1]*L[i,j-1] + alpha[j]*L[i,j] + beta[j]*L[i,j+1]
        end
        h[i] = A[i,i] - sum([L[i,j]*h[j] for j in 1:i-1])
        alpha[i] = h[i] - beta[i-1]*L[i,i-1]
    end
    T = Tridiagonal(beta,alpha,beta)
    return L,T,perms
end

