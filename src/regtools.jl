"""
    _qrE!(A, E, Q, B; withQ = true) 

Reduce the regular matrix pencil `A - λE` to an equivalent form `A1 - λE1 = Q1'*(A - λE)` using an
orthogonal or unitary transformation matrix `Q1` such that the transformed matrix `E1` is upper triangular.
The reduction is performed using the QR-decomposition of E.

The performed left orthogonal or unitary transformations Q1 are accumulated in the matrix `Q <- Q*Q1` 
if `withQ = true`. Otherwise, `Q` is unchanged.   

`Q1'*B` is returned in `B` unless `B = missing`.              
"""
function _qrE!(A::AbstractMatrix{T1}, E::AbstractMatrix{T1}, Q::Union{AbstractMatrix{T1},Nothing}, 
               B::Union{AbstractMatrix{T1},Missing} = missing; withQ::Bool = true) where T1 <: BlasFloat
   
   # fast return for dimensions 0 or 1
   size(A,1) <= 1 && return
   T = eltype(A)
   T <: Complex ? tran = 'C' : tran = 'T'
   # compute in-place the QR-decomposition E = Q1*E1 
   _, τ = LinearAlgebra.LAPACK.geqrf!(E) 
   # A <- Q1'*A
   LinearAlgebra.LAPACK.ormqr!('L',tran,E,τ,A)
   # Q <- Q*Q1
   withQ && LinearAlgebra.LAPACK.ormqr!('R','N',E,τ,Q)
   # B <- Q1'*B
   ismissing(B) || LinearAlgebra.LAPACK.ormqr!('L',tran,E,τ,B)
   triu!(E)
end
"""
    _svdlikeAE!(A, E, Q, Z, B, C; fast = true, atol1 = 0, atol2 = 0, rtol, withQ = true, withZ = true) -> (rE, rA22)

Reduce the regular matrix pencil `A - λE` to an equivalent form `A1 - λE1 = Q1'*(A - λE)*Z1` using 
orthogonal or unitary transformation matrices `Q1` and `Z1` such that the transformed matrices `A1` and `E1` 
are in the following SVD-like coordinate form

                   | A11-λE11 |  A12  |  A13  |
                   |----------|-------|-------|, 
        A1 - λE1 = |    A21   |  A22  |   0   |
                   |----------|-------|-------|
                   |    A31   |   0   |   0   |

where the `rE x rE` matrix `E11` and `rA22 x rA22` matrix `A22` are nosingular, and `E11` and `A22` are upper triangular, 
if `fast = true`, and diagonal, if `fast = false`. 
The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of `A`, the absolute tolerance for the nonzero elements of `E`,  and the relative tolerance 
for the nonzero elements of `A` and `E`. 

The reduction is performed using rank decisions based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.

The performed left orthogonal or unitary transformations Q1 are accumulated in the matrix `Q <- Q*Q1` 
if `withQ = true`. Otherwise, `Q` is unchanged.   
The performed right orthogonal or unitary transformations Z1 are accumulated in the matrix `Z <- Z*Z1` if `withZ = true`. 
Otherwise, `Z` is unchanged.  

`Q1'*B` is returned in `B` unless `B = missing` and `C*Z1` is returned in `C` unless `C = missing` .              

"""
function _svdlikeAE!(A::AbstractMatrix{T1}, E::AbstractMatrix{T1}, 
                     Q::Union{AbstractMatrix{T1},Nothing}, Z::Union{AbstractMatrix{T1},Nothing},
                     B::Union{AbstractMatrix{T1},Missing} = missing, C::Union{AbstractMatrix{T1},Missing} = missing; 
                     fast::Bool = true, atol1::Real = zero(real(T1)), atol2::Real = zero(real(T1)), 
                     rtol::Real = (size(A,1)*eps(real(float(one(T1)))))*iszero(min(atol1,atol2)), 
                     withQ::Bool = true, withZ::Bool = true) where T1 <: BlasFloat

   # fast returns for null dimensions
   n = size(A,1)
   if n == 0 
      return 0, 0
   end
   T = eltype(A)
   T <: Complex ? tran = 'C' : tran = 'T'
   
   if fast
      # compute in-place the QR-decomposition E*P1 = Q1*[E1;0] with column pivoting 
      _, τ, jpvt = LinearAlgebra.LAPACK.geqp3!(E)
      tol = max(atol2, rtol*abs(E[1,1]))
      rE = count(x -> x > tol, abs.(diag(E))) 
      n2 = n-rE
      i1 = 1:rE
      # A <- Q1'*A
      LinearAlgebra.LAPACK.ormqr!('L',tran,E,τ,A)
      # A <- A*P1
      A[:,:] = A[:,jpvt]      
      # Q <- Q*Q1
      withQ && LinearAlgebra.LAPACK.ormqr!('R','N',E,τ,Q)
      # Z <- Z*P1
      withZ && (Z[:,:] = Z[:,jpvt])
      # B <- Q1'*B
      ismissing(B) || LinearAlgebra.LAPACK.ormqr!('L',tran,E,τ,B)
      # E = [E1;0] 
      E[:,:] = [ triu(E[i1,:]); zeros(T,n2,n) ]
      # C <- C*P1
      ismissing(C) || (C[:,:] = C[:,jpvt])
      
      # compute in-place the complete orthogonal decomposition E*Z1 = [E11 0; 0 0] with E11 nonsingular and UT
      E1 = view(E,i1,:)
      _, tau = LinearAlgebra.LAPACK.tzrzf!(E1)
      # A <- A*Z1
      LinearAlgebra.LAPACK.ormrz!('R',tran,E1,tau,A)
      withZ && LinearAlgebra.LAPACK.ormrz!('R',tran,E1,tau,Z)
      # C <- C*Z1
      ismissing(C) || LinearAlgebra.LAPACK.ormrz!('R',tran,E1,tau,C); 
      E1[:,:] = [ triu(E[i1,i1]) zeros(T,rE,n2)  ] 
      n2 == 0 && (return rE, 0)
      # assume 
      #    A = [A11 A12]
      #        [A21 A22]
      # compute in-place the QR-decomposition A22*P2 = Q2*[R2;0] with column pivoting 
      i22 = rE+1:n
      A22 = view(A,i22,i22)
      _, τ, jpvt = LinearAlgebra.LAPACK.geqp3!(A22)
      tolA = max(atol2, rtol*opnorm(A,1))
      rA22 = count(x -> x > tolA, abs.(diag(A22))) 
      n3 = n2-rA22
      i2 = rE+1:rE+rA22
      i3 = rE+rA22+1:n
      # A21 <- Q2'*A21
      LinearAlgebra.LAPACK.ormqr!('L',tran,A22,τ,view(A,i22,i1))
      # A12 <- A12*P1
      A[i1,i22] = A[i1,i22[jpvt]]      
      # Q <- Q*Q2
      withQ && LinearAlgebra.LAPACK.ormqr!('R','N',A22,τ,view(Q,:,i22))
      # Z <- Z*P2
      withZ && (Z[:,i22] = Z[:,i22[jpvt]])
      # B <- Q2'*B
      ismissing(B) || LinearAlgebra.LAPACK.ormqr!('L',tran,A22,τ,view(B,i22,:))
      # A22 = [R2;0] 
      A22[:,:] = [ triu(A[i2,i22]); zeros(T,n3,n2) ]
      # R <- R*P1
      ismissing(C) || (C[:,i22] = C[:,i22[jpvt]])
   
      # compute in-place the complete orthogonal decomposition R2*Z2 = [A2 0; 0 0] with A22 nonsingular and UT
      A2 = view(A,i2,i22)
      _, tau = LinearAlgebra.LAPACK.tzrzf!(A2)
      # A12 <- A12*Z2
      LinearAlgebra.LAPACK.ormrz!('R',tran,A2,tau,view(A,i1,i22))
      withZ && LinearAlgebra.LAPACK.ormrz!('R',tran,A2,tau,view(Z,:,i22))
      # C <- C*Z2
      ismissing(C) || LinearAlgebra.LAPACK.ormrz!('R',tran,A2,tau,view(C,:,i22)); 
      A2[:,:] = [ triu(A[i2,i2]) zeros(T,rA22,n3)  ] 
   else
      # compute the complete orthogonal decomposition of E using the SVD-decomposition
      U, S, Vt = LinearAlgebra.LAPACK.gesdd!('A',E)
      tolE = max(atol2, rtol*S[1])
      rE = count(x -> x > tolE, S) 
      n2 = n-rE
      # A <- A*V
      A[:,:] = A[:,:]*Vt'
      # A <- U'*A
      A[:,:] = U'*A[:,:]
      # Q <- Q*U
      withQ && (Q[:,:] = Q[:,:]*U)
      # Z <- Q*V
      withZ && (Z[:,:] = Z[:,:]*Vt')
      # B <- U'*B
      ismissing(B) || (B[:,:] = U'*B[:,:])
      # C <- C*V
      ismissing(C) || (C[:,:] = C[:,:]*Vt')
      E[:,:] = [ Diagonal(S[1:rE]) zeros(T,rE,n2) ; zeros(T,n2,n) ]
      n2 == 0 && (return rE, 0)
      # assume 
      #    A = [A11 A12]
      #        [A21 A22]
      # compute the complete orthogonal decomposition of A22 using the SVD-decomposition
      i22 = rE+1:n
      A22 = view(A,i22,i22)
      U, S, Vt = LinearAlgebra.LAPACK.gesdd!('A',A22)
      tolA = max(atol2, rtol*opnorm(A,1))
      rA22 = count(x -> x > tolA, S) 
      n3 = n2-rA22
      i1 = 1:rE
      i2 = rE+1:rE+rA22
      i3 = rE+rA22+1:n
      # A12 <- A12*V
      A[i1,i22] = A[i1,i22]*Vt'
      # A21 <- U'*A21
      A[i22,i1] = U'*A[i22,i1]
      # Q <- Q*U
      withQ && (Q[:,i22] = Q[:,i22]*U)
      # Z <- Q*V
      withZ && (Z[:,i22] = Z[:,i22]*Vt')
      # B <- U'*B
      ismissing(B) || (B[i22,:] = U'*B[i22,:])
      # C <- C*V
      ismissing(C) || (C[:,i22] = C[:,i22]*Vt')
      A22[:,:] = [ Diagonal(S[1:rA22]) zeros(T,rA22,n3); zeros(T,n3,n2) ]
   end
   return rE, rA22   
end  
"""
    isregular(A, E, γ; atol::Real = 0,  rtol::Real = atol > 0 ? 0 : n*ϵ) -> Bool

Test whether the matrix pencil `A-λE` is regular at `λ = γ` (i.e., `A-λE` is square and ``{\\small\\det(A-λE) \\neq 0}``). 
The underlying computational procedure checks the maximal rank of `A-γE` if `γ` is finite and of `E` if 
`γ` is infinite . 

The keyword arguements `atol` and `rtol` specify the absolute and relative tolerances for the nonzero
elements of `A-γE`, respectively. 
The default relative tolerance is `n*ϵ`, where `n` is the size of  `A`, and `ϵ` is the 
machine epsilon of the element type of `A`. 
"""
function isregular(A::AbstractMatrix, E::Union{AbstractMatrix,Nothing}, γ::Number; atol::Real = zero(real(eltype(A))), 
                   rtol::Real = (min(size(A)...)*eps(real(float(one(eltype(A))))))*iszero(atol))
   
   m, n = size(A)
   m == n || (return false)
   E === nothing && (return rank(A, atol = atol, rtol = rtol) == n )
   (m,n) == size(E) || throw(DimensionMismatch("A and E must have the same dimensions"))
   if isinf(γ) 
      return rank(E,atol = atol,rtol=rtol) == n
   else
      return rank(A-γ*E,atol = atol,rtol=rtol) == n
   end
end
"""
    isregular(A, E; atol1::Real = 0, atol2::Real = 0, rtol::Real=min(atol1,atol2)>0 ? 0 : n*ϵ) -> Bool

Test whether the matrix pencil `A-λE` is regular (i.e., `A-λE` is square and ``{\\small\\det(A-λE) \\not\\equiv 0}``). 
The underlying computational procedure reduces the pencil `A-λE` to an appropriate Kronecker-like form, 
which provides information on the rank of `A-λE`. 

The keyword arguements `atol1`, `atol2` and `rtol` specify the absolute tolerance for the nonzero
elements of `A`, the absolute tolerance for the nonzero elements of `E`, and the relative tolerance 
for the nonzero elements of `A` and `E`, respectively. 
The default relative tolerance is `n*ϵ`, where `n` is the size of  `A`, and `ϵ` is the 
machine epsilon of the element type of `A`. 
"""
function isregular(A::AbstractMatrix, E::Union{AbstractMatrix,Nothing}; atol1::Real = zero(real(eltype(A))), atol2::Real = zero(real(eltype(A))), 
                   rtol::Real = (min(size(A)...)*eps(real(float(one(eltype(A))))))*iszero(min(atol1,atol2)))
   mA, nA = size(A)
   mA == nA || (return false)
   E === nothing && (return rank(A, atol = atol1, rtol = rtol) == mA )
   (mA,nA) == size(E) || throw(DimensionMismatch("A and E must have the same dimensions"))
   T = promote_type(eltype(A), eltype(E))
   T <: BlasFloat || (T = promote_type(Float64,T))
   A1 = copy_oftype(A,T)
   E1 = copy_oftype(E,T)

   Q = nothing
   Z = nothing

   # Step 0: Reduce to the standard form
   n, m, p = _preduceBF!(A1, E1, Q, Z; atol = atol2, rtol = rtol, fast = false, withQ = false, withZ = false) 

   mrinf = 0
   tol1 = max(atol1, rtol*opnorm(A1,1))
   while m > 0
      # Steps 1 & 2: Standard algorithm PREDUCE
      i1 = mrinf+1:mA
      τ, ρ = _preduce1!(n, m, m, view(A1,i1,i1), view(E1,i1,i1), Q, Z, tol1; fast = false, withQ = false, withZ = false)
      ρ+τ == m || (return false)
      mrinf += m
      n -= ρ
      m = ρ
   end
   return true                                            
end
"""
    isunimodular(A, E; atol1::Real = 0, atol2::Real = 0, rtol::Real=min(atol1,atol2)>0 ? 0 : n*ϵ) -> Bool

Test whether the matrix pencil `A-λE` is unimodular (i.e., `A-λE` is square, regular and `det(A-λE) == constant`). 
The underlying computational procedure reduces the pencil `A-λE` to an appropriate Kronecker-like form, 
which provides information to check the full rank of `A-λE` and the lack of finite eigenvalues. 

The keyword arguements `atol1`, `atol2` and `rtol` specify the absolute tolerance for the nonzero
elements of `A`, the absolute tolerance for the nonzero elements of `E`, and the relative tolerance 
for the nonzero elements of `A` and `E`, respectively. 
The default relative tolerance is `n*ϵ`, where `n` is the size of  `A`, and `ϵ` is the 
machine epsilon of the element type of `A`. 
"""
function isunimodular(A::AbstractMatrix, E::Union{AbstractMatrix,Nothing}; atol1::Real = zero(real(eltype(A))), atol2::Real = zero(real(eltype(A))), 
                      rtol::Real = (size(A,1)*eps(real(float(one(eltype(A))))))*iszero(min(atol1,atol2)))
   
   mA, nA = size(A)
   mA == nA || (return false)
   E === nothing && (return rank(A, atol = atol1, rtol = rtol) == mA )
   (mA,nA) == size(E) || throw(DimensionMismatch("A and E must have the same dimensions"))
   mA == 0 && (return true)
   T = promote_type(eltype(A), eltype(E))
   T <: BlasFloat || (T = promote_type(Float64,T))
   A1 = copy_oftype(A,T)
   E1 = copy_oftype(E,T)

   Q = nothing
   Z = nothing

   # Step 0: Reduce to the standard form
   n, m, p = _preduceBF!(A1, E1, Q, Z; atol = atol2, rtol = rtol, fast = false, withQ = false, withZ = false) 
   n == 0 && (return true)
   mrinf = 0
   tol1 = max(atol1, rtol*opnorm(A1,1))
   while m > 0
      # Steps 1 & 2: Standard algorithm PREDUCE
      i1 = mrinf+1:mA
      τ, ρ = _preduce1!(n, m, m, view(A1,i1,i1), view(E1,i1,i1), Q, Z, tol1; fast = false, withQ = false, withZ = false)
      ρ+τ == m || (return false)
      mrinf += m
      n -= ρ
      m = ρ
   end
   return n == 0                                           
end
"""
    fisplit(A, E, B, C; fast = true, finite_infinite = false, atol1 = 0, atol2 = 0, rtol, withQ = true, withZ = true) -> (At, Et, Bt, Ct, Q, Z, ν, nf, ni)

Reduce the regular matrix pencil `A - λE` to an equivalent form `At - λEt = Q'*(A - λE)*Z` using 
orthogonal or unitary transformation matrices `Q` and `Z` such that the transformed matrices `At` and `Et` are in one of the
following block upper-triangular forms:

(1) if `finite_infinite = false`, then
 
                   | Ai-λEi |   *     |
        At - λEt = |--------|---------|, 
                   |    O   | Af-λEf  |
 
where the `ni x ni` subpencil `Ai-λEi` contains the infinite elementary 
divisors and the `nf x nf` subpencil `Af-λEf`, with `Ef` nonsingular and upper triangular, contains the finite eigenvalues of the pencil `A-λE`.

The subpencil `Ai-λEi` is in a staircase form, with `Ai` nonsingular and upper triangular and `Ei` nilpotent and upper triangular. 
The `nb`-dimensional vector `ν` contains the dimensions of the diagonal blocks
of the staircase form  `Ai-λEi` such that `i`-th block has dimensions `ν[i] x ν[i]`. 
The difference `ν[i]-ν[i+1]` for `i = 1, 2, ..., nb` is the number of infinite elementary divisors of degree `i` 
(with `ν[nb+1] = 0`).


(2) if `finite_infinite = true`, then
 
                   | Af-λEf |   *    |
        At - λEt = |--------|--------|, 
                   |   O    | Ai-λEi |
 
where the `nf x nf` subpencil `Af-λEf`, with `Ef` nonsingular and upper triangular, 
contains the finite eigenvalues of the pencil `A-λE` and the `ni x ni` subpencil `Ai-λEi` 
contains the infinite elementary divisors.

The subpencil `Ai-λEi` is in a staircase form, with `Ai` nonsingular and upper triangular  and `Ei` nilpotent and upper triangular. 
The `nb`-dimensional vectors `ν` contains the dimensions of the diagonal blocks
of the staircase form `Ai-λEi` such that `i`-th block has dimensions `ν[i] x ν[i]`. 
The difference `ν[nb-j+1]-ν[nb-j]` for `j = 1, 2, ..., nb` is the number of infinite elementary 
divisors of degree `j` (with `ν[0] = 0`).

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of `A`, the absolute tolerance for the nonzero elements of `E`,  and the relative tolerance 
for the nonzero elements of `A` and `E`. 

The reduction is performed using rank decisions based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.

The performed left orthogonal or unitary transformations are accumulated in the matrix `Q` if `withQ = true`. 
Otherwise, `Q` is set to `nothing`.   
The performed right orthogonal or unitary transformations are accumulated in the matrix `Z` if `withZ = true`. 
Otherwise, `Z` is set to `nothing`.  

`Bt = Q'*B`, unless `B = missing`, in which case `Bt = missing` is returned, and `Ct = C*Z`, 
unless `C = missing`, in which case `Ct = missing` is returned .              

"""
function fisplit(A::AbstractMatrix, E::AbstractMatrix, B::Union{AbstractMatrix,Missing}, C::Union{AbstractMatrix,Missing}; 
   fast::Bool = true, finite_infinite::Bool = false, 
   atol1::Real = zero(real(eltype(A))), atol2::Real = zero(real(eltype(E))), 
   rtol::Real = (size(A,1)*eps(real(float(one(eltype(A))))))*iszero(min(atol1,atol2)), 
   withQ::Bool = true, withZ::Bool = true)

   n = LinearAlgebra.checksquare(A)
   n == LinearAlgebra.checksquare(E) || throw(DimensionMismatch("A and E must have the same dimensions"))          
   (!ismissing(B) && n != size(B,1)) && throw(DimensionMismatch("A and B must have the same number of rows"))
   (!ismissing(C) && n != size(C,2)) && throw(DimensionMismatch("A and C must have the same number of columns"))
   T = promote_type(eltype(A), eltype(E))
   ismissing(B) || (T = promote_type(T,eltype(B)))
   ismissing(C) || (T = promote_type(T,eltype(C)))
   T <: BlasFloat || (T = promote_type(Float64,T))

   A1 = copy_oftype(A,T)   
   E1 = copy_oftype(E,T)
   ismissing(B) ? B1 = missing : B1 = copy_oftype(B,T)
   ismissing(C) ? C1 = missing : C1 = copy_oftype(C,T)

   withQ ? (Q = Matrix{T}(I,n,n)) : (Q = nothing)
   withZ ? (Z = Matrix{T}(I,n,n)) : (Z = nothing)

   ν, nf, ni = fisplit!(A1, E1, Q, Z, B1, C1; 
                        fast  = fast, finite_infinite = finite_infinite, 
                        atol1 = atol1, atol2 = atol2, rtol = rtol, withQ = withQ, withZ = withZ)


   
   return A1, E1, B1, C1, Q, Z, ν, nf, ni                                             
end
"""
    fisplit!(A, E, Q, Z, B, C; fast = true, finite_infinite = false, atol1 = 0, atol2 = 0, rtol, withQ = true, withZ = true) -> (ν, nf, ni)

Reduce the regular matrix pencil `A - λE` to an equivalent form `At - λEt = Q1'*(A - λE)*Z1` using 
orthogonal or unitary transformation matrices `Q1` and `Z1` such that the transformed matrices `At` and `Et` are in one of the
following block upper-triangular forms:

(1) if `finite_infinite = false`, then
 
                   | Ai-λEi |   *     |
        At - λEt = |--------|---------|, 
                   |    O   | Af-λEf  |
 
where the `ni x ni` subpencil `Ai-λEi` contains the infinite elementary 
divisors and the `nf x nf` subpencil `Af-λEf`, with `Ef` nonsingular and upper triangular, contains the finite eigenvalues of the pencil `A-λE`.

The subpencil `Ai-λEi` is in a staircase form, with `Ai` nonsingular and upper triangular and `Ei` nilpotent and upper triangular. 
The `nb`-dimensional vector `ν` contains the dimensions of the diagonal blocks
of the staircase form  `Ai-λEi` such that `i`-th block has dimensions `ν[i] x ν[i]`. 
The difference `ν[i]-ν[i+1]` for `i = 1, 2, ..., nb` is the number of infinite elementary divisors of degree `i` 
(with `ν[nb+1] = 0`).


(2) if `finite_infinite = true`, then
 
                   | Af-λEf |   *    |
        At - λEt = |--------|--------|, 
                   |   O    | Ai-λEi |
 
where the `nf x nf` subpencil `Af-λEf`, with `Ef` nonsingular and upper triangular, 
contains the finite eigenvalues of the pencil `A-λE` and the `ni x ni` subpencil `Ai-λEi` 
contains the infinite elementary divisors.

The subpencil `Ai-λEi` is in a staircase form, with `Ai` nonsingular and upper triangular and `Ei` nilpotent and upper triangular. 
The `nb`-dimensional vectors `ν` contains the dimensions of the diagonal blocks
of the staircase form `Ai-λEi` such that `i`-th block has dimensions `ν[i] x ν[i]`. 
The difference `ν[nb-j+1]-ν[nb-j]` for `j = 1, 2, ..., nb` is the number of infinite elementary 
divisors of degree `j` (with `ν[0] = 0`).

The reduced matrices `At` and `Et` are returned in `A` and `E`, respectively, 
while `Q1'*B` is returned in `B`, unless `B = missing`, and `C*Z1`, is returned in `C`, 
unless `C = missing`.   

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of `A`, the absolute tolerance for the nonzero elements of `E`,  and the relative tolerance 
for the nonzero elements of `A` and `E`. 

The reduction is performed using rank decisions based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.

The performed left orthogonal or unitary transformations Q1 are accumulated in the matrix `Q <- Q*Q1` 
if `withQ = true`. Otherwise, `Q` is unchanged.   
The performed right orthogonal or unitary transformations Z1 are accumulated in the matrix `Z <- Z*Z1` 
if `withZ = true`. Otherwise, `Z` is unchanged.            

"""
function fisplit!(A::AbstractMatrix{T1}, E::AbstractMatrix{T1}, 
                  Q::Union{AbstractMatrix{T1},Nothing}, Z::Union{AbstractMatrix{T1},Nothing},
                  B::Union{AbstractMatrix{T1},Missing} = missing, C::Union{AbstractMatrix{T1},Missing} = missing; 
                  fast::Bool = true, finite_infinite::Bool = false, 
                  atol1::Real = zero(real(T1)), atol2::Real = zero(real(T1)), 
                  rtol::Real = (size(A,1)*eps(real(float(one(T1)))))*iszero(min(atol1,atol2)), 
                  withQ::Bool = true, withZ::Bool = true) where T1 <: BlasFloat

   # fast returns for null dimensions
   n = size(A,1)
   ν = Vector{Int}(undef,n)
   if n == 0 
      return ν, 0, 0
   end
   eltype(A) <: Complex ? tran = 'C' : tran = 'T'

   # Step 0: Reduce to the standard form
   n1, m1, p1 = _preduceBF!(A, E, Q, Z, B, C; atol = atol2, rtol = rtol, fast = fast) 
        
   tolA = max(atol1, rtol*opnorm(A,1))     
   
   if finite_infinite

      # Reduce A-λE to the Kronecker-like form by splitting the finite-infinite structures
      #
      #                  | Af - λ Ef |     *      |
      #      At - λ Et = |-----------|------------|,
      #                  |    0      | Ai -  λ Ei |
      # 
      # where Ai - λ Ei is in a staircase form.  

      i = 0
      ni = 0
      nf = n1
      while p1 > 0
         # Step 1 & 2: Dual algorithm PREDUCE
         τ, ρ  = _preduce2!(nf, m1, p1, A, E, Q, Z, tolA, B, C; fast = fast, 
                            rtrail = ni, ctrail = ni, withQ = withQ, withZ = withZ)
         ρ+τ == p1 || error("A-λE is not regular")
         ni += p1
         nf -= ρ
         i += 1
         ν[i] = p1
         p1 = ρ
         m1 -= τ 
      end
      k1 = nf+1
      for k = 1:i
          nk = ν[k]
          k2 = k1+nk-1
          kk = k1:k2
          if nk > 1
             Ak = view(A,kk,kk)
             tau = similar(A,nk)
             LinearAlgebra.LAPACK.geqrf!(Ak,tau)
             LinearAlgebra.LAPACK.ormqr!('L',tran,Ak,tau,view(A,kk,k2+1:n))
             withQ && LinearAlgebra.LAPACK.ormqr!('R','N',Ak,tau,view(Q,:,kk))
             LinearAlgebra.LAPACK.ormqr!('L',tran,Ak,tau,view(E,kk,k2+1:n))
             ismissing(B) || LinearAlgebra.LAPACK.ormqr!('L',tran,Ak,tau,view(B,kk,:))
             triu!(Ak)
          end
          k1 = k2+1
      end        
      return reverse(ν[1:i]), nf, ni                                             
   else

      # Reduce A-λE to the Kronecker-like form by splitting the infinite-finite structures
      #
      #                  | Ai - λ Ei |    *      |
      #      At - λ Et = |-----------|-----------|
      #                  |    0      | Af - λ Ef |
      #
      # where Ai - λ Ei is in a staircase form.  

      i = 0
      ni = 0
      nf = n1
      while m1 > 0
         # Steps 1 & 2: Standard algorithm PREDUCE
         τ, ρ = _preduce1!(nf, m1, p1, A, E, Q, Z, tolA, B, C; fast = fast, 
                           roff = ni, coff = ni, withQ = withQ, withZ = withZ)
         ρ+τ == m1 || error("A-λE is not regular")
         ni += m1
         nf -= ρ
         i += 1
         ν[i] = m1
         m1 = ρ
         p1 -= τ 
      end

      k2 = ni 
      for k = i:-1:1
          nk = ν[k]
          k1 = k2-nk+1
          kk = k1:k2
          if nk > 1
             Ak = view(A,kk,kk)
             tau = similar(A,nk)
             LinearAlgebra.LAPACK.gerqf!(Ak,tau)
             LinearAlgebra.LAPACK.ormrq!('R',tran,Ak,tau,view(A,1:k1-1,kk))
             withZ && LinearAlgebra.LAPACK.ormrq!('R',tran,Ak,tau,view(Z,:,kk)) 
             LinearAlgebra.LAPACK.ormrq!('R',tran,Ak,tau,view(E,1:k1-1,kk))
             ismissing(C) || LinearAlgebra.LAPACK.ormrq!('R',tran,Ak,tau,view(C,:,kk)) 
             triu!(Ak)
          end
         #  the following code can be used to make the supradiagonal blocks of E upper triangular
         #  but it is not necessary in this context
         #  if k > 1 
         #     nk1 = ν[k-1]
         #     k1e = k1 - nk1
         #     k2e = k1 - 1
         #     kke = k1e:k2e
         #     if nk1 > 1
         #        Ek = view(E,kke,kk)
         #        tau = similar(A,nk)
         #        LinearAlgebra.LAPACK.geqrf!(Ek,tau)
         #        LinearAlgebra.LAPACK.ormqr!('L',tran,Ek,tau,view(A,kke,k1e:n))
         #        withQ && LinearAlgebra.LAPACK.ormqr!('R','N',Ek,tau,view(Q,:,kke)) 
         #        LinearAlgebra.LAPACK.ormqr!('L',tran,Ek,tau,view(E,kke,k2+1:n))
         #        ismissing(B) || LinearAlgebra.LAPACK.ormqr!('L',tran,Ek,tau,view(B,kke,:))
         #        triu!(Ek)
         #     end
         k2 = k1-1
      end        
   
      return ν[1:i], nf, ni                                             
   end
end
