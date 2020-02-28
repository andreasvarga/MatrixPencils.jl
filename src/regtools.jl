"""
    isregular(M, N; atol1::Real = 0, atol2::Real = 0, rtol::Real=min(atol1,atol2)>0 ? 0 : n*ϵ) -> Bool

Test whether the linear pencil `M-λN` is regular (i.e., det(M-λN) !== 0). The underlying computational procedure
reduces the pencil `M-λN` to an appropriate Kronecker-like form (KLF), which provides information on the rank of `M-λN`. 

The keyword arguements `atol1`, `atol2` and `rtol` specify the absolute tolerance for the nonzero
elements of `M`, the absolute tolerance for the nonzero elements of `N`, and the relative tolerance for the nonzero elements of `M` and `N`, respectively. 
The default relative tolerance is `n*ϵ`, where `n` is the size of the smallest dimension of `M`, and `ϵ` is the 
machine epsilon of the element type of `M`. 
"""
function isregular(M ::AbstractMatrix, N::AbstractMatrix; atol1::Real = zero(eltype(M)), atol2::Real = zero(eltype(M)), 
                   rtol::Real = (min(size(M)...)*eps(real(float(one(eltype(M))))))*iszero(min(atol1,atol2)))
   
   mM, nM = size(M)
   (mM,nM) == size(N) || throw(DimensionMismatch("M and N must have the same dimensions"))
   mM == nM || (return false)
   isa(M,Adjoint) && (M = copy(M))
   isa(N,Adjoint) && (N = copy(N))
   T = promote_type(eltype(M), eltype(N))
   T <: BlasFloat || (T = promote_type(Float64,T))
   eltype(M) == T || (M = convert(Matrix{T},M))
   eltype(N) == T || (N = convert(Matrix{T},N)) 
   Q = nothing
   Z = nothing

   # Step 0: Reduce to the standard form
   n, m, p = _preduceBF!(M, N, Q, Z; atol = atol2, rtol = rtol, fast = false, withQ = false, withZ = false) 

   mrinf = 0
   nrinf = 0
   tol1 = max(atol1, rtol*opnorm(M,1))
   while m > 0
      # Steps 1 & 2: Standard algorithm PREDUCE
      τ, ρ = _preduce1!(n, m, p, M, N, Q, Z, tol1; fast = false, 
                        roff = mrinf, coff = nrinf, withQ = false, withZ = false)
      ρ+τ == m || (return false)
      mrinf += ρ+τ
      nrinf += m
      n -= ρ
      m = ρ
      p -= τ 
   end
   return true                                            
end
"""
    fisplit(A, E, B, C; fast = true, finite_infinite = false, atol1 = 0, atol2 = 0, rtol, withQ = true, withZ = true) -> (F, G, H, L, Q, Z, ν, nf, ni)

Reduce the linear regular pencil `A - λE` to an equivalent form `F - λG = Q'*(A - λE)*Z` using 
orthogonal or unitary transformation matrices `Q` and `Z` such that the transformed matrices `F` and `G` are in one of the
following block upper-triangular forms:

(1) if `finite_infinite = false`, then
 
                   | Ai-λEi |   *     |
          F - λG = |--------|---------|, 
                   |    O   | Af-λEf  |
 
where the `ni x ni` subpencil `Ai-λEi` contains the infinite elementary 
divisors and the `nf x nf` subpencil `Af-λEf`, with `Ef` nonsingular and upper triangular, contains the finite eigenvalues of the pencil `A-λE`.

The subpencil `Ai-λEi` is in a staircase form, with `Ai` nonsingular and `Ei` nilpotent. 
The `nb`-dimensional vector `ν` contains the dimensions of the diagonal blocks
of the staircase form  `Ai-λEi` such that `i`-th block has dimensions `ν[i] x ν[i]`. 
The difference `ν[i]-ν[i+1]` for `i = 1, 2, ..., nb` is the number of infinite elementary divisors of degree `i` 
(with `ν[nb+1] = 0`).


(2) if `finite_infinite = true`, then
 
                   | Af-λEf |   *    |
          F - λG = |--------|--------|, 
                   |   O    | Ai-λEi |
 
where the `nf x nf` subpencil `Af-λEf`, with `Ef` nonsingular and upper triangular, 
contains the finite eigenvalues of the pencil `A-λE` and the `ni x ni` subpencil `Ai-λEi` 
contains the infinite elementary divisors.

The subpencil `Ai-λEi` is in a staircase form, with `Ai` nonsingular and `Ei` nilpotent. 
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

`Q'*B` is returned in `B` unless `B = missing` and `C*Z` is returned in `C` unless `C = missing` .              

"""
function fisplit(A::AbstractMatrix, E::AbstractMatrix, B::Union{AbstractMatrix,Missing}, C::Union{AbstractMatrix,Missing}; 
   fast::Bool = true, finite_infinite::Bool = false, 
   atol1::Real = zero(real(eltype(A))), atol2::Real = zero(real(eltype(E))), 
   rtol::Real = (size(A,1)*eps(real(float(one(eltype(A))))))*iszero(min(atol1,atol2)), 
   withQ::Bool = true, withZ::Bool = true)

   n = LinearAlgebra.checksquare(A)
   n == LinearAlgebra.checksquare(E) || throw(DimensionMismatch("A and E must have the same dimensions"))          
   (!ismissing(B) && n !== size(B,1)) && throw(DimensionMismatch("A and B must have the same number of rows"))
   (!ismissing(C) && n !== size(C,2)) && throw(DimensionMismatch("A and C must have the same number of columns"))
   isa(A,Adjoint) && (A = copy(A))
   isa(E,Adjoint) && (E = copy(E))
   ismissing(B) || (isa(B,Adjoint) && (B = copy(B)))
   ismissing(C) || (isa(C,Adjoint) && (C = copy(C)))
   T = promote_type(eltype(A), eltype(E))
   ismissing(B) || (T = promote_type(T,eltype(B)))
   ismissing(C) || (T = promote_type(T,eltype(C)))
   T <: BlasFloat || (T = promote_type(Float64,T))
   eltype(A) == T || (A = convert(Matrix{T},A))
   eltype(E) == T || (E = convert(Matrix{T},E))
   ismissing(B) || (eltype(B) == T || (B = convert(Matrix{T},B)))
   ismissing(C) || (eltype(C) == T || (C = convert(Matrix{T},C)))

   withQ ? (Q = Matrix{T}(I,n,n)) : (Q = nothing)
   withZ ? (Z = Matrix{T}(I,n,n)) : (Z = nothing)

   # Step 0: Reduce to the standard form
   n1, m1, p1 = _preduceBF!(A, E, Q, Z, B, C; atol = atol2, rtol = rtol, fast = fast) 

   ν = Vector{Int}(undef,n)
        
   # fast returns for null dimensions
   if n == 0 
      return A, E, B, C, Q, Z, ν, 0, 0
   end

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
      return A, E, B, C, Q, Z, reverse(ν[1:i]), nf, ni                                             
   else

      # Reduce A-λE to the to the Kronecker-like form by splitting the infinite-finite structures
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
   
      return A, E, B, C, Q, Z, ν[1:i], nf, ni                                             
   end
end
