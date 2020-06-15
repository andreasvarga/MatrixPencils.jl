"""
    poly2pm(PM; grade = k) -> P

Build a grade `k` matrix polynomial representation `P(λ)` from a polynomial matrix, polynomial vector or scalar polynomial `PM(λ)`. 

`PM(λ)` is a matrix, vector or scalar of elements of the `Polynomial` type 
provided by the [Polynomials](https://github.com/JuliaMath/Polynomials.jl) package.   

`P(λ)` is a grade `k` polynomial matrix of the form `P(λ) = P_1 + λ P_2 + ... + λ**k P_(k+1)`, for which 
the coefficient matrices `P_l`, `l = 1, ..., k+1`, are stored in the 3-dimensional matrix `P`, 
where `P[:,:,l]` contains the `l`-th coefficient matrix `P_l` (multiplying `λ**(l-1)`). 
If `grade = missing`, then `k` is chosen the largest degree of the elements of `PM`.
The coefficients of the degree `d` element `(i,j)` of `PM(λ)` result in `P[i,j,1:d+1]`.
"""
function poly2pm(PM::Matrix{Polynomial{T}}; grade::Union{Int,Missing} = missing) where T
   p, m = size(PM)
   degs = degree.(PM)
   d = maximum(degs)
   ismissing(grade) ? k = d+1 : k = max(d,grade)+1
   k == 0 && (return zeros(T,p,m,1))
   P = zeros(T,p,m,k)
   for j = 1:m
      for i = 1:p
         degs[i,j] < 0 || (P[i,j,1:degs[i,j]+1] = coeffs(PM[i,j])) 
      end
   end
   return P      
end
function poly2pm(PM::Matrix{T}; grade::Union{Int,Missing} = missing) where T
   p, m = size(PM)
   d = 0
   ismissing(grade) ? k = d+1 : k = max(d,grade)+1
   k == 0 && (return zeros(T,p,m,1))
   if k == 1
      P = reshape(PM,p,m,1)
   else
      P = zeros(T,p,m,k)
      P[:,:,1] = PM
   end
   return P      
end
function poly2pm(PM::Vector{Polynomial{T}}; grade::Union{Int,Missing} = missing) where T
   m = length(PM)
   degs = degree.(PM)
   d = maximum(degs)
   ismissing(grade) ? k = d+1 : k = max(d,grade)+1
   k == 0 && (return zeros(T,m,1,1))
   P = zeros(T,m,1,k)
   for i = 1:m
      degs[i] < 0 || (P[i,1,1:degs[i]+1] = coeffs(PM[i])) 
   end
   return P      
end
function poly2pm(PM::Vector{T}; grade::Union{Int,Missing} = missing) where T
   m = length(PM)
   d = 0
   ismissing(grade) ? k = d+1 : k = max(d,grade)+1
   k == 0 && (return zeros(T,m,1,1))
   if k == 1
      P = reshape(PM,m,1,1)
   else
      P = zeros(T,m,1,k)
      P[:,:,1] = PM
   end
   return P      
end
function poly2pm(PM::Union{Adjoint{T,Vector{T}},Transpose{T,Vector{T}}}; grade::Union{Int,Missing} = missing) where T
   m = length(PM)
   d = 0
   ismissing(grade) ? k = d+1 : k = max(d,grade)+1
   k == 0 && (return zeros(T,1,m,1))
   adj = typeof(PM) <: Adjoint
   if k == 1
      adj ? P = reshape(conj(PM.parent),1,m,1) : P = reshape(PM.parent,1,m,1)
   else
      P = zeros(T,1,m,k)
      adj ? P[:,:,1] = reshape(conj(PM.parent),1,m,1) : P[:,:,1] = reshape(PM.parent,1,m,1)
   end
   return P      
end
function poly2pm(PM::Polynomial{T}; grade::Union{Int,Missing} = missing) where T
   d = degree(PM)
   ismissing(grade) ? k = d+1 : k = max(d,grade)+1
   k == 0 && (return zeros(T,1,1,1))
   P = zeros(T,1,1,k)
   d < 0 || (P[1,1,1:d+1] = coeffs(PM))
   return P      
end
function poly2pm(PM::Number; grade::Union{Int,Missing} = missing)
   T = typeof(PM)
   PM == zero(T) ? d = -1 : d = 0
   ismissing(grade) ? k = d+1 : k = max(d,grade)+1
   k == 0 && (return zeros(T,1,1,1))
   P = zeros(T,1,1,k)
   d < 0 || (P[1,1,1] = PM)
   return P      
end
"""
    pm2poly(P[,var = 'x']) -> PM

Build the polynomial matrix `PM(λ)` from its matrix polynomial representation `P(λ)`. 

`P(λ)` is a grade `k` polynomial matrix of the form `P(λ) = P_1 + λ P_2 + ... + λ**k P_(k+1)`, for which 
the coefficient matrices `P_l`, `l = 1, ..., k+1`, are stored in the 3-dimensional matrix `P`, 
where `P[:,:,l]` contains the `l`-th coefficient matrix `P_l` (multiplying `λ**(l-1)`). 

`PM(λ)` is a matrix of elements of the `Polynomial` type 
provided by the [Polynomials](https://github.com/JuliaMath/Polynomials.jl) package. 
The element `(i,j)` of `PM(λ)` is built from the coefficients contained in `P[i,j,1:k+1]`.
The symbol to be used for the indeterminate `λ` can be specified in the optional input variable `var`. 
"""
function pm2poly(PM::AbstractArray{T,3},var::Union{Char, AbstractString, Symbol}='x') where T
   m, n, k = size(PM)
   P = zeros(Polynomial{T},m,n)
   for i = 1:m
      for j = 1:n
          P[i,j] = Polynomial(PM[i,j,1:k],var)
      end
   end
   return P      
end
"""
    pmdeg(P) -> deg

Determine the degree `deg` of a polynomial matrix `P(λ)`. 

`P(λ)` is a grade `k` polynomial matrix of the form `P(λ) = P_1 + λ P_2 + ... + λ**k P_(k+1)`, for which 
the coefficient matrices `P_i`, `i = 1, ..., k+1`, are stored in the 3-dimensional matrix `P`, 
where `P[:,:,i]` contains the `i`-th coefficient matrix `P_i` (multiplying `λ**(i-1)`). 
The degree of `P(λ)` is `deg = j-1`, where `j` is the largest index for which `P[:,:,j]` is nonzero. The degree of   
the zero polynomial matrix is defined to be `deg = -1`.

`P(λ)` can also be specified as a matrix, vector or scalar of elements of the `Polynomial` type 
provided by the [Polynomials](https://github.com/JuliaMath/Polynomials.jl) package.   
The degree of `P(λ)` is the largest degree of the elements of `P(λ)`. 
The degree of the zero polynomial matrix is defined to be `-1`.
"""
function pmdeg(P::AbstractArray{T,3}) where T
   for j = size(P,3):-1:1
       norm(P[:,:,j],Inf) > 0 && return j-1
   end
   return -1
end
function pmdeg(P::Union{AbstractVecOrMat{Polynomial{T}},Polynomial{T},Number}) where T
   typeof(P) <: Number && (P == 0 ? (return -1) : (return 0) )
   return maximum(degree.(P))
end
"""
    pmeval(P,val) -> R

Evaluate `R = P(val)` for a polynomial matrix `P(λ)`, using Horner's scheme. 

`P(λ)` is a grade `k` polynomial matrix of the form `P(λ) = P_1 + λ P_2 + ... + λ**k P_(k+1)`, for which 
the coefficient matrices `P_i`, `i = 1, ..., k+1`, are stored in the 3-dimensional matrix `P`, 
where `P[:,:,i]` contains the `i`-th coefficient matrix `P_i` (multiplying `λ**(i-1)`). 

`P(λ)` can also be specified as a matrix, vector or scalar of elements of the `Polynomial` type 
provided by the [Polynomials](https://github.com/JuliaMath/Polynomials.jl) package.   
"""
function pmeval(P::AbstractArray{T,3},val::Number) where {T}
   # Horner's scheme
   p, m, k1 = size(P)
   nd = pmdeg(P)+1
   S = typeof(val)
   nd == 0 && return zeros(promote_type(T,S),p,m)
   R = P[:,:,nd]*one(S)
   for k = nd-1:-1:1
      R = R*val+ P[:,:,k]
   end
   return R
end
pmeval(P::Union{AbstractVecOrMat{Polynomial{T}},Polynomial{T},Number,AbstractVecOrMat},val::Number) where T =
      pmeval(poly2pm(P),val::Number)
"""
    pmreverse(P[,j]) -> Q

Build `Q(λ) = λ^j*P(1/λ)`, the `j`-reversal of a polynomial matrix `P(λ)` for `j ≥ deg(P(λ))`. 
If `j` is not specified, the default value `j = deg(P(λ))` is used. 

`P(λ)` can be specified as a grade `k` polynomial matrix of the form `P(λ) = P_1 + λ P_2 + ... + λ**k P_(k+1)`, 
for which the coefficient matrices `P_i`, `i = 1, ..., k+1`, are stored in the 3-dimensional matrix `P`, 
where `P[:,:,i]` contains the `i`-th coefficient matrix `P_i` (multiplying `λ**(i-1)`). 

`P(λ)` can also be specified as a matrix, vector or scalar of elements of the `Polynomial` type 
provided by the [Polynomials](https://github.com/JuliaMath/Polynomials.jl) package.   

If deg(P(λ)), then `Q(λ)` is a grade `j` polynomial matrix of the form 
`Q(λ) = Q_1 + λ Q_2 + ... + λ**j Q_(j+1)`, for which 
the coefficient matrices `Q_i`, `i = 1, ..., j+1`, are stored in the 3-dimensional matrix `Q`, 
where `Q[:,:,i]` contains the `i`-th coefficient matrix `Q_i` (multiplying `λ**(i-1)`). 
The coefficient matrix `Q_i` is either `0` if `i ≤ j-d` or `Q_(j-d+i) = P_(d-i+1)` for `i = 1, ..., d`.
"""
function pmreverse(P::AbstractArray{T,3}, j::Int = pmdeg(P)) where T
   d = pmdeg(P)
   j < d && error("j must be at least $d")
   m, n, k1 = size(P)
   Q = zeros(eltype(P),m,n,j+1)
   Q[:,:,j-d+1:j+1] = reverse(P[:,:,1:d+1],dims=3)
   return Q
end
pmreverse(P::Union{AbstractVecOrMat{Polynomial{T}},Polynomial{T},Number}; kwargs...) where T =
      pmreverse(poly2pm(P); kwargs...)
"""
     pm2lpCF1(P; grade = l) -> (M, N)

Build a strong linearization `M - λN` of a polynomial matrix `P(λ)` in the first companion Frobenius form. 

`P(λ)` is a grade `k` polynomial matrix assumed of the form `P(λ) = P_1 + λ P_2 + ... + λ**k P_(k+1)`, with 
the coefficient matrices `P_i`, `i = 1, ..., k+1` stored in the 3-dimensional matrix `P`, 
where `P[:,:,i]` contains the `i`-th coefficient matrix `P_i` (multiplying `λ**(i-1)`). 
The effective grade `l` to be used for linearization can be specified via the keyword argument `grade` as 
`grade = l`, where `l` must be chosen equal to or greater than the degree of `P(λ)`.
The default value used for `l` is `l = deg(P(λ))`.

`P(λ)` can also be specified as a matrix, vector or scalar of elements of the `Polynomial` type 
provided by the [Polynomials](https://github.com/JuliaMath/Polynomials.jl) package.   

If `P(λ)` is a `m x n` polynomial matrix of effective grade `l` and degree `d`, 
then the resulting matrix pencil `M - λN` satisfies the following conditions [1]:

(1) `M - λN` has dimension `(m+n*(l-1)) x n*l` and `M - λN` is regular if `P(λ)` is regular;

(2) `M - λN` and `P(λ)` have the same finite eigenvalues;

(3) the partial multiplicities of infinite eigenvalues of `M - λN` are in excess with `l-d` to the
partial multiplicities of the infinite eigenvalues of `P(λ)`;

(4) `M - λN` and `P(λ)` have the same number of right Kronecker indices and the right
Kronecker indices of `M - λN` are in excess with `l-1` to the right Kronecker indices of `P(λ)`;

(5) `M - λN` and `P(λ)` have the same left Kronecker structure (i.e., the same left Kronecker indices).

[1] F. De Terán, F. M. Dopico, D. S. Mackey, Spectral equivalence of polynomial matrices and
the Index Sum Theorem, Linear Algebra and Its Applications, vol. 459, pp. 264-333, 2014.
"""
function pm2lpCF1(P::AbstractArray{T,3}; grade::Int = pmdeg(P)) where T
   m, n, k1 = size(P)
   deg = pmdeg(P)
   grade < deg && error("The selected grade must be at least $deg")
   grade == -1 && (return zeros(T,m,n), nothing )
   grade == 0 && (return P[:,:,1], nothing )
   grade == 1 && (grade > deg ? (return P[:,:,1], zeros(T,m,n)) : (return P[:,:,1], -P[:,:,2]) )
   nd = n*grade
   nd1 = nd-n
   grade > deg ? (N = [ zeros(T,nd1+m,n)  [zeros(T,m,nd1); I] ]) : 
                 (N = [ [P[:,:,grade+1]; zeros(T,nd1,n)]  [zeros(T,m,nd1); I] ])
   M = [ zeros(T,m,nd);  [I zeros(T,nd1,n)] ]
   deg == -1 && (return M, N)
   k = nd
   it = 1:m
   grade > deg ? ne = deg+1 : ne = max(1,deg)
   for i = 1:ne
      M[it,k-n+1:k] = -P[:,:,i]
      k -= n
   end
   return M, N
end
pm2lpCF1(P::Union{AbstractVecOrMat{Polynomial{T}},Polynomial{T},Number}; kwargs...) where {T} =
       pm2lpCF1(poly2pm(P); kwargs...)
"""
    pm2lpCF2(P; grade = l) -> (M, N)

Build a strong linearization `M - λN` of a polynomial matrix `P(λ)` in the second companion Frobenius form. 

`P(λ)` is a grade `k` polynomial matrix assumed of the form `P(λ) = P_1 + λ P_2 + ... + λ**k P_(k+1)`, with 
the coefficient matrices `P_i`, `i = 1, ..., k+1` stored in the 3-dimensional matrix `P`, 
where `P[:,:,i]` contains the `i`-th coefficient matrix `P_i` (multiplying `λ**(i-1)`).
The effective grade `l` to be used for linearization can be specified via the keyword argument `grade` as 
`grade = l`, where `l` must be chosen equal to or greater than the degree of `P(λ)`. 
The default value used for `l` is `l = deg(P(λ))`.

`P(λ)` can also be specified as a matrix, vector or scalar of elements of the `Polynomial` type 
provided by the [Polynomials](https://github.com/JuliaMath/Polynomials.jl) package.   

If `P(λ)` is a `m x n` polynomial matrix of effective grade `l` and degree `d`, then the resulting matrix pencil 
`M - λN` satisfies the following conditions [1]:

(1) `M - λN` has dimension `l*m x (n+(l-1)*m)` and `M - λN` is regular if `P(λ)` is regular;

(2) `M - λN` and `P(λ)` have the same finite eigenvalues;

(3) the partial multiplicities of infinite eigenvalues of `M - λN` are in excess with `l-d` to the
partial multiplicities of the infinite eigenvalues of `P(λ)`;

(4) `M - λN` and `P(λ)` have the same right Kronecker structure (i.e., the same right Kronecker indices);

(5) `M - λN` and `P(λ)` have the same number of left Kronecker indices and the left
Kronecker indices of `M - λN` are in excess with `l-1` to the left Kronecker indices of `P(λ)`.

[1] F. De Terán, F. M. Dopico, D. S. Mackey, Spectral equivalence of polynomial matrices and
the Index Sum Theorem, Linear Algebra and Its Applications, vol. 459, pp. 264-333, 2014.
"""
function pm2lpCF2(P::AbstractArray{T,3}; grade::Int = pmdeg(P)) where T
   m, n, k1 = size(P)
   deg = pmdeg(P)
   grade < deg && error("The selected grade must be at least $deg")
   grade == -1 && (return zeros(T,m,n), nothing )
   grade == 0 && (return P[:,:,1], nothing  )
   grade == 1 && (grade > deg ? (return P[:,:,1], zeros(T,m,n)) : (return P[:,:,1], -P[:,:,2]) )
   md = m*grade
   md1 = md-m
   grade > deg ? (N = [ zeros(T,md,n)  [zeros(T,m,md1); I] ]) : 
                 (N = [ [P[:,:,grade+1]; zeros(T,md1,n)]  [zeros(T,m,md1); I] ])
   M = [ zeros(T,md,n)  [I; zeros(T,m,md1)] ]
   deg == -1 && (return M, N)
   k = md
   it = 1:n
   grade > deg ? ne = deg+1 : ne = max(1,deg)
   for i = 1:ne
       M[k-m+1:k,it] = -P[:,:,i]
       k -= m
   end
   return M, N
end
pm2lpCF2(P::Union{AbstractVecOrMat{Polynomial{T}},Polynomial{T},Number}; kwargs...) where {T} =
       pm2lpCF2(poly2pm(P); kwargs...)
"""
     pm2ls(P; contr = false, obs = false, noseig = false, minimal = false,
              fast = true, atol = 0, rtol) -> (A, E, B, C, D)

Build a structured linearization 

              | A-λE | B | 
     M - λN = |------|---|
              |  C   | D |  
      
of a polynomial matrix `P(λ)` which preserves a part of the Kronecker structure of `P(λ)`. 

`P(λ)` can be specified as a grade `k` polynomial matrix of the form `P(λ) = P_1 + λ P_2 + ... + λ**k P_(k+1)`, 
for which the coefficient matrices `P_i`, `i = 1, ..., k+1`, are stored in the 3-dimensional matrix `P`, 
where `P[:,:,i]` contains the `i`-th coefficient matrix `P_i` (multiplying `λ**(i-1)`). 

`P(λ)` can also be specified as a matrix, vector or scalar of elements of the `Polynomial` type 
provided by the [Polynomials](https://github.com/JuliaMath/Polynomials.jl) package.   

If `d` is the degree of `P(λ)` and `n` is the order of `A-λE`, then the computed linearization satisfies:

(1) `A-λE` is regular and `P(λ) = C*inv(λE-A)*B+D`;

(2) `rank[B A-λE] = n` (controllability) if `minimal = true` or `contr = true`, in which case 
the finite and right Kronecker structures are preserved;

(3) `rank[A-λE; C] = n` (observability)  if `minimal = true` or `contr = true`, in which case 
the finite and left Kronecker structures are preserved;

(4) `A-λE` has no non-dynamic modes if `minimal = true` or `noseig = true`. 

If conditions (1)-(4) are satisfied, the linearization is called `minimal` and the resulting order `n`
is the least achievable order. If conditions (1)-(3) are satisfied, the linearization is called `irreducible` 
and the resulting order `n` is the least achievable order using orthogonal similarity transformations.

The underlying pencil manipulation algorithms [1] and [2] to compute reduced order linearizations 
employ rank determinations based on either the use of 
rank revealing QR-decomposition with column pivoting, if `fast = true`, or the SVD-decomposition, if `fast = false`.
The rank decision based on the SVD-decomposition is generally more reliable, but the involved computational effort is higher.

The keyword arguments `atol` and `rtol`, specify the absolute and relative tolerances for the 
nonzero coefficients of `P(λ)`, respectively.

[1] P. Van Dooreen, The generalized eigenstructure problem in linear system theory, 
IEEE Transactions on Automatic Control, vol. AC-26, pp. 111-129, 1981.

[2] A. Varga, Solving Fault Diagnosis Problems - Linear Synthesis Techniques, Springer Verlag, 2017. 
"""
function pm2ls(P::AbstractArray{T,3}; minimal::Bool = false, contr::Bool = false, obs::Bool = false, noseig::Bool = false, 
               fast::Bool = true, atol::Real = zero(real(T)), 
               rtol::Real = (min(size(P)...)*eps(real(float(one(T)))))*iszero(atol)) where T
   minimal && (contr = true; obs = true; noseig = true)
   p, m, k1 = size(P)
   nd = pmdeg(P)+1
   nd == 0 && (return zeros(T,0,0), zeros(T,0,0), zeros(T,0,m), zeros(T,p,0), zeros(T,p,m))
   D = P[:,:,1]
   nd == 1 && (return zeros(T,0,0), zeros(T,0,0), zeros(T,0,m), zeros(T,p,0), D)
   if xor(contr,obs)
      if obs
         # build an observable linearization
         n = p*nd
         E = [zeros(T,n,p) [I; zeros(T,p,p*(nd-1))]]
         B = zeros(T,n,m)
         k = p
         for i = 2:nd
             B[k+1:k+p,:] = P[:,:,i]
             k += p
         end
         C = [ -I  zeros(T,p,p*(nd-1))  ]
      else
         # build a controllable linearization
         n = m*nd
         E = [zeros(T,n,m) [I; zeros(T,m,m*(nd-1))]]
         B = [zeros(T,m*(nd-1),m); -I ]
         C = zeros(T,p,n)
         k = 0
         for i = 1:nd-1
             C[:,k+1:k+m] = P[:,:,nd-i+1]
             k += m
         end
      end
      A = Matrix{T}(I,n,n)
   else
      if p <= m
         n = p*nd
         E = [zeros(T,n,p) [I; zeros(T,p,p*(nd-1))]]
         B = zeros(T,n,m)
         k = p
         for i = 2:nd
             B[k+1:k+p,:] = P[:,:,i]
             k += p
         end
         C = [ -I  zeros(T,p,p*(nd-1))  ]
         if contr
            # remove uncontrollable part
            T <: BlasFloat ? T1 = T : T1 = promote_type(Float64,T)        
            Er = copy_oftype(E,T1)
            Br = copy_oftype(B,T1)
            Cr = copy_oftype(C,T1)
            _, _, nr, nuc = sklf_right!(Er, Br, Cr; fast = fast, atol1 = atol, atol2 = atol, rtol = rtol, withQ = false) 
            if nuc > 0
               ir = 1:nr
               # save intermediary results
               E = Er[ir,ir]
               B = Br[ir,:]
               C = Cr[:,ir]
            end
            A = Matrix{T1}(I,nr,nr)
         else
            A = Matrix{T}(I,n,n)
         end
      else
         # build a controllable linearization
         n = m*nd
         E = [zeros(T,n,m) [I; zeros(T,m,m*(nd-1))]]
         B = [zeros(T,m*(nd-1),m); -I ]
         C = zeros(T,p,n)
         k = 0
         for i = 1:nd-1
             C[:,k+1:k+m] = P[:,:,nd-i+1]
             k += m
         end
         if obs
            # remove unobservable part
            T <: BlasFloat ? T1 = T : T1 = promote_type(Float64,T)        
            Er = copy_oftype(E,T1)
            Br = copy_oftype(B,T1)
            Cr = copy_oftype(C,T1)
            _, _, nr, nuo = sklf_left!(Er, Cr, Br; fast = fast, atol1 = atol, atol2 = atol, rtol = rtol, withQ = false) 
            if nuo > 0
               ir = n-nr+1:n
               # save intermediary results
               E = Er[ir,ir]
               B = Br[ir,:]
               C = Cr[:,ir]
            end
            A = Matrix{T1}(I,nr,nr)
         else
            A = Matrix{T}(I,n,n)
         end
      end
   end        
   if noseig
      A, E, B, C, D  = lsminreal(A,E,B,C,D,contr = false, obs = false, fast = fast, atol1 = atol, atol2 = atol, rtol = rtol)
   end
   return A, E, B, C, D
end
pm2ls(P::Union{AbstractVecOrMat{Polynomial{T}},Polynomial{T},Number}; kwargs...) where {T} =
         pm2ls(poly2pm(P); kwargs...)
"""
     pm2lps(P; contr = false, obs = false) -> (A, E, B, F, C, G, D, H)

Build a structured linearization  

              | A-λE | B-λF | 
     M - λN = |------|------|
              | C-λG | D-λH |  
      
of a polynomial matrix `P(λ)` which preserves a part of the Kronecker structure of `P(λ)`. 

`P(λ)` can be specified as a grade `k` polynomial matrix of the form `P(λ) = P_1 + λ P_2 + ... + λ**k P_(k+1)`, 
for which the coefficient matrices `P_i`, `i = 1, ..., k+1`, are stored in the 3-dimensional matrix `P`, 
where `P[:,:,i]` contains the `i`-th coefficient matrix `P_i` (multiplying `λ**(i-1)`). 

`P(λ)` can also be specified as a matrix, vector or scalar of elements of the `Polynomial` type 
provided by the [Polynomials](https://github.com/JuliaMath/Polynomials.jl) package.   

If `d` is the degree of the `p x m` polynomial matrix `P(λ)`, then the computed linearization satisfies:

(1) `A-λE` is a `n x n` regular pencil, where `n = p(d-1)` if `contr = false` and `p <= m`
and `n = m(d-1)` otherwise; 

(2) `P(λ) = (C-λG)*inv(λE-A)*(B-λF)+D-λH`;

(3) `rank[B-λF A-λE] = n` for any finite and infinite `λ` (strong controllability) if `contr = true`, in which case 
the finite and right Kronecker structures are preserved;

(4) `rank[A-λE; C-λG] = n` for any finite and infinite `λ` (strong observability)  if `obs = true`, in which case 
the finite and left Kronecker structures are preserved. 
"""
function pm2lps(P::AbstractArray{T,3}; contr::Bool = false, obs::Bool = false) where T
   p, m, k1 = size(P)
   d = pmdeg(P)
   nd = d+1
   H = zeros(T,p,m)
   d == -1 && (return zeros(T,0,0), zeros(T,0,0), zeros(T,0,m), zeros(T,0,m), zeros(T,p,0), zeros(T,p,0), H, H)
   D = P[:,:,1]
   d == 0 && (return zeros(T,0,0), zeros(T,0,0), zeros(T,0,m), zeros(T,0,m), zeros(T,p,0), zeros(T,p,0), D, H)
   d == 1 && (return zeros(T,0,0), zeros(T,0,0), zeros(T,0,m), zeros(T,0,m), zeros(T,p,0), zeros(T,p,0), D, -P[:,:,2])
   if obs || (!contr && p <= m)
      # build a strongly observable linearization
      n = p*(d-1)
      E = [zeros(T,n,p) [I; zeros(T,p,n-p)]]
      B = zeros(T,n,m)
      F = zeros(T,n,m)
      k = 0
      for i = 2:d
          B[k+1:k+p,:] = P[:,:,i]
          k += p
      end
      F[n-p+1:n,:] = -P[:,:,nd]
      C = zeros(T,p,n)
      G = [I zeros(T,p,n-p) ]
      A = Matrix{T}(I,n,n)
   else
     # build a strongly controllable linearization
     n = m*(d-1)
     E = [zeros(T,n,m) [I; zeros(T,m,n-m)]]
     B = zeros(T,n,m)
     F = [zeros(T,n-m,m); I]
     C = zeros(T,p,n)
     G = [ -P[:,:,nd] zeros(T,p,n-m)]
     k = n
     for i = 2:d
         #C[:,k-m+1:k] = P[:,:,nd-i]
         C[:,k-m+1:k] = P[:,:,i]
         k -= m
     end
     A = Matrix{T}(I,n,n)
   end
   return A, E, B, F, C, G, D, H
end
pm2lps(P::Union{AbstractVecOrMat{Polynomial{T}},Polynomial{T},Number}; kwargs...) where {T} =
         pm2lps(poly2pm(P); kwargs...)
"""
    ls2pm(A, E, B, C, D; fast = true, atol1 = 0, atol2 = 0, gaintol = 0, rtol = min(atol1,atol2) > 0 ? 0 : n*ϵ, val) -> P

Build the polynomial matrix `P(λ) = C*inv(λE-A)*B+D` corresponding to its structured linearization 

     | A-λE | B | 
     |------|---|
     |  C   | D |  

by explicitly determining for each polynomial entry, its coefficients from its roots and corresponding gain. 

The keyword arguments `atol1` and `atol2` specify the absolute tolerances for the elements of `A`, `B`, `C`, `D`, and,  
respectively, of `E`, and `rtol` specifies the relative tolerances for the nonzero elements of `A`, `B`, `C`, `D` and `E`.
The default relative tolerance is `(n+1)*ϵ`, where `n` is the size of the size dimension of `A`, and `ϵ` is the 
machine epsilon of the element type of coefficients of `A`. 

The keyword argument `gaintol` specifies the threshold for the magnitude of the nonzero elements of the gain matrix 
`C*inv(γE-A)*B+D`, where `γ = val` if `val` is a number or `γ` is a randomly chosen complex value of unit magnitude, 
if `val = missing`. Generally, `val` should not be a root of any of entries of `P`.
"""
function ls2pm(A::AbstractMatrix, E::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix; 
                fast::Bool = true, atol1::Real = zero(real(eltype(A))), atol2::Real = zero(real(eltype(E))), 
                gaintol::Real = zero(real(eltype(A))), val::Union{Number,Missing} = missing,
                rtol::Real = ((size(A,1)+1)*eps(real(float(one(eltype(A))))))*iszero(min(atol1,atol2))) 
   n = LinearAlgebra.checksquare(A)
   (n,n) == size(E) || throw(DimensionMismatch("A and E must have the same dimensions"))
   p, m = size(D)
   (n,m) == size(B) || throw(DimensionMismatch("A, B and D must have compatible dimensions"))
   (p,n) == size(C) || throw(DimensionMismatch("A, C and D must have compatible dimensions"))

   T = promote_type(eltype(A),eltype(E), eltype(B),eltype(C),eltype(D))
   T <: BlasFloat || (T = promote_type(Float64,T))
   compl = T <: Complex

   P = zeros(T,p,m,n+1) 
   ismissing(val) && (val = exp(rand()*im))
   A1, E1, B1, C1, D1 = lsminreal2(A, E, B, C, D, infinite = false, noseig = false, atol1 = atol1, atol2 = atol2)
   Pval = lseval(A1, E1, B1, C1, D1, val)
   isunimodular(A1, E1, atol1 = atol1, atol2 = atol2, rtol = rtol) || 
                error("The given linearization cannot be converted to a polynomial form")  
   for i = 1:p
       for j = 1:m
           if abs(Pval[i,j]) > gaintol
              zer, iz, = spzeros(A1, E1, B1[:,j:j], C1[i:i,:], D1[i:i,j:j]; 
                                 fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol) 
              c, pval = poly_coeffval(zer[1:(length(zer)-sum(iz))],val)
              P[i,j,1:length(c)] = compl ? c*(Pval[i,j]/pval) : real(c*(Pval[i,j]/pval)) 
           end
       end
   end
   return P[:,:,1:pmdeg(P)+1]
end
"""
    lps2pm(A, E, B, F, C, G, D, H; fast = true, atol1 = 0, atol2 = 0, gaintol = 0, rtol = min(atol1,atol2) > 0 ? 0 : n*ϵ, val) -> P

Build the polynomial matrix `P(λ) = (C-λG)*inv(λE-A)*(B-λF)+D-λH` corresponding to its structured linearization 

     | A-λE | B-λF | 
     |------|------|
     | C-λG | D-λH |  

by explicitly determining for each polynomial entry, its coefficients from its roots and corresponding gain. 

The keyword arguments `atol1` and `atol2` specify the absolute tolerances for the elements of `A`, `B`, `C`, `D`, and of  
`E`, `F`, `G`, `H`, respectively,  and `rtol` specifies the relative tolerances for the nonzero elements of 
`A`, `B`, `C`, `D`, `E`, F`, `G`, `H`. 
The default relative tolerance is `(n+2)*ϵ`, where `n` is the size of the size dimension of `A`, and `ϵ` is the 
machine epsilon of the element type of coefficients of `A`. 

The keyword argument `gaintol` specifies the threshold for the magnitude of the nonzero elements of the gain matrix 
`C*inv(γE-A)*B+D`, where `γ = val` if `val` is a number or `γ` is a randomly chosen complex value of unit magnitude, 
if `val = missing`. Generally, `val` should not be a root of any of entries of `P`. 
"""
function lps2pm(A::AbstractMatrix, E::AbstractMatrix,B::AbstractMatrix, F::AbstractMatrix, 
   C::AbstractMatrix, G::AbstractMatrix, D::AbstractMatrix, H::AbstractMatrix; 
   fast::Bool = true, atol1::Real = zero(real(eltype(A))), atol2::Real = zero(real(eltype(E))), 
   gaintol::Real = zero(real(eltype(A))), val::Union{Number,Missing} = missing,
   rtol::Real = ((size(A,1)+2)*eps(real(float(one(eltype(A))))))*iszero(min(atol1,atol2))) 
   n = LinearAlgebra.checksquare(A)
   (n,n) == size(E) || throw(DimensionMismatch("A and E must have the same dimensions"))
   p, m = size(D)
   (n,m) == size(B) || throw(DimensionMismatch("A, B and D must have compatible dimensions"))
   (p,n) == size(C) || throw(DimensionMismatch("A, C and D must have compatible dimensions"))
   (n,m) == size(F) || throw(DimensionMismatch("B and F must have the same dimensions"))
   (p,n) == size(G) || throw(DimensionMismatch("C and G must have the same dimensions"))
   (p,m) == size(H) || throw(DimensionMismatch("D and H must have the same dimensions"))
   T = promote_type(eltype(A), eltype(B), eltype(C), eltype(D), eltype(E), eltype(F), eltype(G), eltype(H))
   T <: BlasFloat || (T = promote_type(Float64,T))        

   compl = T <: Complex
   P = zeros(T,p,m,n+2) 
   ismissing(val) && (val = exp(rand()*im))
   A1, E1, B1, F1, C1, G1, D1, H1, V1, W1 = lpsminreal(A, E, B, F, C, G, D, H, atol1 = atol1, atol2 = atol2, rtol = rtol)
   Pval = V1'\lpseval(A1, E1, B1, F1, C1, G1, D1, H1, val)/W1
   isunimodular(A1, E1, atol1 = atol1, atol2 = atol2, rtol = rtol) || 
                error("The given linearization cannot be converted to a polynomial form")  
   M1 = [A1 B1/W1; V1'\C1 V1'\D1/W1]
   N1 = [E1 F1/W1; V1'\G1 V1'\H1/W1]
   n = size(A1,1)
   indi = [1:n;1]
   indj = [1:n;1]
   n1 = n+1
   for i = 1:p
      indi[n1] = n+i
      for j = 1:m
         if abs(Pval[i,j]) > gaintol
            indj[n1] = n+j
            zer, iz, = pzeros(view(M1,indi,indj),view(N1,indi,indj); 
                              fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol) 
            # zer, iz, = pzeros(M1[indi,indj],N1[indi,indj]; 
            #                   fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol) 
            c, pval = poly_coeffval(zer[1:(length(zer)-sum(iz))],val)
            P[i,j,1:length(c)] = compl ? c*(Pval[i,j]/pval) : real(c*(Pval[i,j]/pval)) 
         end
      end
   end
return P[:,:,1:pmdeg(P)+1]
end
function poly_coeffval(r::AbstractVector{T},val::Number) where {T}
   # Compute the coefficients of a polynomial from its roots and evaluate the polynomial
   # for a given value of its argument. Both are equal to one for an empty vector r.
   T1 = promote_type(T,eltype(val))
   n = length(r)
   c = zeros(T1, n+1)
   ONE = one(T1)
   c[1] = ONE
   pval = ONE
   for j = 1:n
       pval = pval*(val-r[j])
       for i = j:-1:1
           c[i+1] = c[i+1]-r[j]*c[i]
       end
   end
   return reverse(c), pval
end
"""
     spm2ls(T, U, V, W; fast = true, contr = false, obs = false, minimal = false, atol = 0, rtol) -> (A, E, B, C, D)
            
Build a structured linearization 

              | A-λE | B | 
     M - λN = |------|---|
              |  C   | D |  

of the structured polynomial matrix 

              | -T(λ) | U(λ) |
       P(λ) = |-------|------|
              | V(λ)  | W(λ) |

such that `V(λ)*inv(T(λ))*U(λ)+W(λ) = C*inv(λE-A)*B+D`. The resulting linearization `M - λN` preserves a part, 
if `minimal = false`, or the complete Kronecker structure, if `minimal = true`, of `P(λ)`. In the latter case, 
the order `n` of `A-λE` is the least possible one and `M - λN` is a strong linearization of `P(λ)`.

`T(λ)`, `U(λ)`, `V(λ)`, and `W(λ)` can be specified as polynomial matrices of the form `X(λ) = X_1 + λ X_2 + ... + λ**k X_(k+1)`, 
for `X = T`, `U`, `V`, and `W`, for which the coefficient matrices `X_i`, `i = 1, ..., k+1`, are stored in 
the 3-dimensional matrices `X`, where `X[:,:,i]` contains the `i`-th coefficient matrix `X_i` (multiplying `λ**(i-1)`). 

`T(λ)`, `U(λ)`, `V(λ)`, and `W(λ)` can also be specified as matrices, vectors or scalars of elements of the `Polynomial` type 
provided by the [Polynomials](https://github.com/JuliaMath/Polynomials.jl) package.   

The computed structured linearization satisfies:

(1) `A-λE` is regular;

(2) `rank[B A-λE] = n` (controllability) if `minimal = true` or `contr = true`, in which case 
the finite and right Kronecker structures are preserved;

(3) `rank[A-λE; C] = n` (observability)  if `minimal = true` or `obs = true`, in which case 
the finite and left Kronecker structures are preserved;

(4) `M-λN` has no simple infinite eigenvalues if `minimal = true`, in which case the complete Kronecker structure is preserved. 

The keyword arguments `atol` and `rtol`, specify, respectively, the absolute and relative tolerance for the 
nonzero coefficients of the matrices `T(λ)`, `U(λ)`, `V(λ)` and `W(λ)`. The default relative tolerance is `nt*ϵ`, 
where `nt` is the size of the square matrix `T(λ)` and `ϵ` is the machine epsilon of the element type of its coefficients. 
"""
function spm2ls(T::Union{AbstractArray{T1,3},AbstractArray{T1,2}},U::Union{AbstractArray{T2,3},AbstractArray{T2,2}},
                V::Union{AbstractArray{T3,3},AbstractArray{T3,2}},W::Union{AbstractArray{T4,3},AbstractArray{T4,2}}; 
                contr::Bool = false, obs::Bool = false, minimal::Bool = false, 
                fast::Bool = true, atol::Real = zero(real(T1)), 
                rtol::Real = size(T,1)*eps(real(float(one(T1))))*iszero(atol)) where {T1, T2, T3, T4}
   if ndims(T) == 2 
      n, nt = size(T)
      n == nt || throw(DimensionMismatch("T(λ) must be a square polynomial matrix"))
      n == rank(T, atol=atol, rtol=rtol) || error("T(λ) must be a regular square polynomial matrix")
      ndT = 1
      T = reshape(T,n,nt,ndT)
   else
      n, nt, ndT = size(T)
      n == nt || throw(DimensionMismatch("T(λ) must be a square polynomial matrix"))
      n == pmrank(T, atol=atol, rtol=rtol) || error("T(λ) must be a regular square polynomial matrix")
      ndT = max(pmdeg(T)+1,1)
   end
   if ndims(U) == 2 
      nt, m = size(U)
      ndU = 1
      U = reshape(U,nt,m,ndU)
   else
      nt, m, ndU = size(U)
      ndU = max(pmdeg(U)+1,1)
   end
   n == nt || throw(DimensionMismatch("T(λ) and U(λ) must have the same number of rows"))
   if ndims(V) == 2 
      p, nt = size(V)
      ndV = 1
      V = reshape(V,p,nt,ndV)
   else
      p, nt, ndV = size(V)
      ndV = max(pmdeg(V)+1,1)
   end
   n == nt || throw(DimensionMismatch("T(λ) and V(λ) must have the same number of columns"))
   if ndims(W) == 2 
      pt, mt = size(W)
      ndW = 1
      W = reshape(W,pt, mt,ndW)
   else
      pt, mt, ndW = size(W)
      ndW = max(pmdeg(W)+1,1)
   end
   pt == p || throw(DimensionMismatch("W(λ) and V(λ) must have the same number of rows"))
   mt == m || throw(DimensionMismatch("W(λ) and U(λ) must have the same number of columns"))
   nd = max(ndT,ndU,ndV,ndW)
   TT = promote_type(T1, T2, T3, T4)
   if nd == 1
      if minimal
         Ar,Er,Br,Cr,Dr = lsminreal(T[:,:,1], zeros(TT,n,n), U[:,:,1], V[:,:,1], W[:,:,1], 
                                    contr = false, obs = false, atol1 = atol, atol2 = atol, rtol = rtol)
         return Ar,Er,Br,Cr,Dr
      else
        return T[:,:,1], zeros(TT,n,n), U[:,:,1], V[:,:,1], W[:,:,1]
      end
   end

   # build the compound polynomial matrix [-T(λ) U(λ); V(λ) W(λ)]
   P = zeros(TT,n+p,n+m,nd)
   ia = 1:n
   jb = n+1:n+m
   ic = n+1:n+p
   P[ia,ia,1:ndT] = -T[:,:,1:ndT]
   P[ia,jb,1:ndU] = U[:,:,1:ndU]
   P[ic,ia,1:ndV] = V[:,:,1:ndV]
   P[ic,jb,1:ndW] = W[:,:,1:ndW]

   # build a linearization of the compound polynomial matrix [-T(λ) U(λ); V(λ) W(λ)]
   A, E, B, C, D = pm2ls(P, contr = contr, obs = obs, noseig = false, 
                           fast = fast, atol = atol, rtol = rtol)

   # form the linearization of P(λ) = V(λ)*inv(T(λ))*U(λ)+W(λ)                        
   nr = size(A,1)
   Ar = [A B[:,ia]; C[ia,:] D[ia,ia]]
   Er = [E zeros(TT,nr,n); zeros(TT,n,n+nr)]
   Br = [B[:,jb]; D[ia,jb]]
   Cr = [C[ic,:] D[ic,ia]]
   Dr = D[ic,jb]
   if minimal 
      Ar,Er,Br,Cr,Dr = lsminreal(Ar,Er,Br,Cr,Dr, fast=fast, atol1 = atol, atol2 = atol, rtol = rtol)
   end
   return Ar, Er, Br, Cr, Dr
end
spm2ls(T::Union{AbstractVecOrMat{Polynomial{T1}},Polynomial{T1},Number,AbstractMatrix{T1}}, U::Union{AbstractVecOrMat{Polynomial{T2}},Polynomial{T2},Number,AbstractVecOrMat{T2}}, 
       V::Union{AbstractVecOrMat{Polynomial{T3}},Polynomial{T3},Number,AbstractVecOrMat{T3}}, W::Union{AbstractVecOrMat{Polynomial{T4}},Polynomial{T4},Number,AbstractVecOrMat{T4}}; kwargs...) where {T1, T2, T3, T4} =
       spm2ls(poly2pm(T),poly2pm(U),poly2pm(V),poly2pm(W); kwargs...)
"""
     spm2lps(T, U, V, W; fast = true, contr = false, obs = false, minimal = false, atol = 0, rtol) -> (A, E, B, F, C, G, D, H)

Build a structured linearization 

              | A-λE | B-λF | 
     M - λN = |------|------|
              | C-λG | D-λH |  

of the structured polynomial matrix 

              | -T(λ) | U(λ) |
       P(λ) = |-------|------|
              |  V(λ) | W(λ) |

such that `V(λ)*inv(T(λ))*U(λ)+W(λ) = (C-λG))*inv(λE-A)*(B-λF)+D-λH`. The resulting linearization `M - λN` preserves a part, 
if `minimal = false`, or the complete Kronecker structure, if `minimal = true`, of `P(λ)`. In the latter case, 
the order `n` of `A-λE` is the least possible one and `M - λN` is a strong linearization of `P(λ)`.

`T(λ)`, `U(λ)`, `V(λ)`, and `W(λ)` can be specified as polynomial matrices of the form `X(λ) = X_1 + λ X_2 + ... + λ**k X_(k+1)`, 
for `X = T`, `U`, `V`, and `W`, for which the coefficient matrices `X_i`, `i = 1, ..., k+1`, are stored in 
the 3-dimensional matrices `X`, where `X[:,:,i]` contains the `i`-th coefficient matrix `X_i` (multiplying `λ**(i-1)`). 

`T(λ)`, `U(λ)`, `V(λ)`, and `W(λ)` can also be specified as matrices, vectors or scalars of elements of the `Polynomial` type 
provided by the [Polynomials](https://github.com/JuliaMath/Polynomials.jl) package.   

The computed structured linearization satisfies:

(1) `A-λE` is regular;

(2) `rank[B-λF A-λE] = n` (strong controllability) if `minimal = true` or `contr = true`, in which case 
the finite and right Kronecker structures are preserved;

(3) `rank[A-λE; C-λG] = n` (strong observability)  if `minimal = true` or `obs = true`, in which case 
the finite and left Kronecker structures are preserved.

The keyword arguments `atol` and `rtol`, specify, respectively, the absolute and relative tolerance for the 
nonzero coefficients of the matrices `T(λ)`, `U(λ)`, `V(λ)` and `W(λ)`. The default relative tolerance is `nt*ϵ`, 
where `nt` is the size of the square matrix `T(λ)` and `ϵ` is the machine epsilon of the element type of its coefficients. 
"""
function spm2lps(T::Union{AbstractArray{T1,3},AbstractArray{T1,2}},U::Union{AbstractArray{T2,3},AbstractArray{T2,2}},
                V::Union{AbstractArray{T3,3},AbstractArray{T3,2}},W::Union{AbstractArray{T4,3},AbstractArray{T4,2}}; 
                contr::Bool = false, obs::Bool = false, minimal::Bool = false, 
                fast::Bool = true, atol::Real = zero(real(T1)), 
                rtol::Real = size(T,1)*eps(real(float(one(T1))))*iszero(atol)) where {T1, T2, T3, T4}
   if ndims(T) == 2 
      n, nt = size(T)
      n == nt || throw(DimensionMismatch("T(λ) must be a square polynomial matrix"))
      n == rank(T, atol=atol, rtol=rtol) || error("T(λ) must be a regular square polynomial matrix")
      ndT = 1
      T = reshape(T,n,nt,ndT)
   else
      n, nt, ndT = size(T)
      n == nt || throw(DimensionMismatch("T(λ) must be a square polynomial matrix"))
      n == pmrank(T, atol=atol, rtol=rtol) || error("T(λ) must be a regular square polynomial matrix")
      ndT = max(pmdeg(T)+1,1)
   end
   if ndims(U) == 2 
      nt, m = size(U)
      ndU = 1
      U = reshape(U,nt,m,ndU)
   else
      nt, m, ndU = size(U)
      ndU = max(pmdeg(U)+1,1)
   end
   n == nt || throw(DimensionMismatch("T(λ) and U(λ) must have the same number of rows"))
   if ndims(V) == 2 
      p, nt = size(V)
      ndV = 1
      V = reshape(V,p,nt,ndV)
   else
      p, nt, ndV = size(V)
      ndV = max(pmdeg(V)+1,1)
   end
   n == nt || throw(DimensionMismatch("T(λ) and V(λ) must have the same number of columns"))
   if ndims(W) == 2 
      pt, mt = size(W)
      ndW = 1
      W = reshape(W,pt, mt,ndW)
   else
      pt, mt, ndW = size(W)
      ndW = max(pmdeg(W)+1,1)
   end
   pt == p || throw(DimensionMismatch("W(λ) and V(λ) must have the same number of rows"))
   mt == m || throw(DimensionMismatch("W(λ) and U(λ) must have the same number of columns"))
   nd = max(ndT,ndU,ndV,ndW)
   TT = promote_type(T1, T2, T3, T4)
   if nd == 1
      if minimal
         Ar,Er,Br,Cr,Dr = lsminreal(T[:,:,1], zeros(TT,n,n), U[:,:,1], V[:,:,1], W[:,:,1], 
                                    contr = false, obs = false, atol1 = atol, atol2 = atol, rtol = rtol)
         nr = size(Ar,1)
         TW = eltype(Ar)
         return Ar,Er,Br,zeros(TW,nr,m),Cr,zeros(TW,p,nr),Dr,zeros(TW,p,m)
      else
        return T[:,:,1], zeros(TT,n,n), U[:,:,1], zeros(TT,n,m), V[:,:,1], zeros(TT,p,n), W[:,:,1], zeros(TT,p,m)
      end
   end

   # build the compound polynomial matrix [-T(λ) U(λ); V(λ) W(λ)]
   P = zeros(TT,n+p,n+m,nd)
   ia = 1:n
   jb = n+1:n+m
   ic = n+1:n+p
   P[ia,ia,1:ndT] = -T[:,:,1:ndT]
   P[ia,jb,1:ndU] = U[:,:,1:ndU]
   P[ic,ia,1:ndV] = V[:,:,1:ndV]
   P[ic,jb,1:ndW] = W[:,:,1:ndW]

   # build a linearization of the compound polynomial matrix [-T(λ) U(λ); V(λ) W(λ)]
   A, E, B, F, C, G, D, H = pm2lps(P, contr = contr, obs = obs)

   # form the linearization of P(λ) = V(λ)*inv(T(λ))*U(λ)+W(λ)                        
   nr = size(A,1)
   Ar = [A B[:,ia]; C[ia,:] D[ia,ia]]
   Er = [E F[:,ia]; G[ia,:] H[ia,ia]]
   Br = [B[:,jb]; D[ia,jb]]
   Fr = [F[:,jb]; H[ia,jb]]
   Cr = [C[ic,:] D[ic,ia]]
   Gr = [G[ic,:] H[ic,ia]]
   Dr = D[ic,jb]
   Hr = H[ic,jb]
   if minimal 
      Ar,Er,Br,Fr,Cr,Gr,Dr,Hr,Vr,Wr = lpsminreal(Ar,Er,Br,Fr,Cr,Gr,Dr,Hr, fast=fast, atol1 = atol, atol2 = atol, rtol = rtol)
      return Ar,Er,Br/Wr,Fr/Wr,Vr'\Cr,Vr'\Gr,Vr'\Dr/Wr,Vr'\Hr/Wr
   else
      return Ar,Er,Br,Fr,Cr,Gr,Dr,Hr
   end
end
spm2lps(T::Union{AbstractVecOrMat{Polynomial{T1}},Polynomial{T1},Number,AbstractMatrix{T1}}, U::Union{AbstractVecOrMat{Polynomial{T2}},Polynomial{T2},Number,AbstractVecOrMat{T2}}, 
       V::Union{AbstractVecOrMat{Polynomial{T3}},Polynomial{T3},Number,AbstractVecOrMat{T3}}, W::Union{AbstractVecOrMat{Polynomial{T4}},Polynomial{T4},Number,AbstractVecOrMat{T4}}; kwargs...) where {T1, T2, T3, T4} =
       spm2lps(poly2pm(T),poly2pm(U),poly2pm(V),poly2pm(W); kwargs...)
