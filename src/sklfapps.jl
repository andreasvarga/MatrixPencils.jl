"""
    spzeros(A, E, B, C, D; fast = false, atol1::Real = 0, atol2::Real = 0, rtol::Real=min(atol1,atol2)>0 ? 0 : n*ϵ) -> (val, iz, KRInfo)

Return the (finite and infinite) Smith zeros of the structured matrix pencil `M-λN` 

              | A-λE | B | 
     M - λN = |------|---|
              |  C   | D |  

in `val`, information on the multiplicities of infinite zeros in `iz` and  
information on the Kronecker-structure in the `KRInfo` object. 

The information on the multiplicities of infinite zeros is provided in the vector `iz`, 
where each `i`-th element `iz[i]` is equal to `k-1`, where `k` is the order of an infinite elementary divisor with `k > 0`.
The number of infinite zeros contained in `val` is the sum of the components of `iz`. 

The information on the Kronecker-structure consists of the right Kronecker indices `rki`, left Kronecker indices `lki`, 
infinite elementary divisors `id`, the number of finite eigenvalues `nf`, the normal rank `nrank`, and can be obtained 
from `KRInfo` as `KRInfo.rki`, `KRInfo.lki`, `KRInfo.id`, `KRInfo.nf` and `KRInfo.nrank`, respectively. 
For more details, see  [`pkstruct`](@ref). 

The computation of the zeros is performed by reducing the pencil `M-λN` to an appropriate Kronecker-like form (KLF) 
which exhibits the splitting of the infinite and finite eigenvalue, 
the multiplicities of the infinite eigenvalues, the left and right Kronecker indices and the normal rank.
The reduction is performed using orthonal similarity transformations and involves rank decisions based on rank revealing QR-decompositions with column pivoting, 
if `fast = true`, or, the more reliable, SVD-decompositions, if `fast = false`. For efficiency purposes, the reduction is only
partially performed, without accumulating the performed orthogonal transformations.

The keyword arguements `atol1`, `atol2`  and `rtol` specify the absolute tolerance for the nonzero
elements of `M`, the absolute tolerance for the nonzero elements of `N`, and the relative tolerance for the nonzero elements of `M` and `N`, respectively. 
The default relative tolerance is `n*ϵ`, where `n` is the size of the smallest dimension of `M`, and `ϵ` is the 
machine epsilon of the element type of `M`. 
"""
function spzeros(A::Union{AbstractMatrix,Missing}, E::Union{AbstractMatrix,UniformScaling{Bool},Missing}, 
    B::Union{AbstractVecOrMat,Missing}, C::Union{AbstractMatrix,Missing}, D::Union{AbstractVecOrMat,Missing}; 
    fast::Bool = true, atol1::Real = ismissing(A) ? zero(real(eltype(D))) : zero(real(eltype(A))), 
    atol2::Real = ismissing(A) ? zero(real(eltype(D))) : zero(real(eltype(A))), 
    rtol::Real = (ismissing(A) ? 1 : min(size(A)...))*eps(real(float(ismissing(A) ? one(real(eltype(D))) : one(real(eltype(A))))))*iszero(min(atol1,atol2))) 
 
   # Step 0: Reduce to the standard form
   M, N, Q, Z, n, m, p = sreduceBF(A, E, B, C, D, atol = atol2, rtol = rtol, fast = fast, withQ = false, withZ = false)
   mM, nM = size(M)
   T = eltype(M)
  
   maxmn = max(mM,nM)
   rki = Vector{Int}(undef,maxmn)
   id = Vector{Int}(undef,maxmn)
   lki = Vector{Int}(undef,maxmn)
 
   tol1 = max(atol1, rtol*opnorm(M,1))
   mrinf = 0
   nrinf = 0
   niz = 0
   i = 0
   while m > 0 
      # Steps 1 & 2: Standard algorithm PREDUCE 
      ired = mrinf+1:mM
      jred = nrinf+1:nM
      τ, ρ = _preduce1!(n, m, p, view(M,ired,jred), view(N,ired,jred), Q, Z, tol1; 
                        fast = fast, withQ = false, withZ = false)
      i += 1
      mui = ρ+τ
      rki[i] = m - mui
      id[i]  = τ
      mrinf += mui
      nrinf += m
      i > 1 && (niz += τ*(i-1))
      n -= ρ
      m = ρ
      p -= τ 
   end

   r = mrinf + n

   rtrail = 0
   ctrail = 0
   j = 0
   while p > 0
      # Step 3: Particular case of the dual PREDUCE algorithm 
      ired = mrinf+1:mM-rtrail
      jred = nrinf+1:nM-ctrail
      ρ = _preduce4!(n, 0, p, view(M,ired,jred),view(N,ired,jred), Q, Z, tol1, fast = fast, withQ = false, withZ = false)
      j += 1
      rtrail += p
      ctrail += ρ
      lki[j] = p - ρ
      n -= ρ
      p = ρ
   end
   if1 = mrinf+1:mrinf+n
   jf1 = nrinf+1:nrinf+n
   return [eigvalsnosort!(view(M,if1,jf1),view(N,if1,jf1)); Inf*ones(real(T),niz) ], minf(id[2:i]), 
           KRInfo(kroni(rki[1:i]), kroni(lki[1:j]), minf(id[1:i]), n, r)
end
"""
    speigvals(A, E, B, C, D; fast = false, atol1::Real = 0, atol2::Real = 0, rtol::Real=min(atol1,atol2)>0 ? 0 : n*ϵ) -> (val, iz, KRInfo)

Return the (finite and infinite) eigenvalues of the structured matrix pencil `M-λN`  

              | A-λE | B | 
     M - λN = |------|---|
              |  C   | D |  

in `val` and information on the  Kronecker-structure in the `KRInfo` object. 

The information on the  Kronecker-structure consists of the right Kronecker indices `rki`, left Kronecker indices `lki`, 
infinite elementary divisors `id`, the number of finite eigenvalues `nf`, the normal rank `nrank`, and can be obtained 
from `KRInfo` as `KRInfo.rki`, `KRInfo.lki`, `KRInfo.id`, `KRInfo.nf` and `KRInfo.nrank`, respectively. 
For more details, see  [`pkstruct`](@ref). 

The computation of the eigenvalues
is performed by reducing the pencil `M-λN` to an appropriate Kronecker-like form (KLF) 
which exhibits the splitting of the infinite and finite eigenvalue, 
the multiplicities of the infinite eigenvalues, the left and right Kronecker indices and the normal rank.
The reduction is performed using orthonal similarity transformations and involves rank decisions based on rank revealing QR-decompositions with column pivoting, 
if `fast = true`, or, the more reliable, SVD-decompositions, if `fast = false`. For efficiency purposes, the reduction is only
partially performed, without accumulating the performed orthogonal transformations. 

The keyword arguements `atol1`, `atol2`  and `rtol` specify the absolute tolerance for the nonzero
elements of `M`, the absolute tolerance for the nonzero elements of `N`, and the relative tolerance for the nonzero elements of `M` and `N`, respectively. 
The default relative tolerance is `n*ϵ`, where `n` is the size of the smallest dimension of `M`, and `ϵ` is the 
machine epsilon of the element type of `M`. 
"""
function speigvals(A::Union{AbstractMatrix,Missing}, E::Union{AbstractMatrix,UniformScaling{Bool},Missing}, 
   B::Union{AbstractVecOrMat,Missing}, C::Union{AbstractMatrix,Missing}, D::Union{AbstractVecOrMat,Missing}; 
   fast::Bool = true, atol1::Real = ismissing(A) ? zero(real(eltype(D))) : zero(real(eltype(A))), 
   atol2::Real = ismissing(A) ? zero(real(eltype(D))) : zero(real(eltype(A))), 
   rtol::Real = (ismissing(A) ? 1 : min(size(A)...))*eps(real(float(ismissing(A) ? one(real(eltype(D))) : one(real(eltype(A))))))*iszero(min(atol1,atol2))) 

   # Step 0: Reduce to the standard form
   M, N, Q, Z, n, m, p = sreduceBF(A, E, B, C, D, atol = atol2, rtol = rtol, fast = fast, withQ = false, withZ = false)
   mM, nM = size(M)
   T = eltype(M)
 
   maxmn = max(mM,nM)
   rki = Vector{Int}(undef,maxmn)
   id = Vector{Int}(undef,maxmn)
   lki = Vector{Int}(undef,maxmn)

   tol1 = max(atol1, rtol*opnorm(M,1))
   mrinf = 0
   nrinf = 0
   ni = 0
   i = 0
   while m > 0 
      # Steps 1 & 2: Standard algorithm PREDUCE 
      ired = mrinf+1:mM
      jred = nrinf+1:nM
      τ, ρ = _preduce1!(n, m, p, view(M,ired,jred), view(N,ired,jred), Q, Z, tol1; 
                        fast = fast, withQ = false, withZ = false)
      i += 1
      mui = ρ+τ
      rki[i] = m - mui
      id[i]  = τ
      mrinf += mui
      nrinf += m
      ni += τ*i
      n -= ρ
      m = ρ
      p -= τ 
   end

   r = mrinf + n

   rtrail = 0
   ctrail = 0
   j = 0
   while p > 0
      # Step 3: Particular case of the dual PREDUCE algorithm 
      ired = mrinf+1:mM-rtrail
      jred = nrinf+1:nM-ctrail
      ρ = _preduce4!(n, 0, p, view(M,ired,jred),view(N,ired,jred), Q, Z, tol1, fast = fast, withQ = false, withZ = false)
      rtrail += p
      ctrail += ρ
      j += 1
      lki[j] = p - ρ
      n -= ρ
      p = ρ
   end
   if1 = mrinf+1:mrinf+n
   jf1 = nrinf+1:nrinf+n
   return [eigvalsnosort!(view(M,if1,jf1),view(N,if1,jf1)); Inf*ones(real(T),ni) ], 
          KRInfo(kroni(rki[1:i]), kroni(lki[1:j]), minf(id[1:i]), n, r)
end
"""
    spkstruct(A, E, B, C, D; fast = false, atol1::Real = 0, atol2::Real = 0, rtol::Real=min(atol1,atol2)>0 ? 0 : n*ϵ) -> KRInfo
  
Determine the Kronecker-structure information of the structured matrix pencil `M-λN` 

              | A-λE | B | 
     M - λN = |------|---|
              |  C   | D |  

and return an `KRInfo` object. 

The information on the  Kronecker-structure consists of the right Kronecker indices `rki`, left Kronecker indices `lki`, 
infinite elementary divisors `id`, the number of finite eigenvalues `nf`, the normal rank `nrank`, and can be obtained 
from `KRInfo` as `KRInfo.rki`, `KRInfo.lki`, `KRInfo.id`, `KRInfo.nf` and `KRInfo.nrank`, respectively. 
For more details, see  [`pkstruct`](@ref). 
 
The right Kronecker indices are provided in the integer vector `rki`, where each `i`-th element `rki[i]` is 
the row dimension `k` of an elementary Kronecker block of size `k x (k+1)`. 
The number of elements of `rki` is the dimension of 
the right nullspace of the pencil `M-λN` and their sum is the least degree of a right polynomial nullspace basis. 

The left Kronecker indices are provided in the integer vector `lki`, where each `i`-th element `lki[i]` is 
the column dimension `k` of an elementary Kronecker block of size `(k+1) x k`. 
The number of elements of `lki` is the dimension of 
the left nullspace of the pencil `M-λN` and their sum is the least degree of a left polynomial nullspace basis. 

The multiplicities of infinite eigenvalues are provided in the integer vector `id`, where each `i`-th element `id[i]` is
the order of an infinite elementary divisor (i.e., the multiplicity of an infinite eigenvalue). 

The determination of the Kronecker-structure information is performed by reducing the pencil `M-λN` to an 
appropriate Kronecker-like form (KLF) which exhibits the number of finite eigenvalues, 
the multiplicities of the infinite eigenvalues, the left and right Kronecker indices and the normal rank.
The reduction is performed using orthonal similarity transformations and involves rank decisions based 
on rank revealing QR-decompositions with column pivoting, if `fast = true`, or, the more reliable, 
SVD-decompositions, if `fast = false`. For efficiency purposes, the reduction is only
partially performed, without accumulating the performed orthogonal transformations. 

The keyword arguements `atol1`, `atol2`  and `rtol` specify the absolute tolerance for the nonzero
elements of `M`, the absolute tolerance for the nonzero elements of `N`, and the relative tolerance for the nonzero elements of `M` and `N`, respectively. 
The default relative tolerance is `n*ϵ`, where `n` is the size of the smallest dimension of `M`, and `ϵ` is the 
machine epsilon of the element type of `M`. 
"""
function spkstruct(A::Union{AbstractMatrix,Missing}, E::Union{AbstractMatrix,UniformScaling{Bool},Missing}, 
   B::Union{AbstractVecOrMat,Missing}, C::Union{AbstractMatrix,Missing}, D::Union{AbstractVecOrMat,Missing}; 
   fast::Bool = true, atol1::Real = ismissing(A) ? zero(real(eltype(D))) : zero(real(eltype(A))), 
   atol2::Real = ismissing(A) ? zero(real(eltype(D))) : zero(real(eltype(A))), 
   rtol::Real = (ismissing(A) ? 1 : min(size(A)...))*eps(real(float(ismissing(A) ? one(real(eltype(D))) : one(real(eltype(A))))))*iszero(min(atol1,atol2))) 

   # Step 0: Reduce to the standard form
   M, N, Q, Z, n, m, p = sreduceBF(A, E, B, C, D, atol = atol2, rtol = rtol, fast = fast, withQ = false, withZ = false)
   mM, nM = size(M)
   T = eltype(M)
  
   maxmn = max(mM,nM)
   rki = Vector{Int}(undef,maxmn)
   id = Vector{Int}(undef,maxmn)
   lki = Vector{Int}(undef,maxmn)
   
   tol1 = max(atol1, rtol*opnorm(M,1))
   mrinf = 0
   nrinf = 0
   ni = 0
   i = 0
   while m > 0 
      # Steps 1 & 2: Standard algorithm PREDUCE 
      ired = mrinf+1:mM
      jred = nrinf+1:nM
      τ, ρ = _preduce1!(n, m, p, view(M,ired,jred), view(N,ired,jred), Q, Z, tol1; 
                        fast = fast, withQ = false, withZ = false)
      i += 1
      mui = ρ+τ
      rki[i] = m - mui
      id[i]  = τ
      mrinf += mui
      nrinf += m
      n -= ρ
      m = ρ
      p -= τ 
   end

   r = mrinf + n

   rtrail = 0
   ctrail = 0
   j = 0
   while p > 0
      # Step 3: Particular case of the dual PREDUCE algorithm 
      ired = mrinf+1:mM-rtrail
      jred = nrinf+1:nM-ctrail
      ρ = _preduce4!(n, 0, p, view(M,ired,jred),view(N,ired,jred), Q, Z, tol1, fast = fast, withQ = false, withZ = false)
      j += 1
      rtrail += p
      ctrail += ρ
      lki[j] = p - ρ
      n -= ρ
      p = ρ
   end
   return KRInfo(kroni(rki[1:i]), kroni(lki[1:j]), minf(id[1:i]), n, r)
end
"""
    sprank(A, E, B, C, D; fastrank = true, atol1 = 0, atol2 = 0, rtol = min(atol1,atol2)>0 ? 0 : n*ϵ)

Compute the normal rank of the structured matrix pencil `M - λN`

              | A-λE | B | 
     M - λN = |------|---|.
              |  C   | D |  

If `fastrank = true`, the rank is evaluated by counting how many singular values of `M - γ N` have magnitude greater than `max(max(atol1,atol2), rtol*σ₁)`,
where `σ₁` is the largest singular value of `M - γ N` and `γ` is a randomly generated value. If `fastrank = false`, 
the rank is evaluated as `nr + ni + nf + nl`, where `nr` and `nl` are the sums of right and left Kronecker indices, 
respectively, while `ni` and `nf` are the number of infinite and finite eigenvalues, respectively. The sums `nr+ni` and  
`nf+nl`, are determined from an appropriate Kronecker-like form (KLF) exhibiting the spliting of the right and left structures 
of the pencil `M - λN`.

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of `M`, the absolute tolerance for the nonzero elements of `N`,  and the relative tolerance 
for the nonzero elements of `M` and `N`. The default relative tolerance is `n*ϵ`, where `n` is the size of the smallest dimension of `M`, and `ϵ` is the 
machine epsilon of the element type of `M`. 
For efficiency purpose, the reduction to the relevant KLF is only partially performed 
using rank decisions based on rank revealing SVD-decompositions. 

!!! compat "Julia 1.1"
    The use of `atol` and `rtol` keyword arguments in rank determinations requires at least Julia 1.1. 
    To enforce compatibility with Julia 1.0, the newer function rank in Julia 1.1 has been explicitly included. 
"""
function sprank(A::Union{AbstractMatrix,Missing}, E::Union{AbstractMatrix,UniformScaling{Bool},Missing}, 
   B::Union{AbstractVecOrMat,Missing}, C::Union{AbstractMatrix,Missing}, D::Union{AbstractVecOrMat,Missing}; 
   fastrank::Bool = true, atol1::Real = ismissing(A) ? (ismissing(D) ? zero(1.) : zero(real(eltype(D)))) : zero(real(eltype(A))), 
   atol2::Real = ismissing(A) ? (ismissing(D) ? zero(1.) : zero(real(eltype(D)))) : zero(real(eltype(A))), 
   rtol::Real = (ismissing(A) ? 1 : min(size(A)...))*eps(real(float(ismissing(A) ? (ismissing(D) ? one(1.) : one(real(eltype(D)))) : one(real(eltype(A))))))*iszero(min(atol1,atol2))) 

   if fastrank
      xor(ismissing(A),ismissing(E)) && error("A and E must be both either present or missing")               
      ismissing(A) && !ismissing(B) && error("B can not be present if A is missing")  
      ismissing(A) && !ismissing(C) && error("C can not be present if A is missing")  
      !ismissing(D) && !ismissing(B) && ismissing(C)  && error("D can not be present if C is missing") 
      !ismissing(D) && !ismissing(C) && ismissing(B)  && error("D can not be present if B is missing") 
      eident = (typeof(E) == UniformScaling{Bool}) 
      if ismissing(A) && ismissing(D)
         return 0
      elseif ismissing(A) 
         return rank(D,atol = atol1, rtol = rtol)
      elseif ismissing(B) && ismissing(C)
         isa(A,Adjoint) && (A = copy(A))
         isa(E,Adjoint) && (E = copy(E))
         ndx, nx = size(A)
         eident || (ndx,nx) == size(E) || throw(DimensionMismatch("A and E must have the same dimensions"))
         eident ? T = eltype(A) : T = promote_type(eltype(A), eltype(E))
         T <: BlasFloat || (T = promote_type(Float64,T))
         nu = 0
         B = zeros(T,ndx,nu)
         ny = 0
         C = zeros(T,ny,nx)
         D = zeros(T,ny,nu)
      elseif ismissing(B)
         isa(A,Adjoint) && (A = copy(A))
         isa(E,Adjoint) && (E = copy(E))
         ndx, nx = size(A)
         eident || (ndx,nx) == size(E) || throw(DimensionMismatch("A and E must have the same dimensions"))
         nx == size(C,2) || throw(DimensionMismatch("A and C must have the same number of columns"))
         T = promote_type(eltype(A), eltype(E), eltype(C) )
         T <: BlasFloat || (T = promote_type(Float64,T))
         nu = 0
         B = zeros(T,ndx,nu)
         ny = size(C,1)
         D = zeros(T,ny,nu)
      elseif ismissing(C)
         isa(A,Adjoint) && (A = copy(A))
         isa(E,Adjoint) && (E = copy(E))
         ndx, nx = size(A)
         eident || (ndx,nx) == size(E) || throw(DimensionMismatch("A and E must have the same dimensions"))
         ndx1, nu = typeof(B) <: AbstractVector ? (length(B),1) : size(B)
         ndx == ndx1 || throw(DimensionMismatch("A and B must have the same number of rows"))
         T = promote_type(eltype(A), eltype(E), eltype(B), eltype(E), eltype(A), eltype(E), )
         T <: BlasFloat || (T = promote_type(Float64,T))
         ny = 0
         C = zeros(T,ny,nx)
         D = zeros(T,ny,nu)
      else
         isa(A,Adjoint) && (A = copy(A))
         isa(E,Adjoint) && (E = copy(E))
         ndx, nx = size(A)
         T = promote_type(eltype(A), eltype(B), eltype(C), eltype(D))
         eident || (ndx,nx) == size(E) || throw(DimensionMismatch("A and E must have the same dimensions"))
         ny, nu = typeof(D) <: AbstractVector ? (length(D),1) : size(D)
         ndx1, nu1 = typeof(B) <: AbstractVector ? (length(B),1) : size(B)
         (ndx,nu) == (ndx1, nu1) || throw(DimensionMismatch("A, B and D must have compatible dimensions"))
         (ny,nx) == size(C) || throw(DimensionMismatch("A, C and D must have compatible dimensions"))
         T = promote_type(eltype(A), eltype(E), eltype(B), eltype(C), eltype(D))
         T <: BlasFloat || (T = promote_type(Float64,T))        
      end
   
      eident || (T = promote_type(T,eltype(E)))
      T <: BlasFloat || (T = promote_type(Float64,T))
      if eident 
         ndx == nx || throw(DimensionMismatch("A must be a square matrix"))
      end
      # fast finishing if possible
      max(ndx,nx,nu,ny) == 0 && (return 0)
     
      (!ismissing(A) && eltype(A) != T) && (A = convert(Matrix{T},A))
      (!eident && !ismissing(E) && eltype(E) != T) && (E = convert(Matrix{T},E))
      (!ismissing(B) && eltype(B) != T) && (B = convert(Matrix{T},B))
      (!ismissing(C) && eltype(C) != T) && (C = convert(Matrix{T},C))
      (!ismissing(D) && eltype(D) != T) && (D = convert(Matrix{T},D))
      if eident 
         scale = opnorm(A,1)*rand()
         return rank([A+scale*E B; C D], atol = atol1, rtol = rtol)
      else
         nrmN = opnorm(E,1)
         nrmN == zero(nrmN) && (return rank([A B; C D], atol = atol1, rtol = rtol))
         nrmM = max(opnorm(A,1), typeof(B) <: AbstractVector ? norm(B,1) : opnorm(B,1), opnorm(C,Inf), typeof(D) <: AbstractVector ? norm(D,1) : opnorm(D,1))
         scale = nrmM/nrmN*rand()
         return rank([A+scale*E B; C D], atol = max(atol1,atol2), rtol = rtol)
      end
   else
      M, N, Q, Z, n, m, p = sreduceBF(A, E, B, C, D, atol = atol2, rtol = rtol, 
                                      fast = false, withQ = false, withZ = false)
      mM, nM = size(M)
      
      n == min(mM,nM) && (return n)
      tol1 = max(atol1, rtol*opnorm(M,1))
      mrinf = 0
      nrinf = 0
      while m > 0 
         # Steps 1 & 2: Standard algorithm PREDUCE 
         ired = mrinf+1:mM
         jred = nrinf+1:nM
         τ, ρ = _preduce1!(n, m, p, view(M,ired,jred), view(N,ired,jred), Q, Z, tol1; 
                           fast = false, withQ = false, withZ = false)
         mrinf += ρ+τ
         nrinf += m
         n -= ρ
         m = ρ
         p -= τ 
      end
      return mrinf+n
   end
end
