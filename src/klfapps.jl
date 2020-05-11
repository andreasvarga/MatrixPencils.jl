"""
    pzeros(M, N; fast = false, atol1::Real = 0, atol2::Real = 0, rtol::Real=min(atol1,atol2)>0 ? 0 : n*ϵ) -> (val, iz, KRInfo)

Return the finite and infinite zeros of the linear pencil `M-λN` in `val`, the multiplicities of infinite zeros in `iz` and  the 
information on the complete Kronecker-structure in the `KRInfo` object. 

The information on the multiplicities of infinite zeros is provided in the vector `iz`, 
where each `i`-th element `iz[i]` is equal to `k-1`, where `k` is the order of an infinite elementary divisor with `k > 0`.
The number of infinite zeros contained in `val` is the sum of the components of `iz`. 

The information on the complete Kronecker-structure consists of the right Kronecker indices `rki`, left Kronecker indices `lki`, 
infinite elementary divisors `id` and the
number of finite eigenvalues `nf`, and can be obtained from `KRInfo` as `KRInfo.rki`, `KRInfo.lki`, `KRInfo.id` 
and `KRInfo.nf`, respectively. For more details, see  [`pkstruct`](@ref). 

The computation of the zeros is performed by reducing the pencil `M-λN` to an appropriate Kronecker-like form (KLF) 
exhibiting the spliting of the infinite and finite eigenvalue structures of the pencil `M-λN`. 
The reduction is performed using orthonal similarity transformations and involves rank decisions based on rank reevealing QR-decompositions with column pivoting, 
if `fast = true`, or, the more reliable, SVD-decompositions, if `fast = false`. For efficiency purposes, the reduction is only
partially performed, without accumulating the performed orthogonal transformations.

The keyword arguements `atol1`, `atol2`  and `rtol` specify the absolute tolerance for the nonzero
elements of `M`, the absolute tolerance for the nonzero elements of `N`, and the relative tolerance for the nonzero elements of `M` and `N`, respectively. 
The default relative tolerance is `n*ϵ`, where `n` is the size of the smallest dimension of `M`, and `ϵ` is the 
machine epsilon of the element type of `M`. 
"""
function pzeros(M::AbstractMatrix, N::Union{AbstractMatrix,Nothing}; fast::Bool = true, 
   atol1::Real = zero(real(eltype(M))), atol2::Real = zero(real(eltype(M))), 
   rtol::Real = (min(size(M)...)*eps(real(float(one(eltype(M))))))*iszero(min(atol1,atol2)))

   mM, nM = size(M)
   if N === nothing
      r = rank(M,atol = atol1, rtol = rtol)
      return eltype(M)[], Int[], KRInfo(zeros(Int,nM-r), zeros(Int,mM-r), zeros(Int,0), 0)
   end

   # Step 0: Reduce to the standard form
   M1, N1, Q, Z, n, m, p = preduceBF(M, N; atol = atol2, rtol = rtol, fast = fast, withQ = false, withZ = false) 

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
      τ, ρ = _preduce1!(n, m, p, view(M1,ired,jred), view(N1,ired,jred), Q, Z, tol1; 
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

   rtrail = 0
   ctrail = 0
   j = 0
   while p > 0
      # Step 3: Particular case of the dual PREDUCE algorithm 
      ired = mrinf+1:mM-rtrail
      jred = nrinf+1:nM-ctrail
      ρ = _preduce4!(n, 0, p, view(M1,ired,jred),view(N1,ired,jred), Q, Z, tol1, fast = fast, withQ = false, withZ = false)
      j += 1
      rtrail += p
      ctrail += ρ
      lki[j] = p - ρ
      n -= ρ
      p = ρ
   end
   if1 = mrinf+1:mrinf+n
   jf1 = nrinf+1:nrinf+n
   return [eigvals(M1[if1,jf1],N1[if1,jf1],sortby=nothing); Inf*ones(real(eltype(M1)),niz) ], minf(id[2:i]), 
           KRInfo(kroni(rki[1:i]), kroni(lki[1:j]), minf(id[1:i]), n)
end
"""
    peigvals(M, N; fast = false, atol1::Real = 0, atol2::Real = 0, rtol::Real=min(atol1,atol2)>0 ? 0 : n*ϵ) -> (val, KRInfo)

Return the (finite and infinite) eigenvalues of the linear pencil `M-λN` in `val` and the 
information on the complete Kronecker-structure in the `KRInfo` object. 

The information on the complete Kronecker-structure consists of the right Kronecker indices `rki`, left Kronecker indices `lki`, infinite elementary divisors `id` and the
number of finite eigenvalues `nf`, and can be obtained from `KRInfo` as `KRInfo.rki`, `KRInfo.lki`, `KRInfo.id` 
and `KRInfo.nf`, respectively. For more details, see  [`pkstruct`](@ref).  

The computation of the eigenvalues
is performed by reducing the pencil `M-λN` to an appropriate Kronecker-like form (KLF) exhibiting the spliting 
of the infinite and finite eigenvalue structures of the pencil `M-λN`.
The reduction is performed using orthonal similarity transformations and involves rank decisions based on rank reevealing QR-decompositions with column pivoting, 
if `fast = true`, or, the more reliable, SVD-decompositions, if `fast = false`. For efficiency purposes, the reduction is only
partially performed, without accumulating the performed orthogonal transformations. 

The keyword arguements `atol1`, `atol2`  and `rtol` specify the absolute tolerance for the nonzero
elements of `M`, the absolute tolerance for the nonzero elements of `N`, and the relative tolerance for the nonzero elements of `M` and `N`, respectively. 
The default relative tolerance is `n*ϵ`, where `n` is the size of the smallest dimension of `M`, and `ϵ` is the 
machine epsilon of the element type of `M`. 
"""
function peigvals(M::AbstractMatrix, N::Union{AbstractMatrix,Nothing}; fast::Bool = true, 
                  atol1::Real = zero(real(eltype(M))), atol2::Real = zero(real(eltype(M))), 
                  rtol::Real = (min(size(M)...)*eps(real(float(one(eltype(M))))))*iszero(min(atol1,atol2)))

   mM, nM = size(M)
   if N === nothing
      r = rank(M,atol = atol1, rtol = rtol)
      return eltype(M)[], KRInfo(zeros(Int,nM-r), zeros(Int,mM-r), zeros(Int,0), 0)
   end

   # Step 0: Reduce to the standard form
   M1, N1, Q, Z, n, m, p = preduceBF(M, N; atol = atol2, rtol = rtol, fast = fast, withQ = false, withZ = false) 

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
      τ, ρ = _preduce1!(n, m, p, view(M1,ired,jred), view(N1,ired,jred), Q, Z, tol1; 
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
   rtrail = 0
   ctrail = 0
   j = 0
   while p > 0
      # Step 3: Particular case of the dual PREDUCE algorithm 
      ired = mrinf+1:mM-rtrail
      jred = nrinf+1:nM-ctrail
      ρ = _preduce4!(n, 0, p, view(M1,ired,jred),view(N1,ired,jred), Q, Z, tol1, fast = fast, withQ = false, withZ = false)
      rtrail += p
      ctrail += ρ
      j += 1
      lki[j] = p - ρ
      n -= ρ
      p = ρ
   end
   if1 = mrinf+1:mrinf+n
   jf1 = nrinf+1:nrinf+n
   return [eigvals(M1[if1,jf1],N1[if1,jf1],sortby=nothing); Inf*ones(real(eltype(M1)),ni) ], 
          KRInfo(kroni(rki[1:i]), kroni(lki[1:j]), minf(id[1:i]), n)
end
"""
    KRInfo
  
Kronecker-structure object definition. 

If `info::KRInfo` is the Kronecker-structure object, then:

`info.rki` is a vector, whose components contains the column dimensions of  
elementary Kronecker blocks of the form `(k-1) x k`, called the right Kronecker indices;

`info.lki` is a vector, whose components contains the row dimensions of  
elementary Kronecker blocks of the form `k x (k-1)`, called the left Kronecker indices;

`info.id` is a vector, whose components contains the orders of the infinite elementary divisors (i.e., the
multiplicities of infinite eigenvalues). 

`info.nf` is the number of finite eigenvalues.

Destructuring via iteration produces the components `info.rki`, `info.lki`, `info.id`, and `info.nf`.
"""
mutable struct KRInfo
   rki::Vector{Int}
   lki::Vector{Int}
   id::Vector{Int}
   nf::Int
   function KRInfo(rki,lki,id,nf)
      (any(rki .< 0) || any(lki .< 0) || any(id .< 0) || nf < 0) && error("no negative components allowed")
      new(rki,lki,id,nf)
   end
end
# iteration for destructuring into components
Base.iterate(info::KRInfo) = (info.rki, Val(:lki))
Base.iterate(info::KRInfo, ::Val{:lki}) = (info.lki, Val(:id))
Base.iterate(info::KRInfo, ::Val{:id}) = (info.id, Val(:nf))
Base.iterate(info::KRInfo, ::Val{:nf}) = (info.nf, Val(:done))
#Base.iterate(info::KRInfo, ::Val{:done}) = nothing
"""
    pkstruct(M, N; fast = false, atol1::Real = 0, atol2::Real = 0, rtol::Real=min(atol1,atol2)>0 ? 0 : n*ϵ) -> KRInfo
  
Determine the Kronecker-structure information of the linear pencil `M-λN` and return an `KRInfo` object. 

The right Kronecker indices `rki`, left Kronecker indices `lki`, infinite elementary divisors `id` and the
number of finite eigenvalues `nf` can be obtained from `KRInfo` as `KRInfo.rki`, `KRInfo.lki`, `KRInfo.id` 
and `KRInfo.nf`, respectively.  
The determination of the Kronecker-structure information is performed by reducing the pencil `M-λN` to an 
appropriate Kronecker-like form (KLF) exhibiting all structural elements of the pencil `M-λN`.
The reduction is performed using orthogonal similarity transformations and involves rank decisions based 
on rank reevealing QR-decompositions with column pivoting, if `fast = true`, or, the more reliable, 
SVD-decompositions, if `fast = false`. For efficiency purposes, the reduction is only
partially performed, without accumulating the performed orthogonal transformations. 

The right Kronecker indices are provided in the integer vector `rki`, where each `i`-th element `rki[i]` is 
the column dimension `k` of an elementary Kronecker block of size `(k-1) x k`. 
The number of elements of `rki` is the dimension of 
the right nullspace of the pencil `M-λN` and their sum is the least degree of a right polynomial nullspace basis. 

The left Kronecker indices are provided in the integer vector `lki`, where each `i`-th element `lki[i]` is 
the row dimension `k` of an elementary Kronecker block of size `k x (k-1)`. 
The number of elements of `lki` is the dimension of 
the left nullspace of the pencil `M-λN` and their sum is the least degree of a left polynomial nullspace basis. 

The multiplicities of infinite eigenvalues are provided in the integer vector `id`, where each `i`-th element `id[i]` is
the order of an infinite elementary divisor (i.e., the multiplicity of an infinite eigenvalue). 

The keyword arguements `atol1`, `atol2`  and `rtol` specify the absolute tolerance for the nonzero
elements of `M`, the absolute tolerance for the nonzero elements of `N`, and the relative tolerance for the nonzero elements of `M` and `N`, respectively. 
The default relative tolerance is `n*ϵ`, where `n` is the size of the smallest dimension of `M`, and `ϵ` is the 
machine epsilon of the element type of `M`. 
"""
function pkstruct(M::AbstractMatrix, N::Union{AbstractMatrix,Nothing}; fast = false, atol1::Real = zero(eltype(M)), atol2::Real = zero(eltype(M)), 
   rtol::Real = (min(size(M)...)*eps(real(float(one(eltype(M))))))*iszero(min(atol1,atol2))) 

   mM, nM = size(M)
   if N === nothing
      r = rank(M,atol = atol1, rtol = rtol)
      return KRInfo(zeros(Int,nM-r), zeros(Int,mM-r), zeros(Int,0), 0)
   end
   # Step 0: Reduce to the standard form
   M1, N1, Q, Z, n, m, p = preduceBF(M, N; atol = atol2, rtol = rtol, fast = fast, withQ = false, withZ = false) 
   
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
      τ, ρ = _preduce1!(n, m, p, view(M1,ired,jred), view(N1,ired,jred), Q, Z, tol1; 
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
   rtrail = 0
   ctrail = 0
   j = 0
   while p > 0
      # Step 3: Particular case of the dual PREDUCE algorithm 
      ired = mrinf+1:mM-rtrail
      jred = nrinf+1:nM-ctrail
      ρ = _preduce4!(n, 0, p, view(M1,ired,jred),view(N1,ired,jred), Q, Z, tol1, fast = fast, withQ = false, withZ = false)
      j += 1
      rtrail += p
      ctrail += ρ
      lki[j] = p - ρ
      n -= ρ
      p = ρ
   end
   return KRInfo(kroni(rki[1:i]), kroni(lki[1:j]), minf(id[1:i]), n)
end
function deltrz(ind)
   k = findlast(!iszero,ind)
   k === nothing ? (return ind[1:0]) : (return ind[1:k]) 
end
function kroni(ind)
   # Kronecker indices evaluated from the ranks of blocks of the staircase form
   k = findlast(!iszero,ind)
   k === nothing && (return ind[1:0]) 
   ni = sum(ind[1:k])
   ki = similar(ind,ni)
   ii = 0
   for i = 1:k
       iip = ii+ind[i]
       for j = ii+1:iip
         ki[j] = i-1
       end
       ii = iip 
   end
   return ki 
end
function minf(ind)
   # multiplicities of infinite eigenvalues evaluated from the blocksizes of the staircase form
   k = findlast(!iszero,ind)
   k === nothing && (return ind[1:0]) 
   ni = sum(ind[1:k])
   mi = similar(ind,ni)
   ii = 0
   for i = 1:k
       iip = ii+ind[i]
       for j = ii+1:iip
         mi[j] = i
       end
       ii = iip 
   end
   return mi 
end
"""
    prank(M::AbstractMatrix, N::AbstractMatrix; fastrank = true, atol1::Real=0, atol2::Real=0, rtol::Real=min(atol1,atol2)>0 ? 0 : n*ϵ)

Compute the normal rank of a linear matrix pencil `M - λN`. If `fastrank = true`, the rank is evaluated 
by counting how many singular values of `M - γ N` have magnitude greater than `max(max(atol1,atol2), rtol*σ₁)`,
where `σ₁` is the largest singular value of `M - γ N` and `γ` is a randomly generated value. If `fastrank = false`, 
the rank is evaluated as `nr + ni + nf + nl`, where `nr` and `nl` are the sums of right and left Kronecker indices, 
respectively, while `ni` and `nf` are the number of finite and infinite eigenvalues, respectively. The sums `nr+ni` and  
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
function prank(M::AbstractMatrix, N::Union{AbstractMatrix,Nothing}; fastrank::Bool = true, 
   atol1::Real = zero(real(eltype(M))), atol2::Real = zero(real(eltype(M))), 
   rtol::Real = (min(size(M)...)*eps(real(float(one(eltype(M))))))*iszero(min(atol1,atol2)))

   mM, nM = size(M)
   if N === nothing
      return rank(M, atol = atol1, rtol = rtol)
   end

   (mM,nM) == size(N) || throw(DimensionMismatch("M and N must have the same dimensions"))

   if fastrank
      T = promote_type(eltype(M), eltype(N))
      T <: BlasFloat || (T = promote_type(Float64,T))
      M1 = copy_oftype(M,T)
      N1 = copy_oftype(N,T)
      nrmM = opnorm(M1,1)
      nrmM == zero(nrmM) && (return rank(N1, atol = atol2, rtol = rtol))
      nrmN = opnorm(N1,1)
      nrmN == zero(nrmN) && (return rank(M1, atol = atol1, rtol = rtol))
      scale = nrmM/nrmN*rand()
      return rank(M1+scale*N1, atol = max(atol1,atol2), rtol = rtol)
   else
      # Step 0: Reduce to the standard form
      M1, N1, Q, Z, n, m, p = preduceBF(M, N; atol = atol2, rtol = rtol, fast = false, withQ = false, withZ = false) 

      n == min(mM,nM) && (return n)
      prnk = 0
      tol1 = max(atol1, rtol*opnorm(M,1))
      mrinf = 0
      nrinf = 0
      while m > 0 
         # Steps 1 & 2: Standard algorithm PREDUCE 
         ired = mrinf+1:mM
         jred = nrinf+1:nM
         τ, ρ = _preduce1!(n,m,p,view(M1,ired,jred),view(N1,ired,jred),Q,Z,tol1; fast = false, withQ = false, withZ = false)
         prnk += ρ+τ
         mrinf += ρ+τ
         nrinf += m
         n -= ρ
         m = ρ
         p -= τ 
      end
      return prnk+n
   end
end
