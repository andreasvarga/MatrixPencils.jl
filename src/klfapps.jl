"""
    isregular(M, N; atol1::Real = 0, atol2::Real = 0, rtol::Real=min(atol1,atol2)>0 ? 0 : n*ϵ) -> Bool

Test whether the linear pencil `M-λN` is regular (i.e., det(M-λN) !== 0). The underlying computational procedure
reduces the pencil `M-λN` to an appropriate Kronecker-like form (KLF), which provides information on the rank of `M-λN`. 

The keyword arguements `atol1`, `atol2` and `rtol` specify the absolute tolerance for the nonzero
elements of `M`, the absolute tolerance for the nonzero elements of `N`, and the relative tolerance for the nonzero elements of `M` and `N`, respectively. 
The default relative tolerance is `n*ϵ`, where `n` is the size of the smallest dimension of `M`, and `ϵ` is the 
machine epsilon of the element type of `M`. 
"""
function isregular(M::AbstractMatrix, N::AbstractMatrix; atol1::Real = zero(eltype(M)), atol2::Real = zero(eltype(M)), 
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

   """
   Step 0: Reduce to the standard form
   """
   Q, Z, n, m, p = _preduceBF!(M, N; atol = atol2, rtol = rtol, fast = false, withQ = false, withZ = false) 
   mrinf = 0
   nrinf = 0
   tol1 = max(atol1, rtol*opnorm(M,1))
   while m > 0
      """
      Steps 1 & 2: Standard algorithm PREDUCE
      """
      τ, ρ = _preduce1!(n,m,p,M,N,Q,Z,tol1; fast = false, roff = mrinf, coff = nrinf, withQ = false, withZ = false)
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
    pzeros(M, N; fast = false, atol1::Real = 0, atol2::Real = 0, rtol::Real=min(atol1,atol2)>0 ? 0 : n*ϵ) -> values

Return the (finite and infinite) Smith zeros of the linear pencil `M-λN`. 
The computation of the zeros is performed by reducing the pencil `M-λN` to an appropriate Kronecker-like form (KLF) 
exhibiting the spliting of the infinite and finite eigenvalue structures of the pencil `M-λN`. 
The multiplicities of infinite eigenvalues are in excess with one with respect to the multiplicities of infinite zeros. 
The reduction is performed using orthonal similarity transformations and involves rank decisions based on rank reevealing QR-decompositions with column pivoting, 
if `fast = true`, or, the more reliable, SVD-decompositions, if `fast = false`. For efficiency purposes, the reduction is only
partially performed, without accumulating the performed orthogonal transformations.

The keyword arguements `atol1`, `atol2`  and `rtol` specify the absolute tolerance for the nonzero
elements of `M`, the absolute tolerance for the nonzero elements of `N`, and the relative tolerance for the nonzero elements of `M` and `N`, respectively. 
The default relative tolerance is `n*ϵ`, where `n` is the size of the smallest dimension of `M`, and `ϵ` is the 
machine epsilon of the element type of `M`. 
"""
function pzeros(M::AbstractMatrix, N::AbstractMatrix; fast::Bool = true, 
   atol1::Real = zero(real(eltype(M))), atol2::Real = zero(real(eltype(M))), 
   rtol::Real = (min(size(M)...)*eps(real(float(one(eltype(M))))))*iszero(min(atol1,atol2)))

   mM, nM = size(M)
   (mM,nM) == size(N) || throw(DimensionMismatch("M and N must have the same dimensions"))
   isa(M,Adjoint) && (M = copy(M))
   isa(N,Adjoint) && (N = copy(N))
   T = promote_type(eltype(M), eltype(N))
   T <: BlasFloat || (T = promote_type(Float64,T))
   eltype(M) == T || (M = convert(Matrix{T},M))
   eltype(N) == T || (N = convert(Matrix{T},N))
   
   # # alternative computation
   # M, N, Q, Z, νr, μr, nf, ν, μ = klf_right(M, N, atol1 = atol1, atol2 = atol2, rtol = rtol,  
   # withQ = false, withZ = false, fast = false)

   # nzi = 0
   # nb = length(μ)
   # if nb > 0
   #    id = reverse([μ[1:1];μ[2:nb]-ν[1:nb-1]])
   #    for i = 2:nb
   #       nzi += id[i]*(i-1)
   #    end
   # end
   # mr = sum(νr)  
   # nr = sum(μr)
   # if1 = mr+1:mr+nf
   # jf1 = nr+1:nr+nf
   # return [eigvals(M[if1,jf1],N[if1,jf1]); Inf*ones(nzi) ]
   Q, Z, n, m, p = _preduceBF!(M, N; atol = atol2, rtol = rtol, fast = fast, withQ = false, withZ = false) 
   tol1 = max(atol1, rtol*opnorm(M,1))
   mrinf = 0
   nrinf = 0
   nzi = 0
   i = 0
   muim1 = 0
   while m > 0 
      """
      Steps 1 & 2: Standard algorithm PREDUCE 
      """
      ired = mrinf+1:mM
      jred = nrinf+1:nM
      τ, ρ = _preduce1!(n,m,p,view(M,ired,jred),view(N,ired,jred),Q,Z,tol1; fast = fast, withQ = false, withZ = false)
      i += 1
      i > 2 && (nzi += (muim1 - m)*(i-2))
      muim1 = ρ+τ
      mrinf += muim1
      nrinf += m
      n -= ρ
      m = ρ
      p -= τ 
   end
   nzi += muim1*(i-1)
   rtrail = 0
   ctrail = 0
   while p > 0
      """
      Step 3: Particular case of the dual PREDUCE algorithm 
      """
      ired = mrinf+1:mM-rtrail
      jred = nrinf+1:nM-ctrail
      ρ = _preduce4!(n, 0, p, view(M,ired,jred),view(N,ired,jred), Q, Z, tol1, fast = fast, withQ = false, withZ = false)
      rtrail += p
      ctrail += ρ
      n -= ρ
      p = ρ
   end
   if1 = mrinf+1:mrinf+n
   jf1 = nrinf+1:nrinf+n
   return [eigvals(M[if1,jf1],N[if1,jf1]); Inf*ones(real(T),nzi) ]
end
"""
    peigvals(M, N; fast = false, atol1::Real = 0, atol2::Real = 0, rtol::Real=min(atol1,atol2)>0 ? 0 : n*ϵ) -> values

Return the (finite and infinite) eigenvalues of the linear pencil `M-λN`. The computation of the eigenvalues
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
function peigvals(M::AbstractMatrix, N::AbstractMatrix; fast::Bool = true, 
   atol1::Real = zero(real(eltype(M))), atol2::Real = zero(real(eltype(M))), 
   rtol::Real = (min(size(M)...)*eps(real(float(one(eltype(M))))))*iszero(min(atol1,atol2)))

   mM, nM = size(M)
   (mM,nM) == size(N) || throw(DimensionMismatch("M and N must have the same dimensions"))
   isa(M,Adjoint) && (M = copy(M))
   isa(N,Adjoint) && (N = copy(N))
   T = promote_type(eltype(M), eltype(N))
   T <: BlasFloat || (T = promote_type(Float64,T))
   eltype(M) == T || (M = convert(Matrix{T},M))
   eltype(N) == T || (N = convert(Matrix{T},N))

   # # alternative computation 
   # M, N, Q, Z, νr, μr, nf, ν, μ = klf_right(M, N, atol1 = atol1, atol2 = atol2, rtol = rtol,  
   #                                          withQ = false, withZ = false, fast = false)
   # ni = 0
   # nb = length(μ)
   # if nb > 0
   #    id = reverse([μ[1:1];μ[2:nb]-ν[1:nb-1]])
   #    for i = 1:nb
   #       ni += id[i]*i
   #    end
   # end
   # mr = sum(νr)
   # nr = sum(μr)
   # if1 = mr+1:mr+nf
   # jf1 = nr+1:nr+nf
   # return [eigvals(M[if1,jf1],N[if1,jf1]); Inf*ones(real(T),ni) ]
   Q, Z, n, m, p = _preduceBF!(M, N; atol = atol2, rtol = rtol, fast = fast, withQ = false, withZ = false) 
   tol1 = max(atol1, rtol*opnorm(M,1))
   mrinf = 0
   nrinf = 0
   ni = 0
   i = 0
   muim1 = 0
   while m > 0 
      """
      Steps 1 & 2: Standard algorithm PREDUCE 
      """
      ired = mrinf+1:mM
      jred = nrinf+1:nM
      τ, ρ = _preduce1!(n,m,p,view(M,ired,jred),view(N,ired,jred),Q,Z,tol1; fast = fast, withQ = false, withZ = false)
      i += 1
      i > 1 && (ni += (muim1 - m)*(i-1))
      muim1 = ρ+τ
      mrinf += muim1
      nrinf += m
      n -= ρ
      m = ρ
      p -= τ 
   end
   ni += muim1*i
   rtrail = 0
   ctrail = 0
   while p > 0
      """
      Step 3: Particular case of the dual PREDUCE algorithm 
      """
      ired = mrinf+1:mM-rtrail
      jred = nrinf+1:nM-ctrail
      ρ = _preduce4!(n, 0, p, view(M,ired,jred),view(N,ired,jred), Q, Z, tol1, fast = fast, withQ = false, withZ = false)
      rtrail += p
      ctrail += ρ
      n -= ρ
      p = ρ
   end
   if1 = mrinf+1:mrinf+n
   jf1 = nrinf+1:nrinf+n
   return [eigvals(M[if1,jf1],N[if1,jf1]); Inf*ones(real(T),ni) ]
end
"""
    KRInfo
  
Kronecker-structure object definition. 

If `info::KRInfo` is the Kronecker-structure object, then:

`info.rki` is a vector which specifies the right Kronecker structure: there are `info.rki[i]` 
elementary Kronecker blocks of size `(i-1) x i`;

`info.lki` is a vector which specifies the left Kronecker structure: there are `info.lki[i]` 
elementary Kronecker blocks of size `i x (i-1)`;

`info.id` is a vector which specifies the infinite elementary divisors: there are `info.id[i]` 
infinite elementary divisors of degree `i`; 

`info.nf` is the number of finite eigenvalues.

Destructuring via iteration produces the components `info.rki`, `info.lki`, `info.id`, and `info.nf`.
"""
struct KRInfo
   rki::Vector{Int}
   lki::Vector{Int}
   id::Vector{Int}
   nf::Int
   function KRInfo(rki,lki,id,nf)
      if any(rki .< 0) || any(lki .< 0) || any(id .< 0) || nf < 0
         error("no negative components allowed")
      end
      new(rki,lki,id,nf)
   end
end
# iteration for destructuring into components
Base.iterate(info::KRInfo) = (info.rki, Val(:lki))
Base.iterate(info::KRInfo, ::Val{:lki}) = (info.lki, Val(:id))
Base.iterate(info::KRInfo, ::Val{:id}) = (info.id, Val(:nf))
Base.iterate(info::KRInfo, ::Val{:nf}) = (info.nf, Val(:done))
Base.iterate(info::KRInfo, ::Val{:done}) = nothing
"""
    pkstruct(M, N; fast = false, atol1::Real = 0, atol2::Real = 0, rtol::Real=min(atol1,atol2)>0 ? 0 : n*ϵ) -> KRInfo
  
Determine the Kronecker-structure information of the linear pencil `M-λN` and return an `KRInfo` object. 

The right Kronecker indices `rki`, left Kronecker indices `lki`, infinite elementary divisors `id` and the
number of finite eigenvalues `nf` can be obtained from `KRInfo` as `KRInfo.rki`, `KRInfo.lki`, `KRInfo.id` 
and `KRInfo.nf`, respectively.  
The determination of the Kronecker-structure information is performed by reducing the pencil `M-λN` to an 
appropriate Kronecker-like form (KLF) exhibiting all structural elements of the pencil `M-λN`.
The reduction is performed using orthonal similarity transformations and involves rank decisions based 
on rank reevealing QR-decompositions with column pivoting, if `fast = true`, or, the more reliable, 
SVD-decompositions, if `fast = false`. For efficiency purposes, the reduction is only
partially performed, without accumulating the performed orthogonal transformations. 

The right Kronecker indices are provided in the integer vector `rki`, whose `i`-th element `rki[i]` is 
the number of elementary Kronecker blocks of size `(i-1) x i`. The sum of elements of `rki` is the dimension of 
the right nullspace of the pencil `M-λN`. 

The left Kronecker indices are provided in the integer vector `lki`, whose `i`-th element `lki[i]` is 
the number of elementary Kronecker blocks of size `i x (i-1)`. The sum of elements of `lki` is the dimension of 
the left nullspace of the pencil `M-λN`. 

The infinite elementary divisors are provided in the integer vector `id`, whose `i`-th element `id[i]` is the number of
infinite elementary divisors of degree `i`. The sum `Sum_i(id[i]*i)` is the number of infinite eigenvalues of 
the pencil `M-λN`. 

The keyword arguements `atol1`, `atol2`  and `rtol` specify the absolute tolerance for the nonzero
elements of `M`, the absolute tolerance for the nonzero elements of `N`, and the relative tolerance for the nonzero elements of `M` and `N`, respectively. 
The default relative tolerance is `n*ϵ`, where `n` is the size of the smallest dimension of `M`, and `ϵ` is the 
machine epsilon of the element type of `M`. 
"""
function pkstruct(M::AbstractMatrix, N::AbstractMatrix; fast = false, atol1::Real = zero(eltype(M)), atol2::Real = zero(eltype(M)), 
   rtol::Real = (min(size(M)...)*eps(real(float(one(eltype(M))))))*iszero(min(atol1,atol2))) 

   mM, nM = size(M)
   (mM,nM) == size(N) || throw(DimensionMismatch("M and N must have the same dimensions"))
   isa(M,Adjoint) && (M = copy(M))
   isa(N,Adjoint) && (N = copy(N))
   T = promote_type(eltype(M), eltype(N))
   T <: BlasFloat || (T = promote_type(Float64,T))
   eltype(M) == T || (M = convert(Matrix{T},M))
   eltype(N) == T || (N = convert(Matrix{T},N))
   
   # # alternative computation
   # M, N, Q, Z, νr, μr, nf, ν, μ = klf_right(M, N, atol1 = atol1, atol2 = atol2, rtol = rtol,  
   # withQ = false, withZ = false, fast = false)
   # rki = μr-νr
   # nb = length(μ)
   # if nb > 0
   #    id = [μ[1:1];μ[2:end]-ν[1:end-1]]
   #    k = 0
   #    for i = 1:nb
   #        if id[i] == 0 
   #           k += 1
   #        else
   #           break
   #        end
   #    end
   #    id = reverse(id[k+1:end])
   # else
   #    id = μ
   # end
   # lki = reverse(ν-μ)
   # if nb > 0
   #    k = nb
   #    for i = nb:-1:1
   #       if lki[i] == 0 
   #          k -= 1
   #       else
   #          break
   #       end
   #    end
   #    lki = lki[1:k]
   # end
   # #return rki, id, nf, lki
   # return KRInfo(rki, lki, id, nf)

   maxmn = max(mM,nM)
   rki = Vector{Int}(undef,maxmn)
   id = Vector{Int}(undef,maxmn)
   lki = Vector{Int}(undef,maxmn)

   Q, Z, n, m, p = _preduceBF!(M, N; atol = atol2, rtol = rtol, fast = fast, withQ = false, withZ = false) 
   tol1 = max(atol1, rtol*opnorm(M,1))
   mrinf = 0
   nrinf = 0
   ni = 0
   i = 0
   muim1 = 0
   while m > 0 
      """
      Steps 1 & 2: Standard algorithm PREDUCE 
      """
      ired = mrinf+1:mM
      jred = nrinf+1:nM
      τ, ρ = _preduce1!(n,m,p,view(M,ired,jred),view(N,ired,jred),Q,Z,tol1; fast = fast, withQ = false, withZ = false)
      i += 1
      muim1 = ρ+τ
      rki[i] = m - muim1
      i > 1 && (id[i-1] = m - muim1)
      mrinf += muim1
      nrinf += m
      n -= ρ
      m = ρ
      p -= τ 
   end
   i > 0 && (id[i] = muim1)
   rtrail = 0
   ctrail = 0
   j = 0
   while p > 0
      """
      Step 3: Particular case of the dual PREDUCE algorithm 
      """
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
   return KRInfo(deltrz(rki[1:i]), deltrz(lki[1:j]), deltrz(id[1:i]), n)
end
function deltrz(ind)
   nb = length(ind)
   k = nb
   if nb > 0
      for i = nb:-1:1
         if ind[i] == 0 
            k -= 1
         else
            break
         end
      end
   end
   return  ind[1:k]
end
"""
    prank(M::AbstractMatrix, N::AbstractMatrix; fast = true, atol1::Real=0, atol2::Real=0, rtol::Real=min(atol1,atol2)>0 ? 0 : n*ϵ)

Compute the normal rank of a linear matrix pencil `M - λN`. If `fast = true`, the rank is evaluated 
by counting how many singular values of `M - γ N` have magnitude greater than `max(max(atol1,atol2), rtol*σ₁)`,
where `σ₁` is the largest singular value of `M - γ N` and `γ` is a randomly generated value. If `fast = false`, 
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
    To enforce compatibiöity with Julia 1.0, the newer function rank in Julia 1.1 has been explicitly included. 
"""
function prank(M::AbstractMatrix, N::AbstractMatrix; fast::Bool = true, 
   atol1::Real = zero(real(eltype(M))), atol2::Real = zero(real(eltype(M))), 
   rtol::Real = (min(size(M)...)*eps(real(float(one(eltype(M))))))*iszero(min(atol1,atol2)))

   mM, nM = size(M)
   (mM,nM) == size(N) || throw(DimensionMismatch("M and N must have the same dimensions"))
   isa(M,Adjoint) && (M = copy(M))
   isa(N,Adjoint) && (N = copy(N))
   T = promote_type(eltype(M), eltype(N))
   T <: BlasFloat || (T = promote_type(Float64,T))
   eltype(M) == T || (M = convert(Matrix{T},M))
   eltype(N) == T || (N = convert(Matrix{T},N))

   if fast
      nrmM = opnorm(M,1)
      nrmM == zero(nrmM) && (return rank(N, atol = atol2, rtol = rtol))
      nrmN = opnorm(N,1)
      nrmN == zero(nrmN) && (return rank(M, atol = atol1, rtol = rtol))
      scale = nrmM/nrmN*rand()
      return rank(M+scale*N, atol = max(atol1,atol2), rtol = rtol)
   else
      # # alternative (less efficient) computation using the KLF exhibiting the spliting of right-left Kronecker structures
      # M, N, Q, Z, ν, μ, n = klf_rlsplit(M, N; fast = false, finite_infinite = false, atol1 = atol1, atol2 = atol2, 
      #                                   rtol = rtol, withQ = false, withZ = false)
      # return sum(ν) + n
      Q, Z, n, m, p = _preduceBF!(M, N; atol = atol2, rtol = rtol, fast = false, withQ = false, withZ = false) 
      n == min(mM,nM) && (return n)
      prnk = 0
      tol1 = max(atol1, rtol*opnorm(M,1))
      mrinf = 0
      nrinf = 0
      while m > 0 
         """
         Steps 1 & 2: Standard algorithm PREDUCE 
         """
         ired = mrinf+1:mM
         jred = nrinf+1:nM
         τ, ρ = _preduce1!(n,m,p,view(M,ired,jred),view(N,ired,jred),Q,Z,tol1; fast = false, withQ = false, withZ = false)
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
function prank(M::AbstractMatrix, N::UniformScaling; fast = false, atol1::Real = zero(eltype(M)), atol2::Real = zero(eltype(M)), 
   rtol::Real = (min(size(M)...)*eps(real(float(one(eltype(M))))))*iszero(min(atol1,atol2)))
   nf = LinearAlgebra.checksquare(M) 
   if N.λ == 0
      prank = rank(M,atol = atol1, rtol = rtol)
   else
      prank = nf
   end
end
function prank(M::AbstractMatrix; fast = false, atol1::Real = zero(eltype(M)), atol2::Real = zero(eltype(M)), 
   rtol::Real = (min(size(M)...)*eps(real(float(one(eltype(M))))))*iszero(min(atol1,atol2)))
   prank = LinearAlgebra.checksquare(M) 
end
