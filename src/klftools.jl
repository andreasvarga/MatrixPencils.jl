"""
    klf_rlsplit(M, N; fast = true, finite_infinite = false, atol1 = 0, atol2 = 0, rtol, withQ = true, withZ = true) -> (F, G, Q, Z, ν, μ, n, m, p)

Reduce the linear pencil `M - λN` to an equivalent form `F - λG = Q'(M - λN)Z` using 
orthogonal or unitary transformation matrices `Q` and `Z` such that the transformed matrices `F` and `G` are in one of the
following Kronecker-like forms:

(1) if `finite_infinite = false`, then
 
                   | Mri-λNri |     *      |
          F - λG = |----------|------------|, 
                   |    O     | Mfl-λNfl   |
 
where the subpencil `Mri-λNri` contains the right Kronecker structure and infinite elementary 
divisors and the subpencil `Mfl-λNfl` contains the finite and left Kronecker structure of the pencil `M-λN`.

The full row rank subpencil `Mri-λNri` is in a staircase form. 
The `nb`-dimensional vectors `ν` and `μ` contain the row and, respectively, column dimensions of the blocks
of the staircase form  `Mri-λNri` such that `i`-th block has dimensions `ν[i] x μ[i]` and 
has full row rank. 
The difference `μ[i]-ν[i]` for `i = 1, 2, ..., nb` is the number of elementary Kronecker blocks of size `(i-1) x i`.
The difference `ν[i]-μ[i+1]` for `i = 1, 2, ..., nb` is the number of infinite elementary divisors of degree `i` 
(with `μ[nb+1] = 0`).

The full column rank subpencil `Mfl-λNfl` is in the standard form 
 
                     | B  | A-λE | 
          Mfl-λNfl = |----|------| ,        
                     | D  |  C   |

where `E` is an nxn non-singular upper triangular matrix, and `A`, `B`, `C`, `D` are `nxn`-, `nxm`-, `pxn`- and `pxm`-dimensional matrices,
respectively.  

(2) if `finite_infinite = true`, then
 
                   | Mrf-λNrf |     *      |
          F - λG = |----------|------------|, 
                   |    O     | Mil-λNil   |
 
where the subpencil `Mrf-λNrf` contains the right Kronecker and finite Kronecker structure and 
the subpencil `Mil-λNil` contains the left Kronecker structures and infinite elementary 
divisors of the pencil `M-λN`. 

The full row rank subpencil `Mrf-λNrf` is in the standard form 
 
                     | B  | A-λE | 
          Mrf-λNrf = |----|------| ,        
                     | D  |  C   |
                     
where `E` is an nxn non-singular upper triangular matrix, and `A`, `B`, `C`, `D` are `nxn`-, `nxm`-, `pxn`- and `pxm`-dimensional matrices,
respectively.  

The full column rank sub pencil `Mil-λNil` is in a staircase form. 
The `nb`-dimensional vectors `ν` and `μ` contain the row and, respectively, column dimensions of the blocks
of the staircase form  `Mil-λNil` such that the `i`-th block has dimensions `ν[i] x μ[i]` and has full column rank. 
The difference `ν[nb-j+1]-μ[nb-j+1]` for `j = 1, 2, ..., nb` is the number of elementary Kronecker blocks of size 
`j x (j-1)`. The difference `μ[nb-j+1]-ν[nb-j]` for `j = 1, 2, ..., nb` is the number of infinite elementary 
divisors of degree `j` (with `ν[0] = 0`).

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of `M`, the absolute tolerance for the nonzero elements of `N`,  and the relative tolerance 
for the nonzero elements of `M` and `N`. 

The reduction is performed using rank decisions based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.

The performed left orthogonal or unitary transformations are accumulated in the matrix `Q` if `withQ = true`. 
Otherwise, `Q` is set to `nothing`.   
The performed right orthogonal or unitary transformations are accumulated in the matrix `Z` if `withZ = true`. 
Otherwise, `Z` is set to `nothing`.  
"""
function klf_rlsplit(M::AbstractMatrix, N::AbstractMatrix; fast::Bool = true, finite_infinite::Bool = false, 
   atol1::Real = zero(real(eltype(M))), atol2::Real = zero(real(eltype(M))), 
   rtol::Real = (min(size(M)...)*eps(real(float(one(eltype(M))))))*iszero(min(atol1,atol2)), 
   withQ::Bool = true, withZ::Bool = true)
   
   mM, nM = size(M)
   (mM,nM) == size(N) || throw(DimensionMismatch("M and N must have the same dimensions"))
   isa(M,Adjoint) && (M = copy(M))
   isa(N,Adjoint) && (N = copy(N))
   T = promote_type(eltype(M), eltype(N))
   T <: BlasFloat || (T = promote_type(Float64,T))
   eltype(M) == T || (M = convert(Matrix{T},M))
   eltype(N) == T || (N = convert(Matrix{T},N))

   """
   Step 0: Reduce to the standard form
   """
   Q, Z, n, m, p = _preduceBF!(M, N; atol = atol2, rtol = rtol, fast = fast) 
   
   maxmn = max(mM,nM)
   μ = Vector{Int}(undef,maxmn)
   ν = Vector{Int}(undef,maxmn)
        
   # fast returns for null dimensions
   if mM == 0 && nM == 0
      return M, N, Q, Z, ν, μ, n, m, p
   end


   if finite_infinite
      if mM == 0
         return M, N, Q, Z, ν[1:0], μ[1:0], n, m, p
      elseif nM == 0
         ν[1] = mM
         μ[1] = 0
         p = 0
         return M, N, Q, Z, ν[1:1], μ[1:1], n, m, p
      end
   
      """
      Reduce M-λN to the KLF by splitting the right-finite and infinite-left structures

                       | Mrf - λ Nrf |     *        |
           M1 - λ N1 = |-------------|--------------|
                       |    0        | Mil -  λ Nil |
       
      where Mil -  λ Nil is in a staircase form.                 
      """
  
      mrinf = 0
      nrinf = 0
      rtrail = 0
      ctrail = 0
      i = 0
      tol1 = max(atol1, rtol*opnorm(M,1))     
      while p > 0
         """
         Step 1 & 2: Dual algorithm PREDUCE
         """
         τ, ρ  = _preduce2!(n,m,p,M,N,Q,Z,tol1; fast = fast, roff = mrinf, coff = nrinf, rtrail = rtrail, ctrail = ctrail, withQ = withQ, withZ = withZ)
         i += 1
         ν[i] = p
         μ[i] = ρ+τ
         ctrail += ρ+τ
         rtrail += p
         n -= ρ
         p = ρ
         m -= τ 
      end
      return M, N, Q, Z, reverse(ν[1:i]), reverse(μ[1:i]), n, m, p                                             
   else
      if mM == 0
         ν[1] = 0
         μ[1] = nM
         m = 0
         return M, N, Q, Z, ν[1:1], μ[1:1], n, m, p
      elseif nM == 0
         return M, N, Q, Z, ν[1:0], μ[1:0], n, m, p
      end
      """
      Reduce M-λN to the KLF by splitting the right-infinite and finite-left structures

                       | Mri - λ Nri |     *        |
           M1 - λ N1 = |-------------|--------------|
                       |    0        | Mfl -  λ Nfl |
      where Mri - λ Nri is in a staircase form.                 
      """
      mrinf = 0
      nrinf = 0
      i = 0
      tol1 = max(atol1, rtol*opnorm(M,1))
   
      while m > 0
         """
         Steps 1 & 2: Standard algorithm PREDUCE
         """
         τ, ρ = _preduce1!(n,m,p,M,N,Q,Z,tol1; fast = fast, roff = mrinf, coff = nrinf, withQ = withQ, withZ = withZ)
         i += 1
         ν[i] = ρ+τ
         μ[i] = m
         mrinf += ρ+τ
         nrinf += m
         n -= ρ
         m = ρ
         p -= τ 
      end
      return M, N, Q, Z, ν[1:i], μ[1:i], n, m, p                                             
   end
end
"""
    klf(M, N; fast = true, finite_infinite = false, atol1 = 0, atol2 = 0, rtol, withQ = true, withZ = true) -> (F, G, Q, Z, νr, μr, νi, nf, νl, μl)

Reduce the linear pencil `M - λN` to an equivalent form `F - λG = Q'(M - λN)Z` using 
orthogonal or unitary transformation matrices `Q` and `Z` such that the transformed matrices `F` and `G` are in the
following Kronecker-like form exhibiting the complete Kronecker structure:
 
                   |  Mr-λNr  |     *      |    *    |
                   |----------|------------|---------|
         F - λG =  |    O     | Mreg-λNreg |    *    |
                   |----------|------------|---------|
                   |    O     |     0      |  Ml-λNl |
 
The full row rank pencil `Mr-λNr` is in a staircase form, contains the right Kronecker indices 
of the pencil `M-λN`and has the form

         Mr-λNr  = | Br | Ar-λEr |,
          
where `Er` is upper triangular and nonsingular. 
The `nr`-dimensional vectors `νr` and `μr` contain the row and, respectively, column dimensions of the blocks
of the staircase form  `Mr-λNr` such that the `i`-th block has dimensions `νr[i] x μr[i]` and 
has full row rank. The difference `μr[i]-νr[i]` for `i = 1, 2, ..., nr` is the number of elementary Kronecker 
blocks of size `(i-1) x i`.

The pencil `Mreg-λNreg` is regular, contains the infinite and finite elementary divisors of `M-λN` and has the form

                       | Mi-λNi |    *    |
         Mreg-λNreg  = |--------|---------|, if `finite_infinite = false`, or the form
                       |   0    |  Mf-λNf |


                       | Mf-λNf |    *    |
         Mreg-λNreg  = |--------|---------|, if `finite_infinite = true`, 
                       |   0    |  Mi-λNi |
     
where: (1) `Mi-λNi`, in staircase form, contains the infinite elementary divisors of `M-λN` and 
`Ni` is upper triangular and nilpotent; 
(2) `Mf-λNf` contains the infinite elementary divisors of `M-λN` and 
`Nf` is upper triangular and nonsingular.
The `ni`-dimensional vector `νi` contains the dimensions of the square blocks of the staircase form  `Mi-λNi` 
such that the `i`-th block has dimensions `νi[i] x νi[i]`. The difference `νi[i]-νi[i-1]` for `i = 1, 2, ..., ni` 
is the number of infinite elementary divisors of degree `i` (with `νi[0] = 0`).

The full column rank pencil `Ml-λNl`, in a staircase form, contains the left Kronecker indices of `M-λN` and has the form

                   | Al-λEl |
         Ml-λNl  = |--------|,
                   |   Cl   |

where `El` is upper triangular and nonsingular. 
The `nl`-dimensional vectors `νl` and `μl` contain the row and, respectively, column dimensions of the blocks
of the staircase form  `Ml-λNl` such that the `j`-th block has dimensions `νl[j] x μl[j]` and has full column rank. 
The difference `νl[nl-j+1]-μl[nl-j+1]` for `j = 1, 2, ..., nl` is the number of elementary Kronecker blocks of size 
`j x (j-1)`.

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of `M`, the absolute tolerance for the nonzero elements of `N`,  and the relative tolerance 
for the nonzero elements of `M` and `N`. 
The reduction is performed using rank decisions based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.
The performed left orthogonal or unitary transformations are accumulated in the matrix `Q` if `withQ = true`. 
Otherwise, `Q` is set to `nothing`.   
The performed right orthogonal or unitary transformations are accumulated in the matrix `Z` if `withZ = true`. 
Otherwise, `Z` is set to `nothing`.  
"""
function klf(M::AbstractMatrix, N::AbstractMatrix; fast::Bool = true, finite_infinite::Bool = false, 
   atol1::Real = zero(real(eltype(M))), atol2::Real = zero(real(eltype(M))),
   rtol::Real = (min(size(M)...)*eps(real(float(one(eltype(M))))))*iszero(min(atol1,atol2)), withQ::Bool = true, withZ::Bool = true)
   
   mM, nM = size(M)
   (mM,nM) == size(N) || throw(DimensionMismatch("M and N must have the same dimensions"))
   isa(M,Adjoint) && (M = copy(M))
   isa(N,Adjoint) && (N = copy(N))
   T = promote_type(eltype(M), eltype(N))
   T <: BlasFloat || (T = promote_type(Float64,T))
   eltype(M) == T || (M = convert(Matrix{T},M))
   eltype(N) == T || (N = convert(Matrix{T},N))
   if finite_infinite
      """
      Reduce M-λN to the KLF exhibiting the right and finite structures
                       [ Mr - λ Nr  |   *        |     *        ]
           M1 - λ N1 = [    0       | Mf -  λ Nf |     *        ]
                       [    0       |    0       | Mli -  λ Nli ]
      """
      Q, Z, νr, μr, nf, ν, μ, tol1 = klf_right!(M, N, atol1 = atol1, atol2 = atol2, rtol = rtol,  
                                               withQ = withQ, withZ = withZ, fast = fast)
      if mM == 0 || nM == 0
         return  M, N, Q, Z, νr, μr, ν[1:0], nf, ν, μ
      end

      """
      Reduce Mli-λNli to the KLF exhibiting the infinite and left structures and update M1 - λ N1 to
                       [ Mr - λ Nr  |   *        |     *      |    *       ]
          M2 -  λ N2 = [    0       | Mf -  λ Nf |     *      |    *       ]
                       [    0       |    0       | Mi -  λ Ni |    *       ]
                       [    0       |    0       |    0       | Ml -  λ Nl ]
      """
      mr = sum(νr)+nf
      nr = sum(μr)+nf
      jM2 = nr+1:nr+sum(μ)
      M2 = view(M,:,jM2)
      N2 = view(N,:,jM2)
      if withZ
         Z2 = view(Z,:,jM2)
      else
         Z2 = nothing
      end
      νi, νl, μl = klf_left_refine!(ν, μ, M2, N2, Q, Z2, tol1, roff = mr,    
                                    withQ = withQ, withZ = withZ, fast = fast)
   else
      """
      Reduce M-λN to the KLF exhibiting the left and finite structures
                       [ Mri - λ Nri  |   *        |     *      ]
           M1 - λ N1 = [     0        | Mf -  λ Nf |     *      ]
                       [     0        |    0       | Ml -  λ Nl ]
      """
      Q, Z, ν, μ, nf, νl, μl, tol1 = klf_left!(M, N, atol1 = atol1, atol2 = atol2, rtol = rtol,  
                                               withQ = withQ, withZ = withZ, fast = fast)
      if mM == 0 || nM == 0
         return  M, N, Q, Z, ν, μ, ν[1:0], nf, νl, μl
      end
      """
      Reduce Mri-λNri to the KLF exhibiting the right and infinite structures and update M1 - λ N1 to
                       [ Mr - λ Nr  |   *        |     *      |    *       ]
          M2 -  λ N2 = [    0       | Mi -  λ Ni |     *      |    *       ]
                       [    0       |    0       | Mf -  λ Nf |    *       ]
                       [    0       |    0       |    0       | Ml -  λ Nl ]
      """
      iM11 = 1:sum(ν)
      M1 = view(M,iM11,:)
      N1 = view(N,iM11,:)
      νr, μr, νi = klf_right_refine!(ν, μ, M1, N1, Q, Z, tol1, ctrail = nM-sum(μ),   
                                          withQ = withQ, withZ = withZ, fast = fast)
   end
   return  M, N, Q, Z, νr, μr, νi, nf, νl, μl                                             
end
"""
    klf_right(M, N; fast = true, atol1 = 0, atol2 = 0, rtol, withQ = true, withZ = true) -> (F, G, Q, Z, νr, μr, nf, ν, μ)

Reduce the linear pencil `M - λN` to an equivalent form `F - λG = Q'(M - λN)Z` using 
orthogonal or unitary transformation matrices `Q` and `Z` such that the transformed matrices `F` and `G` are in the
following Kronecker-like form exhibiting the right and finite Kronecker structures:
 
                   |  Mr-λNr    |     *      |    *     |
                   |------------|------------|----------|
         F - λG =  |    O       |   Mf-λNf   |    *     |
                   |------------|------------|----------|
                   |    O       |     0      | Mli-λNli |            

The full row rank pencil `Mr-λNr`, in a staircase form, contains the right Kronecker indices of `M-λN` and has the form

         Mr-λNr  = | Br | Ar-λEr |,
                   
where `Er` is upper triangular and nonsingular. 
The `nr`-dimensional vectors `νr` and `μr` contain the row and, respectively, column dimensions of the blocks
of the staircase form  `Mr-λNr` such that `i`-th block has dimensions `νr[i] x μr[i]` and 
has full row rank. The difference `μr[i]-νr[i]` for `i = 1, 2, ..., nb` is the number of elementary Kronecker blocks
of size `(i-1) x i`.

The `nf x nf` pencil `Mf-λNf` is regular and contains the finite elementary divisors of `M-λN`. 
Nf is upper triangular and nonsingular.

The full column rank pencil `Mli-λNli` is in a staircase form, and contains the  left Kronecker indices 
and infinite elementary divisors of the pencil `M-λN`. 
The `nb`-dimensional vectors `ν` and `μ` contain the row and, respectively, column dimensions of the blocks
of the staircase form  `Mli-λNli` such that the `i`-th block has dimensions `ν[i] x μ[i]` and has full column rank. 
The difference `ν[nb-j+1]-μ[nb-j+1]` for `j = 1, 2, ..., nb` is the number of elementary Kronecker blocks of size 
`j x (j-1)`. The difference `μ[nb-j+1]-ν[nb-j]` for `j = 1, 2, ..., nb` is the number of infinite elementary 
divisors of degree `j` (with `ν[0] = 0`).

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of `M`, the absolute tolerance for the nonzero elements of `N`,  and the relative tolerance 
for the nonzero elements of `M` and `N`. 
The reduction is performed using rank decisions based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.
The performed left orthogonal or unitary transformations are accumulated in the matrix `Q` if `withQ = true`. 
Otherwise, `Q` is set to `nothing`.   
The performed right orthogonal or unitary or unitary transformations are accumulated in the matrix `Z` if `withZ = true`. 
Otherwise, `Z` is set to `nothing`.   

Note: If the pencil `M - λN` has full row rank, then the regular pencil `Mli-λNli` is in a staircase form with
square upper triangular diagonal blocks (i.e.,`μ[j] = ν[j]`), and the difference `ν[nb-j+1]-ν[nb-j]` for 
`j = 1, 2, ..., nb` is the number of infinite elementary divisors of degree `j` (with `ν[0] = 0`).
"""
function klf_right(M::AbstractMatrix, N::AbstractMatrix; fast::Bool = true, atol1::Real = zero(real(eltype(M))), atol2::Real = zero(real(eltype(M))), 
   rtol::Real = (min(size(M)...)*eps(real(float(one(eltype(M))))))*iszero(max(atol1,atol2)), withQ::Bool = true, withZ::Bool = true)
   
   mM, nM = size(M)
   (mM,nM) == size(N) || throw(DimensionMismatch("M and N must have the same dimensions"))
   isa(M,Adjoint) && (M = copy(M))
   isa(N,Adjoint) && (N = copy(N))
   T = promote_type(eltype(M), eltype(N))
   T <: BlasFloat || (T = promote_type(Float64,T))
   eltype(M) == T || (M = convert(Matrix{T},M))
   eltype(N) == T || (N = convert(Matrix{T},N))

   """
   Step 0: Reduce to the standard form
   """
   Q, Z, n, m, p = _preduceBF!(M, N; atol = atol2, rtol = rtol, fast = fast) 
   
   maxmn = max(mM,nM)
   μr = Vector{Int}(undef,maxmn)
   νr = Vector{Int}(undef,maxmn)
   μ = Vector{Int}(undef,maxmn)
   ν = Vector{Int}(undef,maxmn)
   nf = 0
     
   # fast returns for null dimensions
   if mM == 0 && nM == 0
      return M, N, Q, Z, νr, μr, nf, ν, μ
   elseif mM == 0
      νr[1] = 0
      μr[1] = nM
      return M, N, Q, Z, νr[1:1], μr[1:1], nf, ν[1:0], μ[1:0]
   elseif nM == 0
      ν[1] = mM
      μ[1] = 0
      return M, N, Q, Z, νr[1:0], μr[1:0], nf, ν[1:1], μ[1:1]
   end
   
   mrinf = 0
   nrinf = 0
   rtrail = 0
   ctrail = 0
   j = 0
   tol1 = max(atol1, rtol*opnorm(M,1))
   
   while p > 0
      """
      Step 1 & 2: Dual algorithm PREDUCE
      """
      τ, ρ  = _preduce2!(n,m,p,M,N,Q,Z,tol1; fast = fast, roff = mrinf, coff = nrinf, rtrail = rtrail, ctrail = ctrail, withQ = withQ, withZ = withZ)
      j += 1
      ν[j] = p
      μ[j] = ρ+τ
      ctrail += ρ+τ
      rtrail += p
      n -= ρ
      p = ρ
      m -= τ 
   end
   i = 0
   if m > 0
      imM11 = 1:mM-rtrail
      M11 = view(M,imM11,1:nM)
      N11 = view(N,imM11,1:nM)
   end
   while m > 0
      """
      Step 3: Particular case of the standard algorithm PREDUCE
      """
      ρ = _preduce3!(n, m, M11, N11, Q, Z, tol1, fast = fast, coff = nrinf, roff = mrinf, ctrail = ctrail,  withQ = withQ, withZ = withZ)
      i += 1
      νr[i] = ρ
      μr[i] = m
      mrinf += ρ
      nrinf += m
      n -= ρ
      m = ρ
   end
   
   return M, N, Q, Z, νr[1:i], μr[1:i], n, reverse(ν[1:j]), reverse(μ[1:j])
end
"""
    klf_left(M, N; fast = true, atol1 = 0, atol2 = 0, rtol, withQ = true, withZ = true) -> (F, G, Q, Z, ν, μ, nf, νl, μl)

Reduce the linear pencil `M - λN` to an equivalent form `F - λG = Q'(M - λN)Z` using 
orthogonal or unitary transformation matrices `Q` and `Z` such that the transformed matrices `F` and `G` are in the
following Kronecker-like form exhibiting the finite and left Kronecker structures:
 
                   |  Mri-λNri  |     *      |    *    |
                   |------------|------------|---------|
         F - λG =  |    O       |   Mf-λNf   |    *    |
                   |------------|------------|---------|
                   |    O       |     0      |  Ml-λNl |
               
 
The full row rank pencil `Mri-λNri` is in a staircase form, and contains the right Kronecker indices 
and infinite elementary divisors of the pencil `M-λN`.

The `nf x nf` pencil `Mf-λNf` is regular and contains the finite elementary divisors of `M-λN`. 
Nf is upper triangular and nonsingular.

The full column rank pencil `Ml-λNl`, in a staircase form, contains the left Kronecker indices of `M-λN` and has the form

                   | Al-λEl |
         Ml-λNl  = |--------|,
                   |   Cl   |

where `El` is upper triangular and nonsingular. 

The `nb`-dimensional vectors `ν` and `μ` contain the row and, respectively, column dimensions of the blocks
of the staircase form  `Mri-λNri` such that `i`-th block has dimensions `ν[i] x μ[i]` and 
has full row rank. 
The difference `μ[i]-ν[i]` for `i = 1, 2, ..., nb` is the number of elementary Kronecker blocks of size `(i-1) x i`.
The difference `ν[i]-μ[i+1]` for `i = 1, 2, ..., nb` is the number of infinite elementary divisors of degree `i` 
(with `μ[nb+1] = 0`).
The `nl`-dimensional vectors `νl` and `μl` contain the row and, respectively, column dimensions of the blocks
of the staircase form  `Ml-λNl` such that the `j`-th block has dimensions `νl[j] x μl[j]` and has full column rank. 
The difference `νl[nl-j+1]-μl[nl-j+1]` for `j = 1, 2, ..., nl` is the number of elementary Kronecker blocks of size 
`j x (j-1)`.

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of `M`, the absolute tolerance for the nonzero elements of `N`,  and the relative tolerance 
for the nonzero elements of `M` and `N`. 
The reduction is performed using rank decisions based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.
The performed left orthogonal or unitary transformations are accumulated in the matrix `Q` if `withQ = true`. 
Otherwise, `Q` is set to `nothing`.   
The performed right orthogonal or unitary transformations are accumulated in the matrix `Z` if `withZ = true`. 
Otherwise, `Z` is set to `nothing`.  

Note: If the pencil `M - λN` has full column rank, then the regular pencil `Mri-λNri` is in a staircase form with
square upper triangular diagonal blocks (i.e.,`μ[i] = ν[i]`), and the difference `ν[i+1]-ν[i]` for `i = 1, 2, ..., nb` 
is the number of infinite elementary divisors of degree `i` (with `ν[nb+1] = 0`).
"""
function klf_left(M::AbstractMatrix, N::AbstractMatrix; fast::Bool = true, atol1::Real = zero(real(eltype(M))), atol2::Real = zero(real(eltype(M))), 
   rtol::Real = (min(size(M)...)*eps(real(float(one(eltype(M))))))*iszero(max(atol1,atol2)), withQ::Bool = true, withZ::Bool = true)

   mM, nM = size(M)
   (mM,nM) == size(N) || throw(DimensionMismatch("M and N must have the same dimensions"))
   isa(M,Adjoint) && (M = copy(M))
   isa(N,Adjoint) && (N = copy(N))
   T = promote_type(eltype(M), eltype(N))
   T <: BlasFloat || (T = promote_type(Float64,T))
   eltype(M) == T || (M = convert(Matrix{T},M))
   eltype(N) == T || (N = convert(Matrix{T},N))

   """
   Step 0: Reduce to the standard form
   """
   Q, Z, n, m, p = _preduceBF!(M, N; atol = atol2, rtol = rtol, fast = fast) 

   maxmn = max(mM,nM)
   μ = Vector{Int}(undef,maxmn)
   ν = Vector{Int}(undef,maxmn)
   μl = Vector{Int}(undef,maxmn)
   νl = Vector{Int}(undef,maxmn)
   nf = 0


   # fast returns for null dimensions
   if mM == 0 && nM == 0
      return M, N, Q, Z, ν, μ, nf, νl, μl
   elseif mM == 0
      ν[1] = 0
      μ[1] = nM
      return M, N, Q, Z, ν[1:1], μ[1:1], nf, νl[1:0], μl[1:0]
   elseif nM == 0
      νl[1] = mM
      μl[1] = 0
      return M, N, Q, Z, ν[1:0], μ[1:0], nf, νl[1:1], μl[1:1]
   end
 
   mrinf = 0
   nrinf = 0
   i = 0
   tol1 = max(atol1, rtol*opnorm(M,1))

   while m > 0
      """
      Steps 1 & 2: Standard algorithm PREDUCE
      """
      τ, ρ = _preduce1!(n,m,p,M,N,Q,Z,tol1; fast = fast, roff = mrinf, coff = nrinf, withQ = withQ, withZ = withZ)
      i += 1
      ν[i] = ρ+τ
      μ[i] = m
      mrinf += ρ+τ
      nrinf += m
      n -= ρ
      m = ρ
      p -= τ 
   end
   rtrail = 0
   ctrail = 0
   j = 0
   nf = nM - nrinf
   p = mM - mrinf - nf
   while p > 0
      """
      Step 3: Particular case of the dual PREDUCE algorithm 
      """
      ρ = _preduce4!(nf, m, p, M, N, Q, Z, tol1, fast = fast, roff = mrinf, coff = nrinf, rtrail = rtrail, ctrail = ctrail, 
                     withQ = withQ, withZ = withZ) 
      j += 1
      νl[j] = p
      μl[j] = ρ
      rtrail += p
      ctrail += ρ
      nf -= ρ
      p = ρ
   end
 
   return M, N, Q, Z, ν[1:i], μ[1:i], nf, reverse(νl[1:j]), reverse(μl[1:j])
end
"""
    klf_right!(M, N; fast = true, roff = 0, coff = 0, rtrail = 0, ctrail = 0, atol1 = 0, atol2 = 0, rtol, withQ = true, withZ = true) -> (Q, Z, νr, μr, nf, ν, μ, tol)

Reduce the partitioned linear pencil `M - λN` (`*` stands for a not relevant subpencil)

              [   *         *        *  ] roff
     M - λN = [   0      M22-λN22    *  ] m
              [   0         0        *  ] rtrail
                coff        n     ctrail
  
to an equivalent form `F - λG = Q'(M - λN)Z` using orthogonal or unitary transformation matrices `Q` and `Z` 
such that the subpencil `M22 - λN22` is transformed into the following Kronecker-like form exhibiting 
its right and finite Kronecker structures:

 
                   |  Mr-λNr    |     *      |    *     |
                   |------------|------------|----------|
     F22 - λG22 =  |    O       |   Mf-λNf   |    *     |
                   |------------|------------|----------|
                   |    O       |     0      | Mli-λNli |

`F` and `G` are returned in `M` and `N`, respectively.                

The full row rank pencil `Mr-λNr`, in a staircase form, contains the right Kronecker indices of 
the subpencil `M22 - λN22` and has the form

         Mr-λNr  = | Br | Ar-λEr |,
                   
where `Er` is upper triangular and nonsingular. 
The `nr`-dimensional vectors `νr` and `μr` contain the row and, respectively, column dimensions of the blocks
of the staircase form  `Mr-λNr` such that `i`-th block has dimensions `νr[i] x μr[i]` and 
has full row rank. The difference `μr[i]-νr[i]` for `i = 1, 2, ..., nr` is the number of elementary Kronecker blocks
of size `(i-1) x i`.

The `nf x nf` pencil `Mf-λNf` is regular and contains the finite elementary divisors of the subpencil `M22 - λN22`. 
Nf is upper triangular and nonsingular.

The full column rank pencil `Mli-λNli` is in a staircase form, and contains the  left Kronecker indices 
and infinite elementary divisors of the subpencil `M22 - λN22`. 
The `nb`-dimensional vectors `ν` and `μ` contain the row and, respectively, column dimensions of the blocks
of the staircase form  `Mli-λNli` such that the `i`-th block has dimensions `ν[nb-i+1] x μ[nb-i+1]` and has full column rank. 
The difference `ν[nb-i+1]-μ[nb-i+1]` for `i = 1, 2, ..., nb` is the number of elementary Kronecker blocks of size 
`i x (i-1)`. The difference `μ[nb-i+1]-ν[nb-i]` for `i = 1, 2, ..., nb` is the number of infinite elementary 
divisors of degree `i` (with `ν[0] = 0`).

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of `M`, the absolute tolerance for the nonzero elements of `N`,  and the relative tolerance 
for the nonzero elements of `M` and `N`. The internally employed absolute tolerance  for the 
nonzero elements of `M` is returned in `tol`. 
The reduction is performed using rank decisions based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.
The performed left orthogonal or unitary transformations are accumulated in the matrix `Q` if `withQ = true`. 
Otherwise, `Q` is not modified.   
The performed right orthogonal or unitary transformations are accumulated in the matrix `Z` if `withZ = true`. 
Otherwise, `Z` is not modified.   

Note: If the subpencil `M22 - λN22` has full row rank, then the regular pencil `Mli-λNli` is in a staircase form with
square upper triangular diagonal blocks (i.e.,`μ[i] = ν[i]`), and the difference `ν[nb-i+1]-ν[nb-i]` for 
`i = 1, 2, ..., nb` is the number of infinite elementary divisors of degree `i` (with `ν[0] = 0`).
"""
function klf_right!(M::AbstractMatrix{T1}, N::AbstractMatrix{T1}; fast::Bool = true, atol1::Real = zero(real(eltype(M))), atol2::Real = zero(real(eltype(M))), 
                   rtol::Real = (min(size(M)...)*eps(real(float(one(eltype(M))))))*iszero(max(atol1,atol2)), 
                   roff::Int = 0, coff::Int = 0, rtrail::Int = 0, ctrail::Int = 0, withQ::Bool = true, withZ::Bool = true) where T1 <: BlasFloat
   mM, nM = size(M)
   (mM,nM) == size(N) || throw(DimensionMismatch("M and N must have the same dimensions"))
   (!isa(M,Adjoint) && !isa(N,Adjoint)) || error("No adjoint inputs are supported")
   """
   Step 0: Reduce M22-λN22 to the standard form
   """
   Q, Z, n, m, p = _preduceBF!(M, N; atol = atol2, rtol = rtol, fast = fast, roff = roff, coff = coff, rtrail = rtrail, ctrail = ctrail, withQ = withQ, withZ = withQ) 
   
   maxmn = max(mM,nM)
   μr = Vector{Int}(undef,maxmn)
   νr = Vector{Int}(undef,maxmn)
   μ = Vector{Int}(undef,maxmn)
   ν = Vector{Int}(undef,maxmn)
   nf = 0
   tol1 = atol1
 
   # fast returns for null dimensions
   if mM == 0 && nM == 0
      return Q, Z, νr, μr, nf, ν, μ, tol1
   elseif mM == 0
      νr[1] = 0
      μr[1] = nM
      return Q, Z, νr[1:1], μr[1:1], nf, ν[1:0], μ[1:0], tol1
   elseif nM == 0
      ν[1] = mM
      μ[1] = 0
      return Q, Z, νr[1:0], μr[1:0], nf, ν[1:1], μ[1:1], tol1
   end

   j = 0
   tol1 = max(atol1, rtol*opnorm(M,1))
   
   while p > 0
      """
      Steps 1 & 2: Dual algorithm PREDUCE
      """
      τ, ρ = _preduce2!(n,m,p,M,N,Q,Z,tol1; fast = fast, roff = roff, coff = coff, rtrail = rtrail, ctrail = ctrail, withQ = withQ, withZ = withZ)
      j += 1
      ν[j] = p
      μ[j] = ρ+τ
      ctrail += τ+ρ
      rtrail += p
      n -= ρ
      p = ρ
      m -= τ 
   end
   i = 0
   if m > 0
      imM11 = 1:mM-rtrail
      M11 = view(M,imM11,1:nM)
      N11 = view(N,imM11,1:nM)
   end
   while m > 0
      """
      Step 3: Particular case of the standard algorithm PREDUCE
      """
      ρ = _preduce3!(n, m, M11, N11, Q, Z, tol1, fast = fast, coff = coff, roff = roff, ctrail = ctrail,  withQ = withQ, withZ = withZ)
      i += 1
      νr[i] = ρ
      μr[i] = m
      roff += ρ
      coff += m
      n -= ρ
      m = ρ
   end

   return Q, Z, νr[1:i], μr[1:i], n, reverse(ν[1:j]), reverse(μ[1:j]), tol1
end
"""

     klf_right_refine!(ν, μ, M, N, tol; fast = true, roff = 0, coff = 0, rtrail = 0, ctrail = 0, atol1 = 0, atol2 = 0, rtol, withQ = true, withZ = true) -> (νr, μr, νi)

Reduce the partitioned linear pencil `M - λN` (`*` stands for a not relevant subpencil)

             [   *         *        *  ] roff
    M - λN = [   0      Mri-λNri    *  ] mri
             [   0         0        *  ] rtrail
               coff      nri     ctrail

to an equivalent form `F - λG = Q'(M - λN)Z` using orthogonal or unitary transformation matrices `Q` and `Z` 
such that the full row rank subpencil `Mri-λNri`is transformed into the 
following Kronecker-like form  exhibiting its right and infinite Kronecker structures:

                  |  Mr-λNr    |     *      |
     Fri - λGri = |------------|------------|
                  |    O       |   Mi-λNi   |

The full row rank pencil `Mri-λNri` is in a staircase form and the `nb`-dimensional vectors `ν` and `μ` 
contain the row and, respectively, column dimensions of the blocks of the staircase form  `Mri-λNri` such that 
the `i`-th block has dimensions `ν[i] x μ[i]` and has full row rank. The matrix Mri has full row rank and 
the trailing `μ[2]+...+μ[nb]` columns of `Nri` form a full column rank submatrix. 

The full row rank pencil `Mr-λNr` is in a staircase form, contains the right Kronecker indices 
of the pencil `M-λN`and has the form

      Mr-λNr  = | Br | Ar-λEr |,
   
where `Er` is upper triangular and nonsingular. 
The `nr`-dimensional vectors `νr` and `μr` contain the row and, respectively, column dimensions of the blocks
of the staircase form  `Mr-λNr` such that the `i`-th block has dimensions `νr[i] x μr[i]` and 
has full row rank. The difference `μr[i]-νr[i] = μ[i]-ν[i]` for `i = 1, 2, ..., nr` is the number of 
elementary Kronecker blocks of size `(i-1) x i`.

The regular pencil `Mi-λNi`is in a staircase form, contains the infinite elementary divisors of `Mri-λNri` 
and `Ni` is upper triangular and nilpotent. The `ni`-dimensional vector `νi` contains the dimensions of the 
square diagonal blocks of the staircase form  `Mi-λNi` such that the `i`-th block has dimensions `νi[i] x νi[i]`. 
The difference `νi[ni-i+1]-νi[ni-i] = ν[i]-μ[i+1]` for `i = 1, 2, ..., ni` is the number of infinite elementary 
divisors of degree `i` (with `νi[0] = 0` and `μ[nb+1] = 0`).

`F` and `G` are returned in `M` and `N`, respectively.  

The performed left orthogonal or unitary transformations are accumulated in the matrix `Q` if `withQ = true` or  
`Q` is unchanged if `withQ = false` .  
The performed right orthogonal or unitary transformations are accumulated in the matrix `Z` if `withZ = true` or  
`Z` is unchanged if `withZ = false` .  
"""
function klf_right_refine!(ν::Vector{Int}, μ::Vector{Int}, M::AbstractMatrix{T1}, N::AbstractMatrix{T1}, Q::Union{AbstractMatrix{T1},Nothing},
   Z::Union{AbstractMatrix{T1},Nothing}, tol::Real; fast::Bool = true, roff::Int = 0, coff::Int = 0, rtrail::Int = 0, ctrail::Int = 0, 
   withQ::Bool = true, withZ::Bool = true) where T1 <: BlasFloat
   nb = length(ν)
   nb == length(μ) || throw(DimensionMismatch("ν and μ must have the same lengths"))
  
   nb == 0 && (return ν[1:0], ν[1:0], ν[1:0])
   
   mri = sum(ν)
   nri = sum(μ)
   mM = roff + mri + rtrail
   nM = coff + nri + ctrail

   m = μ[1]
   n = nri-m
   p = mri-n

   if n > 0
      """
      Step 0: Reduce Nri = [ 0 E11] to standard form, where E11 is full column rank
                           [ 0 0  ]
      """
      it = roff+1:roff+mri
      jt = coff+m+1:coff+nri
      tau = similar(N,n)
      E11 = view(N,it,jt)
      LinearAlgebra.LAPACK.geqrf!(E11,tau)
      eltype(M) <: Complex ? tran = 'C' : tran = 'T'
      LinearAlgebra.LAPACK.ormqr!('L',tran,E11,tau,view(M,it,coff+1:nM))
      withQ && LinearAlgebra.LAPACK.ormqr!('R','N',E11,tau,view(Q,:,it)) 
      LinearAlgebra.LAPACK.ormqr!('L',tran,E11,tau,view(N,it,coff+nri+1:nM))
      triu!(E11)
   end
   
   μr = Vector{Int}(undef,nb)
   νr = Vector{Int}(undef,nb)
   νi = Vector{Int}(undef,nb)
  
   mrinf = 0
   nrinf = 0
   j = 0

   while p > 0
      """
      Steps 1 & 2: Dual algorithm PREDUCE to separate right-finite and infinite-left parts.

      Reduce Mri-λNri to [ Mr1-λNr1 * ; 0 Mi-λNi], where Mr1-λNr1 contains the right Kronecker structure and the empty
      finite part and Mi-λNi contains the infinite elementary divisors and the empty left Kronecker structure.
      """
      τ, ρ = _preduce2!(n,m,p,M,N,Q,Z,tol; fast = fast, roff = roff, coff = coff, rtrail = rtrail, ctrail = ctrail, withQ = withQ, withZ = withZ)
      ρ+τ == p || error("The reduced pencil must not have left structure: try to adjust the tolerances")
      j += 1
      νi[j] = p
      ctrail += p
      rtrail += p
      n -= ρ
      p = ρ
      m -= τ 
   end

   i = 0
   if m > 0
      imM11 = 1:mM-rtrail
      M11 = view(M,imM11,1:nM)
      N11 = view(N,imM11,1:nM)
   end
   while m > 0
      """
      Step 3: Particular form of the standard algorithm PREDUCE to reduce Mr1-λNr1 to Mr-λNr in  staircase form. 
      """
      ρ = _preduce3!(n, m, M11, N11, Q, Z, tol, fast = fast, coff = coff, roff = roff, ctrail = ctrail, withQ = withQ, withZ = withZ)
      i += 1
      νr[i] = ρ
      μr[i] = m
      roff += ρ
      coff += m
      n -= ρ
      m = ρ
   end
   
   return νr[1:i], μr[1:i], reverse(νi[1:j])
end
"""
    klf_left!(M, N; fast = true, roff = 0, coff = 0, rtrail = 0, ctrail = 0, atol1 = 0, atol2 = 0, rtol, withQ = true, withZ = true) -> (Q, Z, ν, μ, nf, νl, μl, tol)

Reduce the partitioned linear pencil `M - λN` (`*` stands for a not relevant subpencil)

              [   *         *        *  ] roff
     M - λN = [   0      M22-λN22    *  ] m
              [   0         0        *  ] rtrail
                coff        n     ctrail
  
to an equivalent form `F - λG = Q'(M - λN)Z` using orthogonal or unitary transformation matrices `Q` and `Z` 
such that the subpencil `M22 - λN22`  is transformed into the following Kronecker-like form exhibiting 
its finite and left Kronecker structures
 
                   |  Mri-λNri  |     *      |    *    |
                   |------------|------------|---------|
     F22 - λG22 =  |    O       |   Mf-λNf   |    *    |
                   |------------|------------|---------|
                   |    O       |     0      |  Ml-λNl |

`F` and `G` are returned in `M` and `N`, respectively.                
 
The full row rank pencil `Mri-λNri` is in a staircase form, and contains the right Kronecker indices 
and infinite elementary divisors of the subpencil `M22 - λN22`. 
The `nb`-dimensional vectors `ν` and `μ` contain the row and, respectively, column dimensions of the blocks
of the staircase form  `Mri-λNri` such that `i`-th block has dimensions `ν[i] x μ[i]` and 
has full row rank. 
The difference `μ[i]-ν[i]` for `i = 1, 2, ..., nb` is the number of elementary Kronecker blocks of size `(i-1) x i`.
The difference `ν[i]-μ[i+1]` for `i = 1, 2, ..., nb` is the number of infinite elementary divisors of degree `i` 
(with `μ[nb+1] = 0`).

The `nf x nf` pencil `Mf-λNf` is regular and contains the finite elementary divisors of `M-λN`. 
Nf is upper triangular and nonsingular.

The full column rank pencil `Ml-λNl`, in a staircase form, contains the left Kronecker indices of `M-λN` and has the form

                   | Al-λEl |
         Ml-λNl  = |--------|,
                   |   Cl   |

where `El` is upper triangular and nonsingular. 
The `nl`-dimensional vectors `νl` and `μl` contain the row and, respectively, column dimensions of the blocks
of the staircase form  `Ml-λNl` such that `j`-th block has dimensions `νl[nl-j+1] x μl[nl-j+1]` and has full column rank. 
The difference `νl[nl-j+1]-μl[nl-j+1]` for `j = 1, 2, ..., nl` is the number of elementary Kronecker blocks of size 
`j x (j-1)`.

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of `M`, the absolute tolerance for the nonzero elements of `N`,  and the relative tolerance 
for the nonzero elements of `M` and `N`. The internally employed absolute tolerance  for the 
nonzero elements of `M` is returned in `tol`. 
The reduction is performed using rank decisions based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.
The performed left orthogonal or unitary transformations are accumulated in the matrix `Q` if `withQ = true`. 
Otherwise, `Q` is not modified.   
The performed right orthogonal or unitary transformations are accumulated in the matrix `Z` if `withZ = true`. 
Otherwise, `Z` is not modified.  

Note: If the pencil `M22 - λN22` has full column rank, then the regular pencil `Mri-λNri` is in a staircase form with
square diagonal blocks (i.e.,`μ[i] = ν[i]`), and the difference `ν[i]-ν[i+1]` for `i = 1, 2, ..., nb` 
is the number of infinite elementary divisors of degree `i` (with `ν[nb+1] = 0`).
"""
function klf_left!(M::AbstractMatrix{T1}, N::AbstractMatrix{T1}; fast::Bool = true, atol1::Real = zero(real(eltype(M))), atol2::Real = zero(real(eltype(M))), 
   rtol::Real = (min(size(M)...)*eps(real(float(one(eltype(M))))))*iszero(max(atol1,atol2)), 
   roff::Int = 0, coff::Int = 0, rtrail::Int = 0, ctrail::Int = 0, withQ::Bool = true, withZ::Bool = true) where T1 <: BlasFloat
   mM, nM = size(M)
   (mM,nM) == size(N) || throw(DimensionMismatch("M and N must have the same dimensions"))
   (!isa(M,Adjoint) && !isa(N,Adjoint)) || error("No adjoint inputs are supported")

   """
   Step 0: Reduce M22-λN22 to the standard form
   """
   Q, Z, n, m, p = _preduceBF!(M, N; atol = atol2, rtol = rtol, fast = fast, roff = roff, coff = coff, rtrail = rtrail, ctrail = ctrail, withQ = withQ, withZ = withQ) 

   maxmn = max(mM,nM)
   μ = Vector{Int}(undef,maxmn)
   ν = Vector{Int}(undef,maxmn)
   μl = Vector{Int}(undef,maxmn)
   νl = Vector{Int}(undef,maxmn)
   nf = 0
   tol = atol1


   # fast returns for null dimensions
   if mM == 0 && nM == 0
      return Q, Z, ν, μ, nf, νl, μl, tol
   elseif mM == 0
      ν[1] = 0
      μ[1] = nM
      return Q, Z, ν[1:1], μ[1:1], nf, νl[1:0], μl[1:0], tol
   elseif nM == 0
      νl[1] = mM
      μl[1] = 0
      return Q, Z, ν[1:0], μ[1:0], nf, νl[1:1], μl[1:1], tol
   end

   mrinf = 0
   nrinf = 0
   i = 0
   tol1 = max(atol1, rtol*opnorm(M,1))

   while m > 0
      """
      Steps 1 & 2: Standard algorithm PREDUCE
      """
      τ, ρ = _preduce1!(n,m,p,M,N,Q,Z,tol1; fast = fast, roff = roff, coff = coff, rtrail = rtrail, ctrail = ctrail, withQ = withQ, withZ = withZ)
      i += 1
      ν[i] = ρ+τ
      μ[i] = m
      roff += τ+ρ
      coff += m
      n -= ρ
      m = ρ
      p -= τ 
   end
   j = 0
   while p > 0
      """
      Step 3: Particular form of the dual algorithm PREDUCE
      """
      ρ = _preduce4!(n, 0, p, M, N, Q, Z, tol1, fast = fast, roff = roff, coff = coff, rtrail = rtrail, ctrail = ctrail, 
                     withQ = withQ, withZ = withZ) 
      j += 1
      νl[j] = p
      μl[j] = ρ
      rtrail += p
      ctrail += ρ
      n -= ρ
      p = ρ
   end
 
   return Q, Z, ν[1:i], μ[1:i], n, reverse(νl[1:j]), reverse(μl[1:j]), tol1
end
"""
    klf_left_refine!(ν, μ, M, N, tol; fast = true, roff = 0, coff = 0, rtrail = 0, ctrail = 0, atol1 = 0, atol2 = 0, rtol, withQ = true, withZ = true) -> (νi, νl, μl)

Reduce the partitioned linear pencil `M - λN` (`*` stands for a not relevant subpencil)

             [   *         *        *  ] roff
    M - λN = [   0      Mli-λNli    *  ] mli
             [   0         0        *  ] rtrail
               coff      nli     ctrail

to an equivalent form `F - λG = Q'(M - λN)Z` using orthogonal or unitary transformation matrices `Q` and `Z` 
such that the full column rank subpencil `Mli-λNli`is transformed into the 
following Kronecker-like form  exhibiting its infinite and left Kronecker structures:

                  |  Mi-λNi    |     *      |
     Fil - λGil = |------------|------------|
                  |    O       |   Ml-λNl   |

The full column rank pencil `Mli-λNli` is in a staircase form and the `nb`-dimensional vectors `ν` and `μ` 
contain the row and, respectively, column dimensions of the blocks of the staircase form  `Mli-λNli` such that 
the `i`-th block has dimensions `ν[i] x μ[i]` and has full column rank. The matrix Mli has full column rank and 
the leading `μ[1]+...+μ[nb-1]` columns of `Nli` form a full row rank submatrix. 

The regular pencil `Mi-λNi`is in a staircase form, contains the infinite elementary divisors of `Mli-λNli` 
and `Ni` is upper triangular and nilpotent. The `ni`-dimensional vector `νi` contains the dimensions of the 
square diagonal blocks of the staircase form  `Mi-λNi` such that the `i`-th block has dimensions `νi[i] x νi[i]`. 
The difference `νi[i]-νi[i-1] = ν[ni-i+1]-μ[ni-i+1]` for `i = 1, 2, ..., ni` is the number of infinite elementary 
divisors of degree `i` (with `νi[0] = 0` and `μ[nb+1] = 0`).

The full column rank pencil `Mli-λNli` is in a staircase form, and contains the  left Kronecker indices 
and infinite elementary divisors of the subpencil `M22 - λN22`. 
The `nb`-dimensional vectors `ν` and `μ` contain the row and, respectively, column dimensions of the blocks
of the staircase form  `Mli-λNli` such that the `i`-th block has dimensions `ν[nb-i+1] x μ[nb-i+1]` and has full column rank. 
The difference `ν[nb-i+1]-μ[nb-i+1]` for `i = 1, 2, ..., nb` is the number of elementary Kronecker blocks of size 
`i x (i-1)`. The difference `μ[nb-i+1]-ν[nb-i]` for `i = 1, 2, ..., nb` is the number of infinite elementary 
divisors of degree `i` (with `ν[0] = 0`).

The full column rank pencil `Ml-λNl`, in a staircase form, contains the left Kronecker indices of `M-λN` and has the form

               | Al-λEl |
     Ml-λNl  = |--------|,
               |   Cl   |

where `El` is upper triangular and nonsingular. 
The `nl`-dimensional vectors `νl` and `μl` contain the row and, respectively, column dimensions of the blocks
of the staircase form  `Ml-λNl` such that the `j`-th block has dimensions `νl[j] x μl[j]` and has full column rank. 
The difference `νl[nl-j+1]-μl[nl-j+1] = ν[nl-j+1]-μ[nl-j+1]` for `j = 1, 2, ..., nl` is the number of elementary 
Kronecker blocks of size `j x (j-1)`.

`F` and `G` are returned in `M` and `N`, respectively.  

The performed left orthogonal or unitary transformations are accumulated in the matrix `Q` if `withQ = true` or  
`Q` is unchanged if `withQ = false` .  
The performed right orthogonal or unitary transformations are accumulated in the matrix `Z` if `withZ = true` or  
`Z` is unchanged if `withZ = false` .  
"""
function klf_left_refine!(ν::Vector{Int}, μ::Vector{Int}, M::AbstractMatrix{T1}, N::AbstractMatrix{T1}, Q::Union{AbstractMatrix{T1},Nothing},
   Z::Union{AbstractMatrix{T1},Nothing}, tol::Real; fast::Bool = true, roff::Int = 0, coff::Int = 0, rtrail::Int = 0, ctrail::Int = 0, 
   withQ::Bool = true, withZ::Bool = true) where T1 <: BlasFloat
   nb = length(ν)
   nb == length(μ) || throw(DimensionMismatch("ν and μ must have the same lengths"))
 
   nb == 0 && (return ν[1:0], ν[1:0], ν[1:0])

   mli = sum(ν)
   nli = sum(μ)
   mM = roff + mli + rtrail
   nM = coff + nli + ctrail
   p = ν[nb]
   n = mli-p
   m = nli-n

   if n > 0
      """
      Step 0: Reduce N = [ 0 E11] to standard form, where E11 is full row rank
                         [ 0 0  ]
      """
      it = roff+1:roff+n
      jt = coff+1:coff+nli
      tau = similar(N,n)
      E11 = view(N,it,jt)
      LinearAlgebra.LAPACK.gerqf!(E11,tau)
      eltype(M) <: Complex ? tran = 'C' : tran = 'T'
      LinearAlgebra.LAPACK.ormrq!('R',tran,E11,tau,view(M,1:mM,jt))
      withZ && LinearAlgebra.LAPACK.ormrq!('R',tran,E11,tau,view(Z,:,jt)) 
      LinearAlgebra.LAPACK.ormrq!('R',tran,E11,tau,view(N,1:roff,jt))
      triu!(E11,m)
   end
       
   μl = Vector{Int}(undef,nb)
   νl = Vector{Int}(undef,nb)
   νi = Vector{Int}(undef,nb)
 
   i = 0
   while m > 0
      """
      Steps 1 & 2: Standard algorithm
      """
      τ, ρ = _preduce1!(n,m,p,M,N,Q,Z,tol; fast = fast, roff = roff, coff = coff, rtrail = rtrail, ctrail = ctrail, withQ = withQ, withZ = withZ)
      ρ+τ == m || error("The reduced pencil must not have right structure: try to adjust the tolerances")
      i += 1
      νi[i] = m
      roff += m
      coff += m
      n -= ρ
      m = ρ
      p -= τ 
   end
   j = 0
   while p > 0
      """
      Step 3: Dual algorithm
      """
      ρ = _preduce4!(n, 0, p, M, N, Q, Z, tol, fast = fast, roff = roff, coff = coff, rtrail = rtrail, ctrail = ctrail, 
                     withQ = withQ, withZ = withZ) 
      j += 1
      νl[j] = p
      μl[j] = ρ
      rtrail += p
      ctrail += ρ
      n -= ρ
      p = ρ
   end
 
   return reverse(νi[1:i]), reverse(νl[1:j]), reverse(μl[1:j])
end
