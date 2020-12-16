"""
    preduceBF(M, N; fast = true, atol = 0, rtol, withQ = true, withZ = true) -> (F, G, Q, Z, n, m, p)

Reduce the matrix pencil `M - λN` to an equivalent form `F - λG = Q'*(M - λN)*Z` using 
orthogonal or unitary transformation matrices `Q` and `Z` such that the pencil `M - λN` is transformed 
into the following standard form
 
                   | B  | A-λE | 
          F - λG = |----|------| ,        
                   | D  |  C   |

where `E` is an `nxn` non-singular matrix, and `A`, `B`, `C`, `D` are `nxn`-, `nxm`-, `pxn`- and `pxm`-dimensional matrices,
respectively. The order `n` of `E` is equal to the numerical rank of `N` determined using the absolute tolerance `atol` and 
relative tolerance `rtol`. `M` and `N` are overwritten by `F` and `G`, respectively. 

If `fast = true`, `E` is determined upper triangular using a rank revealing QR-decomposition with column pivoting of `N` 
and `n` is evaluated as the number of nonzero diagonal elements of the `R` factor, whose magnitudes are greater than 
`tol = max(atol,abs(R[1,1])*rtol)`. 
If `fast = false`,  `E` is determined diagonal using a rank revealing SVD-decomposition of `N` and 
`n` is evaluated as the number of singular values greater than `tol = max(atol,smax*rtol)`, where `smax` 
is the largest singular value. 
The rank decision based on the SVD-decomposition is generally more reliable, but the involved computational effort is higher.

The performed left orthogonal or unitary transformations are accumulated in the matrix `Q` if `withQ = true`. 
Otherwise, `Q` is set to `nothing`.   
The performed right orthogonal or unitary transformations are accumulated in the matrix `Z` if `withZ = true`. 
Otherwise, `Z` is set to `nothing`.  
"""
function preduceBF(M::AbstractMatrix, N::AbstractMatrix; fast::Bool = true, 
                   atol::Real = zero(real(eltype(M))),  
                   rtol::Real = (min(size(M)...)*eps(real(float(one(eltype(M))))))*iszero(atol), 
                   withQ::Bool = true, withZ::Bool = true)
   # In interest of performance, no dimensional checks are performed                  
   mM, nM = size(M)
   (mM,nM) == size(N) || throw(DimensionMismatch("M and N must have the same dimensions"))
   T = promote_type(eltype(M), eltype(N))
   T <: BlasFloat || (T = promote_type(Float64,T))
   M1 = copy_oftype(M,T)
   N1 = copy_oftype(N,T)

   withQ ? (Q = Matrix{T}(I,mM,mM)) : (Q = nothing)
   withZ ? (Z = Matrix{T}(I,nM,nM)) : (Z = nothing)

   # Step 0: Reduce to the standard form
   n, m, p = _preduceBF!(M1, N1, Q, Z; atol = atol, rtol = rtol, fast = fast, withQ = withQ, withZ = withZ) 

   return M1, N1, Q, Z, n, m, p
end
"""
    klf_rlsplit(M, N; fast = true, finite_infinite = false, atol1 = 0, atol2 = 0, rtol, withQ = true, withZ = true) -> (F, G, Q, Z, ν, μ, n, m, p)

Reduce the matrix pencil `M - λN` to an equivalent form `F - λG = Q'*(M - λN)*Z` using 
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

The full column rank subpencil `Mfl-λNfl` is in the form 
 
                     | A-λE | 
          Mfl-λNfl = |------| ,        
                     |  C   |

where `E` is an nxn non-singular upper triangular matrix, and `A` and `C` are `nxn`- and `pxn`-dimensional matrices,
respectively, and `m = 0`. 

(2) if `finite_infinite = true`, then
 
                   | Mrf-λNrf |     *      |
          F - λG = |----------|------------|, 
                   |    O     | Mil-λNil   |
 
where the subpencil `Mrf-λNrf` contains the right Kronecker and finite Kronecker structures and 
the subpencil `Mil-λNil` contains the left Kronecker structures and infinite elementary 
divisors of the pencil `M-λN`. 

The full row rank subpencil `Mrf-λNrf` is in the form 
 
          Mrf-λNrf = | B  | A-λE | ,        
                     
where `E` is an `nxn` non-singular upper triangular matrix, and `A` and `B` are `nxn`- and `nxm`-dimensional matrices,
respectively, and `p = 0`.  

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
   
   # Step 0: Reduce to the standard form
   M1, N1, Q, Z, n, m, p = preduceBF(M, N; atol = atol2, rtol = rtol, fast = fast) 
   
   mM, nM = size(M)
   maxmn = max(mM,nM)
   μ = Vector{Int}(undef,maxmn)
   ν = Vector{Int}(undef,maxmn)
        
   # fast returns for null dimensions
   if mM == 0 && nM == 0
      return M1, N1, Q, Z, ν, μ, n, m, p
   end


   if finite_infinite
      if mM == 0
         return M1, N1, Q, Z, ν[1:0], μ[1:0], n, m, p
      elseif nM == 0
         ν[1] = mM
         μ[1] = 0
         p = 0
         return M1, N1, Q, Z, ν[1:1], μ[1:1], n, m, p
      end

      # Reduce M-λN to a KLF which exhibits the splitting of the right-finite and infinite-left structures
      #
      #                  | Mrf - λ Nrf |     *        |
      #      M1 - λ N1 = |-------------|--------------|
      #                  |    0        | Mil -  λ Nil |
      # 
      # where Mil -  λ Nil is in a staircase form.  

      mrinf = 0
      nrinf = 0
      rtrail = 0
      ctrail = 0
      i = 0
      tol1 = max(atol1, rtol*opnorm(M,1))     
      while p > 0
         # Step 1 & 2: Dual algorithm PREDUCE
         τ, ρ  = _preduce2!(n, m, p, M1, N1, Q, Z, tol1; fast = fast, 
                            roff = mrinf, coff = nrinf, rtrail = rtrail, ctrail = ctrail, 
                            withQ = withQ, withZ = withZ)
         i += 1
         ν[i] = p
         μ[i] = ρ+τ
         ctrail += ρ+τ
         rtrail += p
         n -= ρ
         p = ρ
         m -= τ 
      end
      return M1, N1, Q, Z, reverse(ν[1:i]), reverse(μ[1:i]), n, m, p                                             
   else
      if mM == 0
         ν[1] = 0
         μ[1] = nM
         m = 0
         return M1, N1, Q, Z, ν[1:1], μ[1:1], n, m, p
      elseif nM == 0
         return M1, N1, Q, Z, ν[1:0], μ[1:0], n, m, p
      end

      # Reduce M-λN to a KLF which exhibits the splitting the right-infinite and finite-left structures
      #
      #                  | Mri - λ Nri |     *        |
      #      M1 - λ N1 = |-------------|--------------|
      #                  |    0        | Mfl -  λ Nfl |
      # where Mri - λ Nri is in a staircase form.  

      mrinf = 0
      nrinf = 0
      i = 0
      tol1 = max(atol1, rtol*opnorm(M,1))
   
      while m > 0
         # Steps 1 & 2: Standard algorithm PREDUCE
          τ, ρ = _preduce1!( n, m, p, M1, N1, Q, Z, tol1; fast = fast, 
                            roff = mrinf, coff = nrinf, withQ = withQ, withZ = withZ)
         i += 1
         ν[i] = ρ+τ
         μ[i] = m
         mrinf += ρ+τ
         nrinf += m
         n -= ρ
         m = ρ
         p -= τ 
      end
      return M1, N1, Q, Z, ν[1:i], μ[1:i], n, m, p                                             
   end
end
"""
    klf(M, N; fast = true, finite_infinite = false, ut = false, atol1 = 0, atol2 = 0, rtol, withQ = true, withZ = true) -> (F, G, Q, Z, νr, μr, νi, nf, νl, μl)

Reduce the matrix pencil `M - λN` to an equivalent form `F - λG = Q'(M - λN)Z` using 
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
If `ut = true`, the full row rank diagonal blocks of `Mr` are reduced to the form `[0 X]` 
with `X` upper triangular and nonsingular and the full column rank supradiagonal blocks of `Nr` are
reduced to the form `[Y; 0]` with `Y` upper triangular and nonsingular. 

The pencil `Mreg-λNreg` is regular, contains the infinite and finite elementary divisors of `M-λN` and has the form

                       | Mi-λNi |    *    |
         Mreg-λNreg  = |--------|---------|, if `finite_infinite = false`, or the form
                       |   0    |  Mf-λNf |


                       | Mf-λNf |    *    |
         Mreg-λNreg  = |--------|---------|, if `finite_infinite = true`, 
                       |   0    |  Mi-λNi |
     
where: (1) `Mi-λNi`, in staircase form, contains the infinite elementary divisors of `M-λN`, 
`Mi` upper triangular if `ut = true` and nonsingular, and `Ni` is upper triangular and nilpotent; 
(2) `Mf-λNf` contains the infinite elementary divisors of `M-λN` and 
`Nf` is upper triangular and nonsingular.
The `ni`-dimensional vector `νi` contains the dimensions of the square blocks of the staircase form  `Mi-λNi` 
such that the `i`-th block has dimensions `νi[i] x νi[i]`. 
If `finite_infinite = true`, the difference `νi[i]-νi[i+1]` for `i = 1, 2, ..., ni` 
is the number of infinite elementary divisors of degree `i` (with `νi[ni] = 0`).
If `finite_infinite = false`, the difference `νi[ni-i+1]-νi[ni-i]` for `i = 1, 2, ..., ni` 
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
If `ut = true`, the full column rank diagonal blocks of `Ml` are reduced to the form `[X; 0]` 
with `X` upper triangular and nonsingular and the full row rank supradiagonal blocks of `Nl` are
reduced to the form `[0 Y]` with `Y` upper triangular and nonsingular. 

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
function klf(M::AbstractMatrix, N::AbstractMatrix; fast::Bool = true, finite_infinite::Bool = false, ut::Bool = false, 
             atol1::Real = zero(real(eltype(M))), atol2::Real = zero(real(eltype(M))),
             rtol::Real = (min(size(M)...)*eps(real(float(one(eltype(M))))))*iszero(min(atol1,atol2)), withQ::Bool = true, withZ::Bool = true)
   
   # Step 0: Reduce to the standard form
   M1, N1, Q, Z, n, m, p = preduceBF(M, N; atol = atol2, rtol = rtol, fast = fast) 
   mM, nM = size(M)

   if finite_infinite
      
      # Reduce M-λN to a KLF exhibiting the right and finite structures
      #                  [ Mr - λ Nr  |   *        |     *        ]
      #      M1 - λ N1 = [    0       | Mf -  λ Nf |     *        ]
      #                  [    0       |    0       | Mil -  λ Nil ]
      
      νr, μr, nf, ν, μ, tol1 = klf_right!(n, m, p, M1, N1, Q, Z, atol = atol1, rtol = rtol,  
                                          withQ = withQ, withZ = withZ, fast = fast)
      if mM == 0 || nM == 0
          return  M1, N1, Q, Z, νr, μr, ν[1:0], nf, ν, μ
      end

      mr = sum(νr)+nf
      nr = sum(μr)+nf
      ut && klf_right_refineut!(νr, μr, M1, N1, Q, Z, ctrail = nM-nr, withQ = withQ, withZ = withZ)

      # Reduce Mil-λNil to a KLF exhibiting the infinite and left structures and update M1 - λ N1 to
      #                  [ Mr - λ Nr  |   *        |     *      |    *       ]
      #     M2 -  λ N2 = [    0       | Mf -  λ Nf |     *      |    *       ]
      #                  [    0       |    0       | Mi -  λ Ni |    *       ]
      #                  [    0       |    0       |    0       | Ml -  λ Nl ]

      jM2 = nr+1:nr+sum(μ)
      M2 = view(M1,:,jM2)
      N2 = view(N1,:,jM2)
      withZ ? (Z2 = view(Z,:,jM2)) : (Z2 = nothing)
      νi, νl, μl = klf_left_refine!(ν, μ, M2, N2, Q, Z2, tol1, roff = mr,    
                                    withQ = withQ, withZ = withZ, fast = fast, ut = ut)
   else

      # Reduce M-λN to a KLF exhibiting the left and finite structures
      #                  [ Mri - λ Nri  |   *        |     *      ]
      #      M1 - λ N1 = [     0        | Mf -  λ Nf |     *      ]
      #                  [     0        |    0       | Ml -  λ Nl ]

      ν, μ, nf, νl, μl, tol1 = klf_left!(n, m, p, M1, N1, Q, Z, atol = atol1, rtol = rtol,  
                                         withQ = withQ, withZ = withZ, fast = fast)
      if mM == 0 || nM == 0
         return  M1, N1, Q, Z, ν, μ, ν[1:0], nf, νl, μl
      end

      ut && klf_left_refineut!(νl, μl, M1, N1, Q, Z, roff = mM-sum(νl), coff = nM-sum(μl), withQ = withQ, withZ = withZ)


      # Reduce Mri-λNri to a KLF exhibiting the right and infinite structures and update M1 - λ N1 to
      #                  [ Mr - λ Nr  |   *        |     *      |    *       ]
      #     M2 -  λ N2 = [    0       | Mi -  λ Ni |     *      |    *       ]
      #                  [    0       |    0       | Mf -  λ Nf |    *       ]
      #                  [    0       |    0       |    0       | Ml -  λ Nl ]

      iM11 = 1:sum(ν)
      M11 = view(M1,iM11,:)
      N11 = view(N1,iM11,:)
      νr, μr, νi = klf_right_refine!(ν, μ, M11, N11, Q, Z, tol1, ctrail = nM-sum(μ),   
                                     withQ = withQ, withZ = withZ, fast = fast, ut = ut)
   end
   return  M1, N1, Q, Z, νr, μr, νi, nf, νl, μl                                             
end
"""
    klf_right(M, N; fast = true, ut = false, atol1 = 0, atol2 = 0, rtol, withQ = true, withZ = true) -> (F, G, Q, Z, νr, μr, nf, ν, μ)

Reduce the matrix pencil `M - λN` to an equivalent form `F - λG = Q'(M - λN)Z` using 
orthogonal or unitary transformation matrices `Q` and `Z` such that the transformed matrices `F` and `G` are in the
following Kronecker-like form exhibiting the right and finite Kronecker structures:
 
                   |  Mr-λNr    |     *      |    *     |
                   |------------|------------|----------|
         F - λG =  |    O       |   Mf-λNf   |    *     |
                   |------------|------------|----------|
                   |    O       |     0      | Mil-λNil |            

The full row rank pencil `Mr-λNr`, in a staircase form, contains the right Kronecker indices of `M-λN` and has the form

         Mr-λNr  = | Br | Ar-λEr |,
                   
where `Er` is upper triangular and nonsingular. 
The `nr`-dimensional vectors `νr` and `μr` contain the row and, respectively, column dimensions of the blocks
of the staircase form  `Mr-λNr` such that `i`-th block has dimensions `νr[i] x μr[i]` and 
has full row rank. The difference `μr[i]-νr[i]` for `i = 1, 2, ..., nr` is the number of elementary Kronecker blocks
of size `(i-1) x i`.
If `ut = true`, the full row rank diagonal blocks of `Mr` are reduced to the form `[0 X]` 
with `X` upper triangular and nonsingular and the full column rank supradiagonal blocks of `Nr` are
reduced to the form `[Y; 0]` with `Y` upper triangular and nonsingular. 

The `nf x nf` pencil `Mf-λNf` is regular and contains the finite elementary divisors of `M-λN`. 
Nf is upper triangular and nonsingular.

The full column rank pencil `Mil-λNil` is in a staircase form, and contains the  left Kronecker indices 
and infinite elementary divisors of the pencil `M-λN`. 
The `nb`-dimensional vectors `ν` and `μ` contain the row and, respectively, column dimensions of the blocks
of the staircase form  `Mil-λNil` such that the `i`-th block has dimensions `ν[i] x μ[i]` and has full column rank. 
The difference `ν[nb-j+1]-μ[nb-j+1]` for `j = 1, 2, ..., nb` is the number of elementary Kronecker blocks of size 
`j x (j-1)`. The difference `μ[nb-j+1]-ν[nb-j]` for `j = 1, 2, ..., nb` is the number of infinite elementary 
divisors of degree `j` (with `ν[0] = 0`).
If `ut = true`, the full column rank diagonal blocks of `Mil` are reduced to the form `[X; 0]` 
with `X` upper triangular and nonsingular and the full row rank supradiagonal blocks of `Nil` are
reduced to the form `[0 Y]` with `Y` upper triangular and nonsingular. 

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of `M`, the absolute tolerance for the nonzero elements of `N`,  and the relative tolerance 
for the nonzero elements of `M` and `N`. 
The reduction is performed using rank decisions based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.

The performed left orthogonal or unitary transformations are accumulated in the matrix `Q` if `withQ = true`. 
Otherwise, `Q` is set to `nothing`.   
The performed right orthogonal or unitary or unitary transformations are accumulated in the matrix `Z` if `withZ = true`. 
Otherwise, `Z` is set to `nothing`.   

`Note:` If the pencil `M - λN` has full row rank, then the regular pencil `Mil-λNil` is in a staircase form with
square upper triangular diagonal blocks (i.e.,`μ[j] = ν[j]`), and the difference `ν[nb-j+1]-ν[nb-j]` for 
`j = 1, 2, ..., nb` is the number of infinite elementary divisors of degree `j` (with `ν[0] = 0`).
"""
function klf_right(M::AbstractMatrix, N::AbstractMatrix; fast::Bool = true, ut::Bool = false, 
                   atol1::Real = zero(real(eltype(M))), atol2::Real = zero(real(eltype(M))), 
                   rtol::Real = (min(size(M)...)*eps(real(float(one(eltype(M))))))*iszero(max(atol1,atol2)), 
                   withQ::Bool = true, withZ::Bool = true)
   
   # Step 0: Reduce to the standard form
   M1, N1, Q, Z, n, m, p = preduceBF(M, N; atol = atol2, rtol = rtol, fast = fast) 

   νr, μr, nf, ν, μ, tol1 = klf_right!(n, m, p, M1, N1, Q, Z, atol = atol1, rtol = rtol,  
                                       withQ = withQ, withZ = withZ, fast = fast)
   
   if ut
      mM, nM = size(M)
      mr = sum(νr)+nf
      nr = sum(μr)+nf
      klf_right_refineut!(νr, μr, M1, N1, Q, Z, ctrail = nM-nr, withQ = withQ, withZ = withZ)
      klf_left_refineut!(ν, μ, M1, N1, Q, Z, roff = mM-sum(ν), coff = nM-sum(μ), withQ = withQ, withZ = withZ)
   end
   return M1, N1, Q, Z, νr, μr, nf, ν, μ

end
"""
    klf_left(M, N; fast = true, ut = false, atol1 = 0, atol2 = 0, rtol, withQ = true, withZ = true) -> (F, G, Q, Z, ν, μ, nf, νl, μl)

Reduce the matrix pencil `M - λN` to an equivalent form `F - λG = Q'(M - λN)Z` using 
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
If `ut = true`, the full row rank diagonal blocks of `Mri` are reduced to the form `[0 X]` 
with `X` upper triangular and nonsingular and the full column rank supradiagonal blocks of `Nri` are
reduced to the form `[Y; 0]` with `Y` upper triangular and nonsingular. 

The `nl`-dimensional vectors `νl` and `μl` contain the row and, respectively, column dimensions of the blocks
of the staircase form  `Ml-λNl` such that the `j`-th block has dimensions `νl[j] x μl[j]` and has full column rank. 
The difference `νl[nl-j+1]-μl[nl-j+1]` for `j = 1, 2, ..., nl` is the number of elementary Kronecker blocks of size 
`j x (j-1)`.
If `ut = true`, the full column rank diagonal blocks of `Ml` are reduced to the form `[X; 0]` 
with `X` upper triangular and nonsingular and the full row rank supradiagonal blocks of `Nl` are
reduced to the form `[0 Y]` with `Y` upper triangular and nonsingular. 

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of `M`, the absolute tolerance for the nonzero elements of `N`,  and the relative tolerance 
for the nonzero elements of `M` and `N`. 
The reduction is performed using rank decisions based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.

The performed left orthogonal or unitary transformations are accumulated in the matrix `Q` if `withQ = true`. 
Otherwise, `Q` is set to `nothing`.   
The performed right orthogonal or unitary transformations are accumulated in the matrix `Z` if `withZ = true`. 
Otherwise, `Z` is set to `nothing`.  

`Note:` If the pencil `M - λN` has full column rank, then the regular pencil `Mri-λNri` is in a staircase form with
square upper triangular diagonal blocks (i.e.,`μ[i] = ν[i]`), and the difference `ν[i+1]-ν[i]` for `i = 1, 2, ..., nb` 
is the number of infinite elementary divisors of degree `i` (with `ν[nb+1] = 0`).
"""
function klf_left(M::AbstractMatrix, N::AbstractMatrix; fast::Bool = true, ut::Bool = false, 
                  atol1::Real = zero(real(eltype(M))), atol2::Real = zero(real(eltype(M))), 
                  rtol::Real = (min(size(M)...)*eps(real(float(one(eltype(M))))))*iszero(max(atol1,atol2)), 
                  withQ::Bool = true, withZ::Bool = true)
   # Step 0: Reduce to the standard form
   M1, N1, Q, Z, n, m, p = preduceBF(M, N; atol = atol2, rtol = rtol, fast = fast) 

   ν, μ, nf, νl, μl, tol1 = klf_left!(n, m, p, M1, N1, Q, Z, atol = atol1, rtol = rtol,  
                                      withQ = withQ, withZ = withZ, fast = fast)
   if ut
      mM, nM = size(M)
      klf_left_refineut!(νl, μl, M1, N1, Q, Z, roff = mM-sum(νl), coff = nM-sum(μl), withQ = withQ, withZ = withZ)
      klf_right_refineut!(ν, μ, M1, N1, Q, Z, ctrail = nM-sum(μ), withQ = withQ, withZ = withZ)
   end
   return M1, N1, Q, Z, ν, μ, nf, νl, μl
end
"""
    klf_rightinf(M, N; fast = true, ut = false, atol1 = 0, atol2 = 0, rtol, withQ = true, withZ = true) 
                -> (F, G, Q, Z, νr, μr, νi, n, p)

Reduce the matrix pencil `M - λN` to an equivalent form `F - λG = Q'(M - λN)Z` using 
orthogonal or unitary transformation matrices `Q` and `Z` such that the transformed matrices `F` and `G` are in the
following Kronecker-like form exhibiting the infinite and left Kronecker structures:
 
                   |  Mr-λNr  |     *      |    *     |
                   |----------|------------|----------|
         F - λG =  |    O     |   Mi-λNi   |    *     |
                   |----------|------------|----------|
                   |    O     |     0      | Mfl-λNfl |
               

The full row rank pencil `Mr-λNr` is in a staircase form, contains the right Kronecker indices 
of the pencil `M-λN`and has the form

         Mr-λNr  = | Br | Ar-λEr |,
          
where `Er` is upper triangular and nonsingular. 
The `nr`-dimensional vectors `νr` and `μr` contain the row and, respectively, column dimensions of the blocks
of the staircase form  `Mr-λNr` such that the `i`-th block has dimensions `νr[i] x μr[i]` and 
has full row rank. The difference `μr[i]-νr[i]` for `i = 1, 2, ..., nr` is the number of elementary Kronecker 
blocks of size `(i-1) x i`. 
If `ut = true`, the full row rank diagonal blocks of `Mr` are reduced to the form `[0 X]` 
with `X` upper triangular and nonsingular and the full column rank supradiagonal blocks of `Nr` are
reduced to the form `[Y; 0]` with `Y` upper triangular and nonsingular. 

The regular pencil `Mi-λNi` is in a staircase form, contains the infinite elementary divisors of `M-λN` 
with `Mi` upper triangular if `ut = true` and nonsingular, and `Ni` is upper triangular and nilpotent. 
The `ni`-dimensional vector `νi` contains the dimensions of the 
square diagonal blocks of the staircase form  `Mi-λNi` such that the `i`-th block has dimensions `νi[i] x νi[i]`. 
The difference `νi[ni-i+1]-νi[ni-i]` for `i = 1, 2, ..., ni` is the number of infinite elementary divisors of degree `i` (with `νi[0] = 0`).

The full column rank pencil `Mfl-λNfl` contains the left Kronecker  
and finite Kronecker structures of the pencil `M-λN` and is in the form 
 
                     | A-λE | 
          Mfl-λNfl = |------| ,        
                     |  C   |

where `E` is an nxn non-singular upper triangular matrix, and `A` and `C` are `nxn`- and `pxn`-dimensional matrices,
respectively.                      

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
function klf_rightinf(M::AbstractMatrix, N::AbstractMatrix; fast::Bool = true, ut::Bool = false, 
                  atol1::Real = zero(real(eltype(M))), atol2::Real = zero(real(eltype(M))), 
                  rtol::Real = (min(size(M)...)*eps(real(float(one(eltype(M))))))*iszero(max(atol1,atol2)), 
                  withQ::Bool = true, withZ::Bool = true)
   M1, N1, Q, Z, ν, μ, n, m, p = klf_rlsplit(M, N; fast = fast, finite_infinite = false, 
                                             atol1 = atol1, atol2 = atol2, rtol = rtol, withQ = withQ, withZ = withZ) 
   tol1 = max(atol1, rtol*opnorm(M,1))     
   νr, μr, νi = klf_right_refine!(ν, μ, M1, N1, Q, Z, tol1, ut = ut, withQ = withQ, withZ = withZ, fast = fast)
   return M1, N1, Q, Z, νr, μr, νi, n, p
end
"""
    klf_leftinf(M, N; fast = true, ut = false, atol1 = 0, atol2 = 0, rtol, withQ = true, withZ = true) 
                -> (F, G, Q, Z, n, m, νi, νl, μl)

Reduce the matrix pencil `M - λN` to an equivalent form `F - λG = Q'(M - λN)Z` using 
orthogonal or unitary transformation matrices `Q` and `Z` such that the transformed matrices `F` and `G` are in the
following Kronecker-like form exhibiting the infinite and left Kronecker structures:
 
                   |  Mrf-λNrf  |     *      |    *    |
                   |------------|------------|---------|
         F - λG =  |    O       |   Mi-λNi   |    *    |
                   |------------|------------|---------|
                   |    O       |     0      |  Ml-λNl |
               
 
The full row rank pencil `Mrf-λNrf` contains the right Kronecker  
and finite Kronecker structures of the pencil `M-λN` and is in the standard form 
 
          Mrf-λNrf = | B  | A-λE | ,        
                     
where `E` is an `nxn` non-singular upper triangular matrix, and `A` and `B` are `nxn`- and `nxm`-dimensional matrices,
respectively.                      

The regular pencil `Mi-λNi` is in a staircase form, contains the infinite elementary divisors of `M-λN` 
with `Mi` upper triangular if `ut = true` and nonsingular, and `Ni` is upper triangular and nilpotent. 
The `ni`-dimensional vector `νi` contains the dimensions of the 
square diagonal blocks of the staircase form  `Mi-λNi` such that the `i`-th block has dimensions `νi[i] x νi[i]`. 
The difference `νi[i]-νi[i-1] = ν[ni-i+1]-μ[ni-i+1]` for `i = 1, 2, ..., ni` is the number of infinite elementary 
divisors of degree `i` (with `νi[0] = 0` and `μ[nb+1] = 0`).

The full column rank pencil `Ml-λNl`, in a staircase form, contains the left Kronecker indices of `M-λN` and has the form

                   | Al-λEl |
         Ml-λNl  = |--------|,
                   |   Cl   |

where `El` is upper triangular and nonsingular. 

The `nl`-dimensional vectors `νl` and `μl` contain the row and, respectively, column dimensions of the blocks
of the staircase form  `Ml-λNl` such that the `j`-th block has dimensions `νl[j] x μl[j]` and has full column rank. 
The difference `νl[nl-j+1]-μl[nl-j+1]` for `j = 1, 2, ..., nl` is the number of elementary Kronecker blocks of size 
`j x (j-1)`.
If `ut = true`, the full column rank diagonal blocks of `Ml` are reduced to the form `[X; 0]` 
with `X` upper triangular and nonsingular and the full row rank supradiagonal blocks of `Nl` are
reduced to the form `[0 Y]` with `Y` upper triangular and nonsingular. 

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
function klf_leftinf(M::AbstractMatrix, N::AbstractMatrix; fast::Bool = true, ut::Bool = false, 
                  atol1::Real = zero(real(eltype(M))), atol2::Real = zero(real(eltype(M))), 
                  rtol::Real = (min(size(M)...)*eps(real(float(one(eltype(M))))))*iszero(max(atol1,atol2)), 
                  withQ::Bool = true, withZ::Bool = true)
   M1, N1, Q, Z, ν, μ, n, m, p = klf_rlsplit(M, N; fast = fast, finite_infinite = true, 
            atol1 = atol1, atol2 = atol2, rtol = rtol, withQ = withQ, withZ = withZ) 
   mr = n+p
   nr = n+m
   jM2 = nr+1:nr+sum(μ)
   M2 = view(M1,:,jM2)
   N2 = view(N1,:,jM2)
   withZ ? (Z2 = view(Z,:,jM2)) : (Z2 = nothing)
   tol1 = max(atol1, rtol*opnorm(M2,1))     
   νi, νl, μl = klf_left_refine!(ν, μ, M2, N2, Q, Z2, tol1, ut = ut, roff = mr,    
                                 withQ = withQ, withZ = withZ, fast = fast)
   return M1, N1, Q, Z, n, m, νi, νl, μl
end
"""
    klf_right!(M, N; fast = true, roff = 0, coff = 0, rtrail = 0, ctrail = 0, atol = 0, rtol, withQ = true, withZ = true) -> (Q, Z, νr, μr, nf, ν, μ, tol)

Reduce the partitioned matrix pencil `M - λN` (`*` stands for a not relevant subpencil)

              [   *         *        *  ] roff
     M - λN = [   0      M22-λN22    *  ] m
              [   0         0        *  ] rtrail
                coff        n     ctrail
  
to an equivalent form `F - λG = Q1'*(M - λN)*Z1` using orthogonal or unitary transformation matrices `Q1` and `Z1` 
such that the subpencil `M22 - λN22` is transformed into the following Kronecker-like form exhibiting 
its right and finite Kronecker structures:

 
                   |  Mr-λNr    |     *      |    *     |
                   |------------|------------|----------|
     F22 - λG22 =  |    O       |   Mf-λNf   |    *     |
                   |------------|------------|----------|
                   |    O       |     0      | Mil-λNil |

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

The full column rank pencil `Mil-λNil` is in a staircase form, and contains the  left Kronecker indices 
and infinite elementary divisors of the subpencil `M22 - λN22`. 
The `nb`-dimensional vectors `ν` and `μ` contain the row and, respectively, column dimensions of the blocks
of the staircase form  `Mil-λNil` such that the `i`-th block has dimensions `ν[nb-i+1] x μ[nb-i+1]` and has full column rank. 
The difference `ν[nb-i+1]-μ[nb-i+1]` for `i = 1, 2, ..., nb` is the number of elementary Kronecker blocks of size 
`i x (i-1)`. The difference `μ[nb-i+1]-ν[nb-i]` for `i = 1, 2, ..., nb` is the number of infinite elementary 
divisors of degree `i` (with `ν[0] = 0`).

The keyword arguments `atol` and `rtol`, specify the absolute and relative tolerances for the 
nonzero elements of `M`, respectively. The internally employed absolute tolerance  for the 
nonzero elements of `M` is returned in `tol`. 
The reduction is performed using rank decisions based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.

The performed left orthogonal or unitary transformations are accumulated in the matrix `Q` (i.e., `Q <- Q*Q1`) 
if `withQ = true`. Otherwise, `Q` is not modified.   
The performed right orthogonal or unitary transformations are accumulated in the matrix `Z` (i.e., `Z <- Z*Z1`) 
if `withZ = true`. Otherwise, `Z` is not modified.   
 
`Note:` If the subpencil `M22 - λN22` has full row rank, then the regular pencil `Mil-λNil` is in a staircase form with
square upper triangular diagonal blocks (i.e.,`μ[i] = ν[i]`), and the difference `ν[nb-i+1]-ν[nb-i]` for 
`i = 1, 2, ..., nb` is the number of infinite elementary divisors of degree `i` (with `ν[0] = 0`).
"""
function klf_right!(n::Int, m::Int, p::Int, M::AbstractMatrix{T1}, N::AbstractMatrix{T1}, 
                    Q::Union{AbstractMatrix{T1},Nothing}, Z::Union{AbstractMatrix{T1},Nothing}; 
                    fast::Bool = true, atol::Real = zero(real(eltype(M))), 
                    rtol::Real = (min(size(M)...)*eps(real(float(one(eltype(M))))))*iszero(atol), 
                    roff::Int = 0, coff::Int = 0, rtrail::Int = 0, ctrail::Int = 0, withQ::Bool = true, withZ::Bool = true) where T1 <: BlasFloat
   mM, nM = size(M)
   (mM,nM) == size(N) || throw(DimensionMismatch("M and N must have the same dimensions"))
   (!isa(M,Adjoint) && !isa(N,Adjoint)) || error("No adjoint inputs are supported")

   # Step 0: Reduce M22-λN22 to the standard form
   #n, m, p = _preduceBF!(M, N, Q, Z; atol = atol2, rtol = rtol, fast = fast, roff = roff, coff = coff, rtrail = rtrail, ctrail = ctrail, withQ = withQ, withZ = withQ) 
   
   maxmn = max(mM,nM)
   μr = Vector{Int}(undef,maxmn)
   νr = Vector{Int}(undef,maxmn)
   μ = Vector{Int}(undef,maxmn)
   ν = Vector{Int}(undef,maxmn)
   nf = 0
   tol = atol
 
   # fast returns for null dimensions
   if mM == 0 && nM == 0
      return νr, μr, nf, ν, μ, tol
   elseif mM == 0
      νr[1] = 0
      μr[1] = nM
      return νr[1:1], μr[1:1], nf, ν[1:0], μ[1:0], tol
   elseif nM == 0
      ν[1] = mM
      μ[1] = 0
      return νr[1:0], μr[1:0], nf, ν[1:1], μ[1:1], tol
   end

   j = 0
   tol = max(atol, rtol*opnorm(M,1))
   
   while p > 0
      # Steps 1 & 2: Dual algorithm PREDUCE
      τ, ρ = _preduce2!(n, m, p, M, N, Q, Z, tol; 
                        fast = fast, roff = roff, coff = coff, rtrail = rtrail, ctrail = ctrail, 
                        withQ = withQ, withZ = withZ)
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
      # Step 3: Particular case of the standard algorithm PREDUCE
      ρ = _preduce3!(n, m, M11, N11, Q, Z, tol, 
                     fast = fast, coff = coff, roff = roff, ctrail = ctrail,  withQ = withQ, withZ = withZ)
      i += 1
      νr[i] = ρ
      μr[i] = m
      roff += ρ
      coff += m
      n -= ρ
      m = ρ
   end

   return νr[1:i], μr[1:i], n, reverse(ν[1:j]), reverse(μ[1:j]), tol
end
"""

     klf_right_refine!(ν, μ, M, N, tol; fast = true, ut = false, roff = 0, coff = 0, rtrail = 0, ctrail = 0, withQ = true, withZ = true) -> (νr, μr, νi)

Reduce the partitioned matrix pencil `M - λN` (`*` stands for a not relevant subpencil)

             [   *         *        *  ] roff
    M - λN = [   0      Mri-λNri    *  ] mri
             [   0         0        *  ] rtrail
               coff      nri     ctrail

to an equivalent form `F - λG = Q1'*(M - λN)*Z1` using orthogonal or unitary transformation matrices `Q1` and `Z1` 
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
If `ut = true`, the full row rank diagonal blocks of `Mr` are reduced to the form `[0 X]` 
with `X` upper triangular and nonsingular and the full column rank supradiagonal blocks of `Nr` are
reduced to the form `[Y; 0]` with `Y` upper triangular and nonsingular. 

The regular pencil `Mi-λNi`is in a staircase form, contains the infinite elementary divisors of `Mri-λNri`, 
`Mi` is upper triangular if `ut = true` and nonsingular and `Ni` is upper triangular and nilpotent. The `ni`-dimensional vector `νi` contains the dimensions of the 
square diagonal blocks of the staircase form  `Mi-λNi` such that the `i`-th block has dimensions `νi[i] x νi[i]`. 
The difference `νi[ni-i+1]-νi[ni-i] = ν[i]-μ[i+1]` for `i = 1, 2, ..., ni` is the number of infinite elementary 
divisors of degree `i` (with `νi[0] = 0` and `μ[nb+1] = 0`).

`F` and `G` are returned in `M` and `N`, respectively.  

The performed left orthogonal or unitary transformations are accumulated in the matrix `Q` (i.e., `Q <- Q*Q1`) 
if `withQ = true`. Otherwise, `Q` is not modified.   
The performed right orthogonal or unitary transformations are accumulated in the matrix `Z` (i.e., `Z <- Z*Z1`) 
if `withZ = true`. Otherwise, `Z` is not modified.   
"""
function klf_right_refine!(ν::Vector{Int}, μ::Vector{Int}, M::AbstractMatrix{T1}, N::AbstractMatrix{T1}, Q::Union{AbstractMatrix{T1},Nothing},
   Z::Union{AbstractMatrix{T1},Nothing}, tol::Real; fast::Bool = true, ut::Bool = false, roff::Int = 0, coff::Int = 0, rtrail::Int = 0, ctrail::Int = 0, 
   withQ::Bool = true, withZ::Bool = true) where T1 <: BlasFloat
   nb = length(ν)
   nb == length(μ) || throw(DimensionMismatch("ν and μ must have the same lengths"))
  
   nb == 0 && (return ν[1:0], ν[1:0], ν[1:0])
   
   mri = sum(ν)
   nri = sum(μ)
   mM = roff + mri + rtrail
   nM = coff + nri + ctrail

   rtrail0 = rtrail
   ctrail0 = ctrail

   m = μ[1]
   n = nri-m
   p = mri-n

   if n > 0
      # Step 0: Reduce Nri = [ 0 E11] to standard form, where E11 is full column rank
      #                      [ 0 0  ]
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
   ni = 0

   while p > 0
      # Steps 1 & 2: Dual algorithm PREDUCE to separate right-finite and infinite-left parts.

      # Reduce Mri-λNri to [ Mr1-λNr1 * ; 0 Mi-λNi], where Mr1-λNr1 contains the right Kronecker structure and the empty
      # finite part and Mi-λNi contains the infinite elementary divisors and the empty left Kronecker structure.
      τ, ρ = _preduce2!(n, m, p, M, N, Q, Z, tol; 
                        fast = fast, roff = roff, coff = coff, rtrail = rtrail, ctrail = ctrail, 
                        withQ = withQ, withZ = withZ)
      ρ+τ == p || error("The reduced pencil must not have left structure: try to adjust the tolerances")
      j += 1
      νi[j] = p
      ni += p
      ctrail += p
      rtrail += p
      n -= ρ
      p = ρ
      m -= τ 
   end

   # make Mi upper triangular 
   reverse!(view(νi,1:j))
   ut && ni > 0 && klf_right_refineinf!(view(νi,1:j), M, N, Z, missing; roff = mM-rtrail0-ni, coff = nM-ctrail0-ni, withZ = withZ) 

   i = 0
   if m > 0
      imM11 = 1:mM-rtrail
      M11 = view(M,imM11,1:nM)
      N11 = view(N,imM11,1:nM)
   end
   while m > 0
      # Step 3: Particular form of the standard algorithm PREDUCE to reduce Mr1-λNr1 to Mr-λNr in  staircase form. 
      ρ = _preduce3!(n, m, M11, N11, Q, Z, tol, fast = fast, 
                     coff = coff, roff = roff, ctrail = ctrail, withQ = withQ, withZ = withZ)
      i += 1
      νr[i] = ρ
      μr[i] = m
      roff += ρ
      coff += m
      n -= ρ
      m = ρ
   end
   ut && i > 0 && klf_right_refineut!(view(νr,1:i), view(νr,1:i), M, N, Q, Z; 
                                      ctrail = nM-sum(view(μr,1:i)), withQ = withQ, withZ = withZ) 
   
   return νr[1:i], μr[1:i], νi[1:j]
end
"""
    klf_left!(M, N; fast = true, roff = 0, coff = 0, rtrail = 0, ctrail = 0, atol = 0, rtol, withQ = true, withZ = true) -> (Q, Z, ν, μ, nf, νl, μl, tol)

Reduce the partitioned matrix pencil `M - λN` (`*` stands for a not relevant subpencil)

              [   *         *        *  ] roff
     M - λN = [   0      M22-λN22    *  ] m
              [   0         0        *  ] rtrail
                coff        n     ctrail
  
to an equivalent form `F - λG = Q1'*(M - λN)*Z1` using orthogonal or unitary transformation matrices `Q1` and `Z1` 
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

The keyword arguments `atol` and `rtol`, specify the absolute and relative tolerances for the 
nonzero elements of `M`, respectively. The internally employed absolute tolerance  for the 
nonzero elements of `M` is returned in `tol`. 
The reduction is performed using rank decisions based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.
   
The performed left orthogonal or unitary transformations are accumulated in the matrix `Q` (i.e., `Q <- Q*Q1`) 
if `withQ = true`. Otherwise, `Q` is not modified.   
The performed right orthogonal or unitary transformations are accumulated in the matrix `Z` (i.e., `Z <- Z*Z1`) 
if `withZ = true`. Otherwise, `Z` is not modified.   

`Note:` If the pencil `M22 - λN22` has full column rank, then the regular pencil `Mri-λNri` is in a staircase form with
square diagonal blocks (i.e.,`μ[i] = ν[i]`), and the difference `ν[i]-ν[i+1]` for `i = 1, 2, ..., nb` 
is the number of infinite elementary divisors of degree `i` (with `ν[nb+1] = 0`).
"""
function klf_left!(n::Int, m::Int, p::Int, M::AbstractMatrix{T1}, N::AbstractMatrix{T1},
                   Q::Union{AbstractMatrix{T1},Nothing}, Z::Union{AbstractMatrix{T1},Nothing}; 
                   fast::Bool = true, atol::Real = zero(real(eltype(M))), 
                   rtol::Real = (min(size(M)...)*eps(real(float(one(eltype(M))))))*iszero(atol), 
                   roff::Int = 0, coff::Int = 0, rtrail::Int = 0, ctrail::Int = 0, withQ::Bool = true, withZ::Bool = true) where T1 <: BlasFloat
   mM, nM = size(M)
   (mM,nM) == size(N) || throw(DimensionMismatch("M and N must have the same dimensions"))
   (!isa(M,Adjoint) && !isa(N,Adjoint)) || error("No adjoint inputs are supported")

   maxmn = max(mM,nM)
   μ = Vector{Int}(undef,maxmn)
   ν = Vector{Int}(undef,maxmn)
   μl = Vector{Int}(undef,maxmn)
   νl = Vector{Int}(undef,maxmn)
   nf = 0
   tol = atol

   # fast returns for null dimensions
   if mM == 0 && nM == 0
      return ν, μ, nf, νl, μl, tol
   elseif mM == 0
      ν[1] = 0
      μ[1] = nM
      return ν[1:1], μ[1:1], nf, νl[1:0], μl[1:0], tol
   elseif nM == 0
      νl[1] = mM
      μl[1] = 0
      return ν[1:0], μ[1:0], nf, νl[1:1], μl[1:1], tol
   end

   mrinf = 0
   nrinf = 0
   i = 0
   tol = max(atol, rtol*opnorm(M,1))

   while m > 0
      # Steps 1 & 2: Standard algorithm PREDUCE
      τ, ρ = _preduce1!(n, m, p, M, N, Q, Z, tol; 
                        fast = fast, roff = roff, coff = coff, rtrail = rtrail, ctrail = ctrail, 
                        withQ = withQ, withZ = withZ)
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
      # Step 3: Particular form of the dual algorithm PREDUCE
      ρ = _preduce4!(n, 0, p, M, N, Q, Z, tol, 
                     fast = fast, roff = roff, coff = coff, rtrail = rtrail, ctrail = ctrail, 
                     withQ = withQ, withZ = withZ) 
      j += 1
      νl[j] = p
      μl[j] = ρ
      rtrail += p
      ctrail += ρ
      n -= ρ
      p = ρ
   end
 
   return ν[1:i], μ[1:i], n, reverse(νl[1:j]), reverse(μl[1:j]), tol
end
"""
    klf_left_refine!(ν, μ, M, N, tol; fast = true, ut = false, roff = 0, coff = 0, rtrail = 0, ctrail = 0, withQ = true, withZ = true) -> (νi, νl, μl)

Reduce the partitioned matrix pencil `M - λN` (`*` stands for a not relevant subpencil)

             [   *         *        *  ] roff
    M - λN = [   0      Mil-λNil    *  ] mli
             [   0         0        *  ] rtrail
               coff      nli     ctrail

to an equivalent form `F - λG = Q1'*(M - λN)*Z1` using orthogonal or unitary transformation matrices `Q1` and `Z1` 
such that the full column rank subpencil `Mil-λNil` is transformed into the 
following Kronecker-like form  exhibiting its infinite and left Kronecker structures:

                  |  Mi-λNi    |     *      |
     Fil - λGil = |------------|------------|
                  |    O       |   Ml-λNl   |

The full column rank pencil `Mil-λNil` is in a staircase form and the `nb`-dimensional vectors `ν` and `μ` 
contain the row and, respectively, column dimensions of the blocks of the staircase form  `Mil-λNil` such that 
the `i`-th block has dimensions `ν[i] x μ[i]` and has full column rank. The matrix Mil has full column rank and 
the leading `μ[1]+...+μ[nb-1]` columns of `Nil` form a full row rank submatrix. 

The regular pencil `Mi-λNi` is in a staircase form, contains the infinite elementary divisors of `Mil-λNil`,  
`Mi` is upper triangular if `ut = true` and nonsingular and `Ni` is upper triangular and nilpotent. The `nbi`-dimensional vector `νi` contains the dimensions of the 
square diagonal blocks of the staircase form  `Mi-λNi` such that the `i`-th block has dimensions `νi[i] x νi[i]`. 
The difference `νi[i]-νi[i-1] = ν[nbi-i+1]-μ[nbi-i+1]` for `i = 1, 2, ..., nbi` is the number of infinite elementary 
divisors of degree `i` (with `νi[0] = 0` and `μ[nbi+1] = 0`).

The full column rank pencil `Mil-λNil` is in a staircase form, and contains the  left Kronecker indices 
and infinite elementary divisors of the subpencil `M22 - λN22`. 
The `nb`-dimensional vectors `ν` and `μ` contain the row and, respectively, column dimensions of the blocks
of the staircase form  `Mil-λNil` such that the `i`-th block has dimensions `ν[nb-i+1] x μ[nb-i+1]` and has full column rank. 
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
If `ut = true`, the full column rank diagonal blocks of `Ml` are reduced to the form `[Y; 0]` 
with `Y` upper triangular and nonsingular and the full row rank supradiagonal blocks of `Nl` are
reduced to the form `[0 Y]` with `Y` upper triangular and nonsingular. 

`F` and `G` are returned in `M` and `N`, respectively.  

The performed left orthogonal or unitary transformations are accumulated in the matrix `Q` (i.e., `Q <- Q*Q1`) 
if `withQ = true`. Otherwise, `Q` is not modified.   
The performed right orthogonal or unitary transformations are accumulated in the matrix `Z` (i.e., `Z <- Z*Z1`) 
if `withZ = true`. Otherwise, `Z` is not modified.   
"""
function klf_left_refine!(ν::Vector{Int}, μ::Vector{Int}, M::AbstractMatrix{T1}, N::AbstractMatrix{T1}, Q::Union{AbstractMatrix{T1},Nothing},
   Z::Union{AbstractMatrix{T1},Nothing}, tol::Real; fast::Bool = true, ut::Bool = false, roff::Int = 0, coff::Int = 0, rtrail::Int = 0, ctrail::Int = 0, 
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
      # Step 0: Reduce N = [ 0 E11] to standard form, where E11 is full row rank
      #                    [ 0 0  ]
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

   roff0 = roff
   coff0 = coff
 
   i = 0
   ni = 0
   while m > 0
      # Steps 1 & 2: Standard algorithm
      τ, ρ = _preduce1!(n, m, p, M, N, Q, Z, tol; 
                        fast = fast, roff = roff, coff = coff, rtrail = rtrail, ctrail = ctrail, 
                        withQ = withQ, withZ = withZ)
      ρ+τ == m || error("The reduced pencil must not have right structure: try to adjust the tolerances")
      i += 1
      νi[i] = m
      ni += m
      roff += m
      coff += m
      n -= ρ
      m = ρ
      p -= τ 
   end
   # make Mi upper triangular
   ut && ni > 0 && klf_left_refineinf!(view(νi,1:i), M, N, Q, missing; roff = roff0, coff = coff0, withQ = withQ) 

   j = 0
   while p > 0
      # Step 3: Dual algorithm
      ρ = _preduce4!(n, 0, p, M, N, Q, Z, tol, fast = fast, 
                     roff = roff, coff = coff, rtrail = rtrail, ctrail = ctrail, 
                     withQ = withQ, withZ = withZ) 
      j += 1
      νl[j] = p
      μl[j] = ρ
      rtrail += p
      ctrail += ρ
      n -= ρ
      p = ρ
   end
   reverse!(view(νl,1:j))
   reverse!(view(μl,1:j))
   # make diagonal blocks of Ml and supradiagonal blocks of Nl upper triangular
   ut && j > 0 && klf_left_refineut!(view(νl,1:j), view(μl,1:j), M, N, Q, Z; roff = roff0+ni, coff = coff0+ni, withQ = withQ, withZ = withZ) 

 
   return νi[1:i], νl[1:j], μl[1:j]
end
"""
    klf_right_refineut!(ν, μ, M, N, Q, Z; ctrail = 0, withQ = true, withZ = true) 

Reduce the partitioned matrix pencil `M - λN` (`*` stands for a not relevant subpencil)

             [ Mri-λNri  *   ] 
    M - λN = [    0      *   ] 
                       ctrail

to an equivalent form `F - λG = Q1'*(M - λN)*Z1` using orthogonal or unitary transformation 
matrices  `Q1` and `Z1`, such that the full row rank subpencil `Mri-λNri`, in staircase form , 
is transformed as follows: the full row rank diagonal blocks of `Mri` are reduced to the form `[0 X]` 
with `X` upper triangular and nonsingular and the full column rank supradiagonal blocks of `Nri` are
reduced to the form `[Y; 0]` with `Y` upper triangular and nonsingular. 
It is assumed that `Mri-λNri` has `nb` diagonal blocks and the dimensions of the diagonal blocks  
are specified by the `nb`-dimensional vectors `ν` and `μ` such that the `i`-th block has dimensions `ν[i] x μ[i]`.  

The performed orthogonal or unitary transformations are accumulated in `Q` (i.e., `Q <- Q*Q1`), 
if `withQ = true` and `Z` (i.e., `Z <- Z*Z1`), if `withZ = true`. 
"""
function klf_right_refineut!(ν::AbstractVector{Int}, μ::AbstractVector{Int}, M::AbstractMatrix{T}, N::AbstractMatrix{T}, 
                             Q::Union{AbstractMatrix{T},Nothing}, Z::Union{AbstractMatrix{T},Nothing}; 
                             rtrail::Int = 0, ctrail::Int = 0, withQ::Bool = true, withZ::Bool = true) where T <: BlasFloat


   nb = length(ν)
   nb == 0 && return 
   nb == length(μ) || throw(DimensionMismatch("ν and μ must have the same lengths"))
   mri = sum(ν)
   nri = sum(μ)
   nM = nri + ctrail
   # mM, nM == size(M) || throw(DimensionMismatch("Incompatible ν, μ, rtrail and ctrail with the dimensions of M"))
   # mM, nM == size(N) || throw(DimensionMismatch("M and N must have the same dimensions"))
   ki2 = mri
   kj2 = nri
   T <: Complex ? tran = 'C' : tran = 'T'
   for i = nb:-1:1 
      #i == 1 && return
      ni1 = ν[i]
      ki1 = ki2-ni1+1
      kki = ki1:ki2
      nj1 = μ[i]
      kj1 = kj2-nj1+1
      kkj = kj1:kj2
      if nj1 > 1
         Mij = view(M,kki,kkj) 
         tau = similar(M,ni1)
         LinearAlgebra.LAPACK.gerqf!(Mij,tau)
         LinearAlgebra.LAPACK.ormrq!('R',tran,Mij,tau,view(M,1:ki1-1,kkj))
         LinearAlgebra.LAPACK.ormrq!('R',tran,Mij,tau,view(N,1:ki1-1,kkj))
         withZ && LinearAlgebra.LAPACK.ormrq!('R',tran,Mij,tau,view(Z,:,kkj)) 
         M[kki,kkj] = [ zeros(T,ni1,nj1-ni1) triu(view(Mij,:,nj1-ni1+1:nj1))]
      end
      if i > 1
         nim1 = ν[i-1]
         kim1 = ki1-nim1
         kim2 = ki1-1
         kkim = kim1:kim2
         if nim1 > 1
            Nimj = view(N,kkim,kkj)
            tau = similar(N,nj1)
            LinearAlgebra.LAPACK.geqrf!(Nimj,tau)
            njm1 = μ[i-1]
            kjm1 = kj1-njm1 
            kjm2 = kj1-1
            LinearAlgebra.LAPACK.ormqr!('L',tran,Nimj,tau,view(M,kkim,kjm1:nM))
            LinearAlgebra.LAPACK.ormqr!('L',tran,Nimj,tau,view(N,kkim,kj2+1:nM))
            #withQ && LinearAlgebra.LAPACK.ormqr!('R',tran,Nimj,tau,view(Q,:,kkim))
            withQ && LinearAlgebra.LAPACK.ormqr!('R','N',Nimj,tau,view(Q,:,kkim))
            triu!(Nimj)
         end
      end
      ki2 = ki1-1
      kj2 = kj1-1
   end
   return 
end
"""
    klf_left_refineut!(ν, μ, M, N, Q, Z; roff = 0, coff = 0, withQ = true, withZ = true) 

Reduce the partitioned matrix pencil `M - λN` (`*` stands for a not relevant subpencil)

             [  *     *     ] roff
    M - λN = [  0  Mil-λNil ] 
              coff

to an equivalent form `F - λG = Q1'*(M - λN)*Z1` using orthogonal or unitary transformation 
matrices  `Q1` and `Z1`, such that the full column rank subpencil `Mil-λNil`, in staircase form , 
is transformed as follows: the full column rank diagonal blocks of `Mil` are reduced to the form `[Y ; 0]` 
with `Y` upper triangular and nonsingular and the full row rank supradiagonal blocks of `Nil` are
reduced to the form `[0 Y]` with `Y` upper triangular and nonsingular. 
It is assumed that `Mil-λNil` has `nb` diagonal blocks and the dimensions of the diagonal blocks  
are specified by the `nb`-dimensional vectors `ν` and `μ` such that the `i`-th block has dimensions `ν[i] x μ[i]`.  

The performed orthogonal or unitary transformations are accumulated in `Q` (i.e., `Q <- Q*Q1`), 
if `withQ = true` and `Z` (i.e., `Z <- Z*Z1`), if `withZ = true`. 
"""
function klf_left_refineut!(ν::AbstractVector{Int}, μ::AbstractVector{Int}, M::AbstractMatrix{T}, N::AbstractMatrix{T}, 
                             Q::Union{AbstractMatrix{T},Nothing}, Z::Union{AbstractMatrix{T},Nothing}; 
                             roff::Int = 0, coff::Int = 0, withQ::Bool = true, withZ::Bool = true) where T <: BlasFloat


   nb = length(ν)
   nb == 0 && return 
   nb == length(μ) || throw(DimensionMismatch("ν and μ must have the same lengths"))
   mli = sum(ν)
   nli = sum(μ)
   mM = mli + roff
   nM = nli + coff
   # mM, nM == size(M) || throw(DimensionMismatch("Incompatible ν, μ, rtrail and ctrail with the dimensions of M"))
   # mM, nM == size(N) || throw(DimensionMismatch("M and N must have the same dimensions"))
   ki1 = roff+1
   kj1 = coff+1
   T <: Complex ? tran = 'C' : tran = 'T'
   for i = 1:nb 
      ni1 = ν[i]
      ki2 = ki1+ni1-1
      kki = ki1:ki2
      nj1 = μ[i]
      kj2 = kj1+nj1-1
      kkj = kj1:kj2
      if ni1 > 1
         Mij = view(M,kki,kkj) 
         tau = similar(M,nj1)
         LinearAlgebra.LAPACK.geqrf!(Mij,tau)
         LinearAlgebra.LAPACK.ormqr!('L',tran,Mij,tau,view(M,kki,kj2+1:nM))
         LinearAlgebra.LAPACK.ormqr!('L',tran,Mij,tau,view(N,kki,kj1+1:nM))
         withQ && LinearAlgebra.LAPACK.ormqr!('R','N',Mij,tau,view(Q,:,kki))
         triu!(Mij)
      end
      if i < nb
         nip1 = ν[i+1]
         njp1 = μ[i+1]
         kjp1 = kj2+1
         kjp2 = kj2+njp1
         kkjp = kjp1:kjp2
         if nip1 > 1
            Nijp = view(N,kki,kkjp)
            tau = similar(N,ni1)
            LinearAlgebra.LAPACK.gerqf!(Nijp,tau)
            nip1 = ν[i+1]
            kip1 = ki2+1
            kip2 = ki2+nip1
            LinearAlgebra.LAPACK.ormrq!('R',tran,Nijp,tau,view(M,1:kip2,kkjp))
            LinearAlgebra.LAPACK.ormrq!('R',tran,Nijp,tau,view(N,kip1:kip2,kkjp))
            LinearAlgebra.LAPACK.ormrq!('R',tran,Nijp,tau,view(N,1:ki1-1,kkjp))
            withZ && LinearAlgebra.LAPACK.ormrq!('R',tran,Nijp,tau,view(Z,:,kkjp)) 
            N[kki,kkjp] = [ zeros(T,ni1,njp1-ni1) triu(view(Nijp,:,njp1-ni1+1:njp1))]
         end
      end
      ki1 = ki2+1
      kj1 = kj2+1
   end
   return 
end
"""
    klf_right_refineinf!(νi, M, N, Z, R; roff = 0, coff = 0, withZ = true) 

Reduce the partitioned matrix pencil `M - λN` (`*` stands for a not relevant subpencil)

             [   *         *        *  ] roff
    M - λN = [   0       Mi-λNi     *  ] ni
             [   0         0        *  ] rtrail
               coff       ni     ctrail

to an equivalent form `F - λG = (M - λN)*Z1` using an orthogonal or unitary transformation 
matrix `Z1`, such that the regular subpencil `Mi-λNi`, in staircase form , 
with `Mi` nonsingular and `Ni` nillpotent and upper triangular, is transformed to obtain 
`Mi` is upper-triangular and preserve `Ni` upper triangular. 
It is assumed that `Mi-λNi` has `nb` diagonal blocks and the `i`-th diagonal block has dimensions 
`νi[i] x νi[i]`.

The performed orthogonal or unitary transformations are accumulated in `Z` (i.e., `Z <- Z*Z1`), 
if `withZ = true`. 

The  matrix `R` is overwritten by `R*Z1` unless `R = missing`. 
"""
function klf_right_refineinf!(νi::AbstractVector{Int}, M::AbstractMatrix{T1}, N::AbstractMatrix{T1}, Z::Union{AbstractMatrix{T1},Nothing},
   R::Union{AbstractMatrix{T1},Missing}; roff::Int = 0, coff::Int = 0, withZ::Bool = true) where T1 <: BlasFloat
   nb = length(νi)
   nb == 0 && return 
   T1 <: Complex ? tran = 'C' : tran = 'T'
   ni = sum(νi)

   ki2 = roff+ni
   kj2 = coff+ni
   for k = nb:-1:1
       nk = νi[k]
       ki1 = ki2-nk+1
       kj1 = kj2-nk+1
       kki = ki1:ki2
       kkj = kj1:kj2
       if nk > 1
          Mk = view(M,kki,kkj)
          tau = similar(M,nk)
          LinearAlgebra.LAPACK.gerqf!(Mk,tau)
          LinearAlgebra.LAPACK.ormrq!('R',tran,Mk,tau,view(M,1:ki1-1,kkj))
          withZ && LinearAlgebra.LAPACK.ormrq!('R',tran,Mk,tau,view(Z,:,kkj)) 
          LinearAlgebra.LAPACK.ormrq!('R',tran,Mk,tau,view(N,1:ki1-1,kkj))
          ismissing(R) || LinearAlgebra.LAPACK.ormrq!('R',tran,Mk,tau,view(R,:,kkj)) 
          triu!(Mk)
       end
       ki2 = ki1-1
       kj2 = kj1-1
   end        
   return 
end
"""
    klf_left_refineinf!(νi, M, N, Q, L; roff = 0, coff = 0, withQ = true) 

Reduce the partitioned matrix pencil `M - λN` (`*` stands for a not relevant subpencil)

             [   *         *        *  ] roff
    M - λN = [   0       Mi-λNi     *  ] ni
             [   0         0        *  ] rtrail
               coff       ni     ctrail

to an equivalent form `F - λG = Q1'*(M - λN)` using an orthogonal or unitary transformation 
matrix `Q1`, such that the regular subpencil `Mi-λNi`, in staircase form, 
with `Mi` nonsingular and `Ni` nillpotent and upper triangular, is transformed to obtain 
`Mi` upper-triangular and preserve `Ni` upper triangular. 
It is assumed that `Mi-λNi` has `nb` diagonal blocks and the `i`-th diagonal block has dimensions 
`νi[i] x νi[i]`.

The performed orthogonal or unitary transformations are accumulated in `Q` (i.e., `Q <- Q*Q1`), 
if `withQ = true`. 

The  matrix `L` is overwritten by `Q1'*L` unless `L = missing`. 
"""
function klf_left_refineinf!(νi::AbstractVector{Int}, M::AbstractMatrix{T1}, N::AbstractMatrix{T1}, Q::Union{AbstractMatrix{T1},Nothing},
   L::Union{AbstractMatrix{T1},Missing}; roff::Int = 0, coff::Int = 0, withQ::Bool = true) where T1 <: BlasFloat
   nb = length(νi)
   nb == 0 && return 
   T1 <: Complex ? tran = 'C' : tran = 'T'
   n = size(M,2)

   ki1 = roff+1
   kj1 = coff+1
   for k = 1:nb
       nk = νi[k]
       ki2 = ki1+nk-1
       kj2 = kj1+nk-1
       kki = ki1:ki2
       kkj = kj1:kj2
       if nk > 1
          Mk = view(M,kki,kkj)
          tau = similar(M,nk)
          LinearAlgebra.LAPACK.geqrf!(Mk,tau)
          LinearAlgebra.LAPACK.ormqr!('L',tran,Mk,tau,view(M,kki,kj2+1:n))
          withQ && LinearAlgebra.LAPACK.ormqr!('R','N',Mk,tau,view(Q,:,kki))
          ismissing(L) || LinearAlgebra.LAPACK.ormqr!('L',tran,Mk,tau,view(L,kki,:))
          LinearAlgebra.LAPACK.ormqr!('L',tran,Mk,tau,view(N,kki,kj2+1:n))
          triu!(Mk)
       end
       ki1 = ki2+1
       kj1 = kj2+1
   end        
   return 
end
