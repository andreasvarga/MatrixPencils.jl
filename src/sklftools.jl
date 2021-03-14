"""
    sreduceBF(A, E, B, C, D; fast = true, atol = 0, rtol, withQ = true, withZ = true) -> F, G, Q, Z, n, m, p

Reduce the partitioned matrix pencil `M - λN` 

               | A-λE | B | ndx
      M - λN = |------|---|
               |  C   | D | ny
                  nx    nu  
  
to an equivalent basic form `F - λG = Q'*(M - λN)*Z` using orthogonal transformation matrices `Q` and `Z` 
such that `M - λN` is transformed into the following standard form
 
               | B1 | A1-λE1 | n
      F - λG = |----|--------|   ,        
               | D1 |   C1   | p
                 m      n

where `E1` is an `nxn` non-singular matrix, and `A1`, `B1`, `C1`, `D1` are `nxn`-, `nxm`-, `pxn`- and `pxm`-dimensional matrices,
respectively. The order `n` of `E1` is equal to the numerical rank of `E` determined using the absolute tolerance `atol` and 
relative tolerance `rtol`. The dimensions `m` and `p` are computed as `m = nu + (nx-n)` and `p = ny + (ndx-n)`. 

If `fast = true`, `E1` is determined upper triangular using a rank revealing QR-decomposition with column pivoting of `E` 
and `n` is evaluated as the number of nonzero diagonal elements of the R factor, whose magnitudes are greater than 
`tol = max(atol,abs(R[1,1])*rtol)`. 
If `fast = false`,  `E1` is determined diagonal using a rank revealing SVD-decomposition of `E` and 
`n` is evaluated as the number of singular values greater than `tol = max(atol,smax*rtol)`, where `smax` 
is the largest singular value. 
The rank decision based on the SVD-decomposition is generally more reliable, but the involved computational effort is higher.

The performed left orthogonal or unitary transformations are accumulated in the matrix `Q` if `withQ = true`. 
Otherwise, `Q` is set to `nothing`.   
The performed right orthogonal or unitary transformations are accumulated in the matrix `Z` if `withZ = true`. 
Otherwise, `Z` is set to `nothing`.  
"""
function sreduceBF(A::Union{AbstractMatrix,Missing}, E::Union{AbstractMatrix,UniformScaling{Bool},Missing}, 
                   B::Union{AbstractVecOrMat,Missing}, C::Union{AbstractMatrix,Missing}, D::Union{AbstractVecOrMat,Missing}; 
                   atol::Real = ismissing(A) ? (ismissing(D) ? zero(1.) : zero(real(eltype(D)))) : zero(real(eltype(A))), 
                   rtol::Real = (ismissing(A) ? 1 : min(size(A)...))*eps(real(float(ismissing(A) ? (ismissing(D) ? zero(1.) : one(real(eltype(D)))) : one(real(eltype(A))))))*iszero(atol), 
                   fast::Bool = true, withQ::Bool = true, withZ::Bool = true) 
   xor(ismissing(A),ismissing(E)) && error("A and E must be both either present or missing")               
   ismissing(A) && !ismissing(B) && error("B can not be present if A is missing")  
   ismissing(A) && !ismissing(C) && error("C can not be present if A is missing")  
   !ismissing(D) && !ismissing(B) && ismissing(C)  && error("D can not be present if C is missing") 
   !ismissing(D) && !ismissing(C) && ismissing(B)  && error("D can not be present if B is missing") 
   eident = (typeof(E) == UniformScaling{Bool}) 
   if ismissing(A) && ismissing(D)
      T = Float64
      return zeros(T,0,0), zeros(T,0,0), zeros(T,0,0), zeros(T,0,0), 0, 0, 0
   elseif ismissing(A) 
      p, m = typeof(D) <: AbstractVector ? (length(D),1) : size(D)
      T = eltype(D)
      T <: BlasFloat || (T = promote_type(Float64,T))
      F = copy_oftype(D,T)   
      G = zeros(T,p,m)
      return F, G,  withQ ? Matrix{T}(I,p,p) : nothing, withZ ? Matrix{T}(I,m,m) : nothing, 0, m, p
   elseif ismissing(B) && ismissing(C)
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
      ndx, nx = size(A)
      eident || (ndx,nx) == size(E) || throw(DimensionMismatch("A and E must have the same dimensions"))
      ndx1, nu = typeof(B) <: AbstractVector ? (length(B),1) : size(B)
      ndx == ndx1 || throw(DimensionMismatch("A and B must have the same number of rows"))
      T = promote_type(eltype(A), eltype(E), eltype(B), eltype(E), eltype(A), eltype(E) )
      T <: BlasFloat || (T = promote_type(Float64,T))
      ny = 0
      C = zeros(T,ny,nx)
      D = zeros(T,ny,nu)
   else
      ndx, nx = size(A)
      T = promote_type(eltype(A), eltype(B), eltype(C), eltype(D))
      eident || (ndx,nx) == size(E) || throw(DimensionMismatch("A and E must have the same dimensions"))
      ny, nu = typeof(D) <: AbstractVector ? (length(D),1) : size(D)
      ndx1, nu1 = typeof(B) <: AbstractVector ? (length(B),1) : size(B)
      (ndx,nu) == (ndx1,nu1) || throw(DimensionMismatch("A, B and D must have compatible dimensions"))
      (ny,nx) == size(C) || throw(DimensionMismatch("A, C and D must have compatible dimensions"))
      T = promote_type(eltype(A), eltype(E), eltype(B), eltype(C), eltype(D))
      T <: BlasFloat || (T = promote_type(Float64,T))        
   end

   eident || (T = promote_type(T,eltype(E)))
   T <: BlasFloat || (T = promote_type(Float64,T))
   if eident 
      ndx == nx || throw(DimensionMismatch("A must be a square matrix"))
      E1 = Matrix{T}(I,nx,nx)
   end

   ismissing(A) ? A1 = missing : A1 = copy_oftype(A,T)   
   ismissing(E) ? E1 = missing : (!eident && (E1 = copy_oftype(E,T)))
   ismissing(B) ? B1 = missing : B1 = copy_oftype(B,T)
   ismissing(C) ? C1 = missing : C1 = copy_oftype(C,T)
   ismissing(D) ? D1 = missing : D1 = copy_oftype(D,T)


   withQ ? (Q = Matrix{T}(I,ndx+ny,ndx+ny)) : (Q = nothing)
   withZ ? (Z = Matrix{T}(I,nx+nu,nx+nu)) : (Z = nothing)
   if eident
      n1 = nx; m1 = 0; p1 = 0;
   else
      n1, m1, p1 = _preduceBF!(A1, E1, Q, Z, B1, C1; atol = atol, rtol = rtol, fast = fast, withQ = withQ, withZ = withQ) 
   end
   F = [ B1 A1; D1 C1]
   G = [ zeros(T,ndx,nu) E1; zeros(T,ny,nx+nu)]
   withZ && (Z = Z[:,[nx+1:nx+nu; 1:nx]])
   
   return F, G, Q, Z, n1, m1+nu, p1+ny
end
"""
    sklf(A, E, B, C, D; fast = true, finite_infinite = false, ut = false, atol1 = 0, atol2 = 0, rtol, withQ = true, withZ = true) -> (F, G, Q, Z, νr, μr, νi, nf, νl, μl)

Reduce the structured matrix pencil `M - λN` 

               | A-λE | B | 
      M - λN = |------|---|
               |  C   | D |  

to an equivalent form `F - λG = Q'*(M - λN)*Z` using 
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
`Mi` is upper triangular if `ut = true` and nonsingular, and 
`Ni` is upper triangular and nilpotent; 
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
function sklf(A::Union{AbstractMatrix,Missing}, E::Union{AbstractMatrix,UniformScaling{Bool},Missing}, 
              B::Union{AbstractVecOrMat,Missing}, C::Union{AbstractMatrix,Missing}, D::Union{AbstractVecOrMat,Missing}; 
              atol1::Real = ismissing(A) ? zero(real(eltype(D))) : zero(real(eltype(A))), 
              atol2::Real = ismissing(A) ? zero(real(eltype(D))) : zero(real(eltype(A))), 
              rtol::Real = (ismissing(A) ? 1 : min(size(A)...))*eps(real(float(ismissing(A) ? one(real(eltype(D))) : one(real(eltype(A))))))*iszero(min(atol1,atol2)), 
              fast::Bool = true, finite_infinite::Bool = false, ut::Bool = false, withQ::Bool = true, withZ::Bool = true) 
   
   # Step 0: Reduce to the standard form
   M, N, Q, Z, n, m, p = sreduceBF(A, E, B, C, D, atol = atol2, rtol = rtol, fast = fast, withQ = withQ, withZ = withZ)
   mM, nM = size(M)
   T = eltype(M)

   if finite_infinite
      
      # Reduce M-λN to a KLF exhibiting the right and finite structures
      #                  [ Mr - λ Nr  |   *        |     *        ]
      #      M1 - λ N1 = [    0       | Mf -  λ Nf |     *        ]
      #                  [    0       |    0       | Mil -  λ Nil ]
      
      νr, μr, nf, ν, μ, tol1 = klf_right!(n, m, p, M, N, Q, Z, atol = atol1, rtol = rtol,  
                                          withQ = withQ, withZ = withZ, fast = fast)
      (mM == 0 || nM == 0) && (return  M, N, Q, Z, νr, μr, ν[1:0], nf, ν, μ)
    
      # Reduce Mil-λNil to a KLF exhibiting the infinite and left structures and update M1 - λ N1 to
      #                  [ Mr - λ Nr  |   *        |     *      |    *       ]
      #     M2 -  λ N2 = [    0       | Mf -  λ Nf |     *      |    *       ]
      #                  [    0       |    0       | Mi -  λ Ni |    *       ]
      #                  [    0       |    0       |    0       | Ml -  λ Nl ]

      mr = sum(νr)+nf
      nr = sum(μr)+nf
      ut && klf_right_refineut!(νr, μr, M, N, Q, Z, ctrail = nM-nr, withQ = withQ, withZ = withZ)

      jM2 = nr+1:nr+sum(μ)
      M2 = view(M,:,jM2)
      N2 = view(N,:,jM2)
      withZ ? (Z2 = view(Z,:,jM2)) : (Z2 = nothing)
      νi, νl, μl = klf_left_refine!(ν, μ, M2, N2, Q, Z2, tol1, roff = mr,    
                                    withQ = withQ, withZ = withZ, fast = fast, ut = ut)
   else

      # Reduce M-λN to a KLF exhibiting the left and finite structures
      #                  [ Mri - λ Nri  |   *        |     *      ]
      #      M1 - λ N1 = [     0        | Mf -  λ Nf |     *      ]
      #                  [     0        |    0       | Ml -  λ Nl ]

      ν, μ, nf, νl, μl, tol1 = klf_left!(n, m, p, M, N, Q, Z, atol = atol1, rtol = rtol,  
                                         withQ = withQ, withZ = withZ, fast = fast)
      (mM == 0 || nM == 0) && (return  M, N, Q, Z, ν, μ, ν[1:0], nf, νl, μl)
   
      ut && klf_left_refineut!(νl, μl, M, N, Q, Z, roff = mM-sum(νl), coff = nM-sum(μl), withQ = withQ, withZ = withZ)

      # Reduce Mri-λNri to a KLF exhibiting the right and infinite structures and update M1 - λ N1 to
      #                  [ Mr - λ Nr  |   *        |     *      |    *       ]
      #     M2 -  λ N2 = [    0       | Mi -  λ Ni |     *      |    *       ]
      #                  [    0       |    0       | Mf -  λ Nf |    *       ]
      #                  [    0       |    0       |    0       | Ml -  λ Nl ]

      iM11 = 1:sum(ν)
      M1 = view(M,iM11,:)
      N1 = view(N,iM11,:)
      νr, μr, νi = klf_right_refine!(ν, μ, M1, N1, Q, Z, tol1, ctrail = nM-sum(μ),   
                                     withQ = withQ, withZ = withZ, fast = fast, ut = ut)
   end
   return  M, N, Q, Z, νr, μr, νi, nf, νl, μl                                             
end
"""
    sklf_right(A, E, B, C, D; fast = true, ut = false, atol1 = 0, atol2 = 0, rtol, withQ = true, withZ = true) -> (F, G, Q, Z, νr, μr, nf, ν, μ)

Reduce the structured matrix pencil `M - λN` 

               | A-λE | B | 
      M - λN = |------|---|
               |  C   | D |  

to an equivalent form `F - λG = Q'*(M - λN)*Z` using 
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
function sklf_right(A::Union{AbstractMatrix,Missing}, E::Union{AbstractMatrix,UniformScaling{Bool},Missing}, 
   B::Union{AbstractVecOrMat,Missing}, C::Union{AbstractMatrix,Missing}, D::Union{AbstractVecOrMat,Missing}; 
   atol1::Real = ismissing(A) ? zero(real(eltype(D))) : zero(real(eltype(A))), 
   atol2::Real = ismissing(A) ? zero(real(eltype(D))) : zero(real(eltype(A))), 
   rtol::Real = (ismissing(A) ? 1 : min(size(A)...))*eps(real(float(ismissing(A) ? one(real(eltype(D))) : one(real(eltype(A))))))*iszero(min(atol1,atol2)), 
   fast::Bool = true, ut::Bool = false, withQ::Bool = true, withZ::Bool = true) 
   
   # Step 0: Reduce to the standard form
   M, N, Q, Z, n, m, p = sreduceBF(A, E, B, C, D, atol = atol2, rtol = rtol, fast = fast, withQ = withQ, withZ = withZ)

   νr, μr, nf, ν, μ, tol1 = klf_right!(n, m, p, M, N, Q, Z, atol = atol1, rtol = rtol,  
                                       withQ = withQ, withZ = withZ, fast = fast)

   if ut
      mM, nM = size(M)
      mr = sum(νr)+nf
      nr = sum(μr)+nf
      klf_right_refineut!(νr, μr, M, N, Q, Z, ctrail = nM-nr, withQ = withQ, withZ = withZ)
      klf_left_refineut!(ν, μ, M, N, Q, Z, roff = mM-sum(ν), coff = nM-sum(μ), withQ = withQ, withZ = withZ)
   end
   return M, N, Q, Z, νr, μr, nf, ν, μ

end
"""
    sklf_left(A, E, B, C, D; fast = true, ut = false, atol1 = 0, atol2 = 0, rtol, withQ = true, withZ = true) -> (F, G, Q, Z, ν, μ, nf, νl, μl)

Reduce the structured matrix pencil `M - λN` 

               | A-λE | B | 
      M - λN = |------|---|
               |  C   | D |  

to an equivalent form `F - λG = Q'*(M - λN)*Z` using 
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
function sklf_left(A::Union{AbstractMatrix,Missing}, E::Union{AbstractMatrix,UniformScaling{Bool},Missing}, 
   B::Union{AbstractVecOrMat,Missing}, C::Union{AbstractMatrix,Missing}, D::Union{AbstractVecOrMat,Missing}; 
   atol1::Real = ismissing(A) ? zero(real(eltype(D))) : zero(real(eltype(A))), 
   atol2::Real = ismissing(A) ? zero(real(eltype(D))) : zero(real(eltype(A))), 
   rtol::Real = (ismissing(A) ? 1 : min(size(A)...))*eps(real(float(ismissing(A) ? one(real(eltype(D))) : one(real(eltype(A))))))*iszero(min(atol1,atol2)), 
   fast::Bool = true, ut::Bool = false, withQ::Bool = true, withZ::Bool = true) 
   
   # Step 0: Reduce to the standard form
   M, N, Q, Z, n, m, p = sreduceBF(A, E, B, C, D, atol = atol2, rtol = rtol, fast = fast, withQ = withQ, withZ = withZ)

   ν, μ, nf, νl, μl, tol1 = klf_left!(n, m, p, M, N, Q, Z, atol = atol1, rtol = rtol,  
                                      withQ = withQ, withZ = withZ, fast = fast)

   if ut
      mM, nM = size(M)
      klf_left_refineut!(νl, μl, M, N, Q, Z, roff = mM-sum(νl), coff = nM-sum(μl), withQ = withQ, withZ = withZ)
      klf_right_refineut!(ν, μ, M, N, Q, Z, ctrail = nM-sum(μ), withQ = withQ, withZ = withZ)
   end
   return M, N, Q, Z, ν, μ, nf, νl, μl
end
"""
    sklf_right!(A, E, B, F, C, G, D, H; fast = true, atol1 = 0, atol2 = 0, atol3 = 0, rtol, withQ = true, withZ = true) -> (Q, Z, nc)

Reduce the partitioned full row rank matrix pencil 

      M - λN = | B-λF | A-λE | n
                  m      n     
  
with `A-λE` regular, to an equivalent form `Mt - λNt = Q'*(M - λN)*Z = | Bt-λFt | At-λEt |` 
using orthogonal or unitary transformation matrices `Q` and `Z`, 
such that `M - λN` is transformed into a Kronecker-like form `| Bt-λFt | At-λEt |` exhibiting its 
right, finite and infinite Kronecker structures (also known as the strong controllability form):
 
                             | Bc-λFc | Ac-λEc     *     | nc
      | Bt-λFt | At-λEt | =  |--------|------------------|
                             |   0    |   0     Auc-λEuc | n-nc
                                 m        nc      n-nc   

The matrices `Bt`, `Ft`, `At`, `Et`, determined from `[Bt At] = Q'*[B A]*Z`, `[Ft Et] = Q'*[F E]*Z`, 
are returned in `B`, `F`, `A` and `E`, respectively. 
Furthermore, the matrices `Ct`, `Dt`, `Gt`, `Ht`, determined from the compatibly partitioned matrices
`[Dt Ct] = [D C]*Z` and `[Ht Gt] = [D C]*Z`, are returned in `C`, `D`, `G` and `H`, respectively 
(unless `C`, `D`, `G` and `H` are set to `missing`).


The subpencil `| Bc-λFc | Ac-λEc |` has full row rank `nc` and the `(n-nc) x (n-nc)` 
subpencil `Auc-λEuc` contains the finite and infinite eigenvalues of `M - λN` 
(also called the uncontrollable eigenvalues of `M - λN`).  

The `(m+n) x (m+n)` orthogonal matrix `Z` has the partitioned form

           | Z11 |  0  |  *  | m 
       Z = |-----|-----|-----|
           |  *  |  *  |  *  | n
              m    nc    n-nc

with the leading `m x m` submatrix `Z11` invertible and upper triangular. 

`Note:` If `Ct-λGt` is partitioned as  `[ Cc-λGc | Cuc-λGuc ]`, then the following relation is fulfilled

      (Cc-λGc)*inv(λEc-Ac)(Bc-λFc) + Dt-λHt = ((C-λG)*inv(λE-A)(B-λF) + D-λH)*Z11

and the structured pencil linearization (Ac-λEc,Bc-λFc,Cc-λGc,Dt-λHt) is `strongly controllable`. 

The keyword arguments `atol1`, `atol2`, , `atol3`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of `A` and `B`, the absolute tolerance for the nonzero elements of `E` and `F`, 
and the relative tolerance for the nonzero elements of `A`, `B`, `E` and `F`.  
The reduction is performed using rank decisions based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.

The performed orthogonal or unitary left transformations are accumulated in the matrix `Q` if `withQ = true`. 
Otherwise, `Q` is set to `nothing`.   
The performed orthogonal or unitary right transformations are accumulated in the matrix `Z` if `withZ = true`. 
If `withZ = false`, `Z` contains the upper triangular matrix `Z11`.   
"""
function sklf_right!(A::AbstractMatrix{T1}, E::AbstractMatrix{T1}, B::AbstractVecOrMat{T1}, F::AbstractVecOrMat{T1}, 
                     C::Union{AbstractMatrix{T1},T2}, G::Union{AbstractMatrix{T1},T2},
                     D::Union{AbstractVecOrMat{T1},T2}, H::Union{AbstractVecOrMat{T1},T2}; 
                     fast::Bool = true, atol1::Real = zero(real(T1)), atol2::Real = zero(real(T1)), atol3::Real = zero(real(T1)), 
                     rtol::Real = ((size(A,1)+2)*eps(real(float(one(T1)))))*iszero(max(atol1,atol2,atol3)), 
                     withQ::Bool = true, withZ::Bool = true) where {T1 <: BlasFloat, T2 <: Missing}
   n = LinearAlgebra.checksquare(A)
   n == LinearAlgebra.checksquare(E) || throw(DimensionMismatch("A and E must have the same dimensions"))          
   n1, m = typeof(B) <: AbstractVector ? (length(B),1) : size(B)
   n == n1 || throw(DimensionMismatch("A and B must have the same number of rows"))
   n2, m2 = typeof(F) <: AbstractVector ? (length(F),1) : size(F)
   (n,m) == (n2,m2) || throw(DimensionMismatch("B and F must have the same dimensions"))
   if !ismissing(C) 
      (p,n2) = size(C)
      n == n2 || throw(DimensionMismatch("A and C must have the same number of columns"))
      (p,n) == size(G) || throw(DimensionMismatch("C and G must have the same dimensions"))
      p1, m1 = typeof(D) <: AbstractVector ? (length(D),1) : size(D)
      (p,m) == (p1,m1) || throw(DimensionMismatch("D must have the same row dimension as C and same column dimension as B"))
      p1, m1 = typeof(H) <: AbstractVector ? (length(H),1) : size(H)
      (p,m) == (p1,m1) || throw(DimensionMismatch("D and H must have the same dimensions"))
   end
   mn = m+n
   (m == 0 || n == 0) && (return withQ ? Matrix{T1}(I,n,n) : nothing,  withZ ? Matrix{T1}(I,mn,mn) : Matrix{T1}(I,m,m), 0)

   BAt, FEt, Q, Z, νr, μr, nf, ν, μ = klf_right([B A], [F E]; fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol, withQ = withQ, withZ = true) 
   ν == μ || error("The pencil [B-λF A-λE] has no full row rank")

   nc = sum(νr)
   mnc = sum(μr)
   ir = 1:nc
   jb = 1:m
   ja = m+1:mn
   T1 <: Complex ? trans = 'C' : trans = 'T' 

   ind = [m+1:mnc;jb]

   Z1, tau = LinearAlgebra.LAPACK.gerqf!(Z[jb,ind]) 
   BAt[ir,ind] = LinearAlgebra.LAPACK.ormrq!('R', trans, Z1, tau, BAt[ir,ind])
   FEt[ir,ind] = LinearAlgebra.LAPACK.ormrq!('R', trans, Z1, tau, FEt[ir,ind])
   Z[ja,ind] = LinearAlgebra.LAPACK.ormrq!('R', trans, Z1, tau, Z[ja,ind] )
   Z[jb,1:mnc] = [triu(Z1[:,mnc-m+1:mnc]) zeros(T1,m,nc)]
   
   ia = 1:n
   B[:,:] = BAt[ia,jb]
   A[:,:] = BAt[ia,ja]
   F[:,:] = FEt[ia,jb]
   E[:,:] = FEt[ia,ja]
   if !ismissing(C) 
      DC = [ D C ]*Z
      HG = [H G]*Z
      D[:,:] = DC[:,jb]
      C[:,:] = DC[:,ja]
      H[:,:] = HG[:,jb]
      G[:,:] = HG[:,ja]
   end
   withZ || (Z = Z[jb,jb])
 
   return Q, Z, nc
end
"""
    sklf_left!(A, E, C, G, B, F, D, H; fast = true, atol1 = 0, atol2 = 0, atol3 = 0, rtol, withQ = true, withZ = true) -> (Q, Z, no)

Reduce the partitioned full row rank matrix pencil 

    M - λN = | A-λE | n
             | C-λG | p
                n     
with `A-λE` regular, to an equivalent form `F - λG = Q'*(M - λN)*Z` 
using orthogonal or unitary transformation matrices `Q` and `Z` 
such that `M - λN` is transformed into a staircase form exhibiting the left, 
finite and infinite Kronecker structures (also known as the strong observability form):

                    | Auo-λEuo  |   *    | n-no
      | At-λEt |    |   0       | Ao-λEo | no
      |--------| =  |-----------|--------|
      | Ct-λGt |    |   0       | Co-λGo | p
                       n-no         no

The matrices `Ct`, `Gt`, `At`, `Et`, determined from `[At; Ct] = Q'*[A; C]*Z`, `[Et; Gt] = Q'*[E; G]*Z`,  
are returned in `C`, `G`, `A` and `E`, respectively. 
Furthermore, the matrices `Bt`, `Dt`, `Ft`, `Ht` determined from the compatibly partitioned
`[Bt; Dt] = Q'*[B; D]` and `[Ft; Ht] = Q'*[F; H]`, are returned in `B`, `D`, `F`, and `H`, respectively
(unless `B`, `D`, `F` and `H` are set to `missing`).

The subpencil 

       | Ao-λEo |   
       |--------|
       | Co-λGo |

has full column rank `no` and the `(n-no) x (n-no)` subpencil `Auo-λEuo` contains the finite and infinite 
eigenvalues of `M - λN` (also called the unobservable eigenvalues of `M - λN`).  

The  `(n+p) x (n+p)` orthogonal matrix `Q` has the partitioned form

           |  *  |  *  |  *   | n 
       Q = |-----|-----|------|
           |  *  |  0  | Q22  | p
             n-no  no     p

with the `p x p` trailing submatrix `Q22` invertible and upper triangular. 

`Note:` If `Bt-λFt` is partitioned as  `[ Buo-λFuo | Bo-λFo ]`, then the following relation is fulfilled

      (Co-λGo)*inv(λEo-Ao)(Bo-λFo) + Dt-λHt = Q22'*((C-λG)*inv(λE-A)(B-λF) + D-λH)

and the structured pencil linearization (Ao-λEo,Bo-λFo,Co-λGo,Dt-λHt) is `strongly observable`. 

The keyword arguments `atol1`, `atol2`, , `atol3`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of `A` and `C`, the absolute tolerance for the nonzero elements of `E` and `G`, 
and the relative tolerance for the nonzero elements of `A`, `C`, `E` and `G`.  
The reduction is performed using rank decisions based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.

The performed orthogonal or unitary left transformations are accumulated in the matrix `Q` if `withQ = true`. 
If `withQ = false`, `Q` contains the upper triangular matrix `Q22`.   
The performed orthogonal or unitary right transformations are accumulated in the matrix `Z` if `withZ = true`. 
Otherwise, `Z` is set to `nothing`.   
"""
function sklf_left!(A::AbstractMatrix{T1}, E::AbstractMatrix{T1}, C::AbstractMatrix{T1}, G::AbstractMatrix{T1}, 
                     B::Union{AbstractVecOrMat{T1},T2}, F::Union{AbstractVecOrMat{T1},T2},
                     D::Union{AbstractVecOrMat{T1},T2}, H::Union{AbstractVecOrMat{T1},T2}; 
                     fast::Bool = true, atol1::Real = zero(real(T1)), atol2::Real = zero(real(T1)), atol3::Real = zero(real(T1)), 
                     rtol::Real = ((size(A,1)+2)*eps(real(float(one(T1)))))*iszero(max(atol1,atol2,atol3)), 
                     withQ::Bool = true, withZ::Bool = true) where {T1 <: BlasFloat, T2 <: Missing}
   n = LinearAlgebra.checksquare(A)
   n == LinearAlgebra.checksquare(E) || throw(DimensionMismatch("A and E must have the same dimensions"))          
   p, n1 = size(C)
   n == n1 || throw(DimensionMismatch("A and C must have the same number of columns"))
   (p,n) == size(G) || throw(DimensionMismatch("C and G must have the same dimensions"))
   if !ismissing(C) 
      n2, m = typeof(B) <: AbstractVector ? (length(B),1) : size(B)
      n == n2 || throw(DimensionMismatch("A and B must have the same number of rows"))
      n2, m2 = typeof(F) <: AbstractVector ? (length(F),1) : size(F)
      (n,m) == (n2,m2) || throw(DimensionMismatch("B and F must have the same dimensions"))
      p1, m1 = typeof(D) <: AbstractVector ? (length(D),1) : size(D)
      (p,m) == (p1,m1) || throw(DimensionMismatch("D must have the same row dimension as C and same column dimension as B"))
      p1, m1 = typeof(H) <: AbstractVector ? (length(H),1) : size(H)
      (p,m) == (p1,m1) || throw(DimensionMismatch("D and H must have the same dimensions"))
   end
   np = n+p
   (p == 0 || n == 0) && (return withQ ? Matrix{T1}(I,n,n) : nothing, withZ ? Matrix{T1}(I,np,np) : Matrix{T1}(I,p,p), 0)

   ACt, EGt, Q, Z, ν, μ, nf, νl, μl  = klf_left([A; C], [E; G]; fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol, withQ = true, withZ = withZ) 
   ν == μ || error("The pencil [A-λE; C-λG ] has no full column rank")

   npo = sum(νl)
   no = sum(μl)
   ic = n+1:np
   ia = 1:n
   it = n-no+1:np
   jt = n-no+1:n
   T1 <: Complex ? trans = 'C' : trans = 'T' 
   Q22 = view(Q,ic,it)
    _, tau = LinearAlgebra.LAPACK.gerqf!(Q22) 
   LinearAlgebra.LAPACK.ormrq!('L', 'N', Q22, tau, view(ACt,it,jt))
   LinearAlgebra.LAPACK.ormrq!('L', 'N', Q22, tau, view(EGt,it,jt))
   LinearAlgebra.LAPACK.ormrq!('R', trans, Q22, tau, view(Q,ia,it))
   Q22[:,:] = [zeros(T1,p,no) triu(Q[ic,ic])]
   
   ia = 1:n
   C[:,:] = ACt[ic,ia]
   A[:,:] = ACt[ia,ia]
   G[:,:] = EGt[ic,ia]
   E[:,:] = EGt[ia,ia]
   if !ismissing(B) 
      BD = Q'*[B; D]
      FH = Q'*[F; H]
      B[:,:] = BD[ia,:]
      D[:,:] = BD[ic,:]
      F[:,:] = FH[ia,:]
      H[:,:] = FH[ic,:]
   end
   withQ || (Q = Q[ic,ic])
 
   return Q, Z, no
end
"""
    sklf_right!(A, E, B, C; fast = true, atol1 = 0, atol2 = 0, atol3 = 0, rtol, withQ = true, withZ = true) -> (Q, Z, νr, nc, nfuc, niuc)

Reduce the partitioned full row rank matrix pencil 

      M - λN = | B | A-λE | n
                 m    n     
  
with `A-λE` regular, to an equivalent form `F - λG = Q'*(M - λN)*diag(I,Z)` using orthogonal or unitary transformation matrices `Q` and `Z` 
such that `M - λN` is transformed into a Kronecker-like form `| Bt | At-λEt |` exhibiting its 
right, finite and infinite Kronecker structures (also known as the generalized controllability staircase form):
 
                         |  Bc | Ac-λEc     *          *      | nc
      | Bt | At-λEt | =  |-----|------------------------------|
                         |  0  |  0     Afuc-λEfuc     *      | nfuc
                         |  0  |  0         0      Aiuc-λEiuc | niuc
                            m     nc       nfuc        niuc

`Bt = Q'*B`, `At = Q'*A*Z`and `Et = Q'*E*Z` are returned in `B`, `A` and `E`, respectively, and `C*Z` is returned in `C` (unless `C = missing`).                 

The subpencil `| Bc | Ac-λEc |` has full row rank `nc`, is in a staircase form, and contains the right Kronecker indices of `M - λN`. 
The `nr`-dimensional vector `νr` contains the row and column dimensions of the blocks
of the staircase form  `| Bc | Ac-λI |` such that `i`-th block has dimensions `νr[i] x νr[i-1]` (with `νr[0] = m`) and 
has full row rank. The difference `νr[i-1]-νr[i]` for `i = 1, 2, ..., nr` is the number of elementary Kronecker blocks
of size `(i-1) x i`.

The `nfuc x nfuc` subpencil `Afuc-λEfuc` contains the finite eigenvalues of `M - λN` (also called the uncontrollable finite eigenvalues of `A - λE`).  

The `niuc x niuc` subpencil `Aiuc-λEiuc` contains the infinite eigenvalues of `M - λN` (also called the uncontrollable infinite eigenvalues of `A - λE`).  

The keyword arguments `atol1`, `atol2`, `atol3`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of `A`, the absolute tolerance for the nonzero elements of `E`, the absolute tolerance for the nonzero elements of `B`,  
and the relative tolerance for the nonzero elements of `A`, `E` and `B`.  
The reduction is performed using rank decisions based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.

The performed orthogonal or unitary left transformations are accumulated in the matrix `Q` if `withQ = true`. 
Otherwise, `Q` is set to `nothing`.   
The performed orthogonal or unitary right transformations are accumulated in the matrix `Z` if `withZ = true`. 
Otherwise, `Z` is set to `nothing`.   
"""
function sklf_right!(A::AbstractMatrix{T}, E::AbstractMatrix{T}, B::AbstractVecOrMat{T}, C::Union{AbstractMatrix{T},Missing}; fast::Bool = true, 
                   atol1::Real = zero(real(T)), atol2::Real = zero(real(T)), atol3::Real = zero(real(T)), 
                   rtol::Real = ((size(A,1)+1)*eps(real(float(one(T)))))*iszero(max(atol1,atol2,atol3)), 
                   withQ::Bool = true, withZ::Bool = true) where T <: BlasFloat
   n = LinearAlgebra.checksquare(A)
   n == LinearAlgebra.checksquare(E) || throw(DimensionMismatch("A and E must have the same dimensions"))          
   n1, m = typeof(B) <: AbstractVector ? (length(B),1) : size(B)
   n == n1 || throw(DimensionMismatch("A and B must have the same number of rows"))
   (!ismissing(C) && n != size(C,2)) && throw(DimensionMismatch("A and C must have the same number of columns"))
   
   maxmn = max(m,n)
   νr = Vector{Int}(undef,maxmn)
   nfu = 0
   niu = 0
   tol1 = atol1
   withQ ? Q = Matrix{T}(I,n,n) : Q = nothing
   withZ ? Z = Matrix{T}(I,n,n) : Z = nothing
 
   # fast returns for null dimensions
   if m == 0 && n == 0 
      return Q, Z, νr, n, nfu, niu
   elseif n == 0 
      νr[1] = 0
      return Q, Z, νr[1:1], n, nfu, niu
   end

   tolA = max(atol1, rtol*opnorm(A,1))
   typeof(B) <: AbstractVector ? tolB = max(atol3, rtol*norm(B,1)) : tolB = max(atol3, rtol*opnorm(B,1))

   ρ1 = _sreduceB!(A, E, B, Q, tolB; fast = fast, withQ = withQ)
   ρ1 == n && (return Q, Z, [ρ1], n, nfu, niu)

   n1, m1, p1 = _preduceBF!(A, E, Q, Z, missing, C; roff = ρ1, atol = atol2, rtol = rtol, fast = fast, withQ = withQ, withZ = withZ) 
 
   mrinf = ρ1
   nrinf = 0
   rtrail = 0
   ctrail = 0
   niu = 0
   while p1 > 0
      # Step 1 & 2: Dual algorithm PREDUCE
      τ, ρ  = _preduce2!(n1, m1, p1, A, E, Q, Z, tolA, missing, C; fast = fast, 
                         roff = mrinf, coff = nrinf, rtrail = rtrail, ctrail = ctrail, withQ = withQ, withZ = withZ)
      ρ+τ == p1 || error("| B | A-λE | has no full row rank")
      ctrail += p1
      rtrail += p1
      niu += p1
      n1 -= ρ
      p1 = ρ
      m1 -= τ 
   end
   
   if ρ1 > 0
      i = 1 
      νr[1] = ρ1
   else
      return Q, Z, νr[1:0], 0, n1, niu
   end
   if m1 > 0
      imA11 = 1:n-rtrail
      A11 = view(A,imA11,1:n)
      E11 = view(E,imA11,1:n)
      ismissing(C) ? C1 = missing : (C1 = view(C,:,1:n))
   end
   nc = ρ1
   nfu = n1
   while m1 > 0
      # Step 3: Particular case of the standard algorithm PREDUCE
      ρ = _preduce3!(nfu, m1, A11, E11, Q, Z, tolA, missing, C1, fast = fast, coff = nrinf, roff = mrinf, ctrail = ctrail,  withQ = withQ, withZ = withZ)
      ρ == 0 && break
      i += 1
      νr[i] = ρ
      mrinf += ρ
      nrinf += m1
      nc += ρ
      nfu -= ρ
      m1 = ρ
   end

   return Q, Z, νr[1:i], nc, nfu, niu
end
"""
    sklf_rightfin!(A, E, B, C; fast = true, atol1 = 0, atol2 = 0,  
                   rtol, withQ = true, withZ = true) -> (Q, Z, νr, nc, nuc)

Reduce the partitioned full row rank matrix pencil 

      M - λN = | B | A-λE | n
                 m    n     
  
with `A-λE` regular, to an equivalent form `F - λG = Q'*(M - λN)*diag(I,Z)` 
using orthogonal or unitary transformation matrices `Q` and `Z` 
such that `M - λN` is transformed into a staircase form `| Bt | At-λEt |` exhibiting the separation of its finite
eigenvalues:
 
                         |  Bc | Ac-λEc     *    | nc
      | Bt | At-λEt | =  |-----|-----------------|
                         |  0  |  0     Auc-λEuc | nuc
                            m     nc      nuc      

`Bt = Q'*B`, `At = Q'*A*Z`and `Et = Q'*E*Z` are returned in `B`, `A` and `E`, respectively, and `C*Z` is returned in `C` (unless `C = missing`). 
The resulting `Et` is upper triangular. If `E` is already upper triangular, then 
the preliminary reduction of `E` to upper triangular form is not performed.                

The subpencil `| Bc | Ac-λEc |` has full row rank `nc` for all finite values of `λ`, is in a staircase form, and, 
if E is invertible, contains the right Kronecker indices of `M - λN`. 
The `nr`-dimensional vector `νr` contains the row and column dimensions of the blocks
of the staircase form  `| Bc | Ac-λI |` such that `i`-th block has dimensions `νr[i] x νr[i-1]` (with `νr[0] = m`) and 
has full row rank. If E is invertible, the difference `νr[i-1]-νr[i]` for `i = 1, 2, ..., nr` is the number of elementary Kronecker blocks
of size `(i-1) x i`.

The `nuc x nuc` subpencil `Auc-λEuc` contains the finite eigenvalues of `M - λN` (also called the uncontrollable finite eigenvalues of `A - λE`). 
If E is singular, `Auc-λEuc` may also contain a part of the infinite eigenvalues of `M - λN` (also called the uncontrollable infinite eigenvalues of `A - λE`).

The keyword arguments `atol1`, `atol2` and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of `A`, the absolute tolerance for the nonzero elements of `B`,  
and the relative tolerance for the nonzero elements of `A` and `B`.  
The reduction is performed using rank decisions based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.

The performed orthogonal or unitary left transformations are accumulated in the matrix `Q` if `withQ = true`. 
Otherwise, `Q` is set to `nothing`.   
The performed orthogonal or unitary right transformations are accumulated in the matrix `Z` if `withZ = true`. 
Otherwise, `Z` is set to `nothing`.   

`Note:` This function, called with reversed input parameters `E` and `A` (i.e., instead `A` and `E`), performs the 
separation all infinite and nonzero finite eigenvalues of the pencil `M - λN`.
"""
function sklf_rightfin!(A::AbstractMatrix{T}, E::AbstractMatrix{T}, B::AbstractVecOrMat{T}, C::Union{AbstractMatrix{T},Missing}; 
                        fast::Bool = true, atol1::Real = zero(real(T)), atol2::Real = zero(real(T)), 
                        rtol::Real = ((size(A,1)+1)*eps(real(float(one(T)))))*iszero(max(atol1,atol2)), 
                        withQ::Bool = true, withZ::Bool = true) where T <: BlasFloat
   n = LinearAlgebra.checksquare(A)
   n == LinearAlgebra.checksquare(E) || throw(DimensionMismatch("A and E must have the same dimensions"))          
   n1, m = typeof(B) <: AbstractVector ? (length(B),1) : size(B)
   n == n1 || throw(DimensionMismatch("A and B must have the same number of rows"))
   (!ismissing(C) && n != size(C,2)) && throw(DimensionMismatch("A and C must have the same number of columns"))
   
   νr = Vector{Int}(undef,max(n,1))
   withQ ? Q = Matrix{T}(I,n,n) : Q = nothing
   withZ ? Z = Matrix{T}(I,n,n) : Z = nothing

   # fast returns for null dimensions
   m == 0 && n == 0 && (return Q, Z, νr[1:0], 0, 0)
   n == 0 && (return Q, Z, [0], 0, 0)

   # Reduce E to upper triangular form if necessary
   istriu(E) || _qrE!(A, E, Q, B; withQ = withQ) 
   m == 0 && (return Q, Z, νr[1:0], 0, n)
  
   tolA = max(atol1, rtol*opnorm(A,1))
   typeof(B) <: AbstractVector ? tolB = max(atol2, rtol*norm(B,1)) : tolB = max(atol2, rtol*opnorm(B,1))

   i = 0
   init = true
   roff = 0
   coff = 0
   nc = 0
   nuc = n
   while m > 0 && nc < n
      init = (i == 0)
      init ? tol = tolB : tol = tolA
      # Step 3: Particular case of the standard algorithm PREDUCE
      ρ = _sreduceBAE!(nuc, m, A, E, B, C, Q, Z, tol, fast = fast, init = init, roff = roff, coff = coff, withQ = withQ, withZ = withZ)
      ρ == 0 && break
      i += 1
      νr[i] = ρ
      roff += ρ
      init || (coff += m)
      nc += ρ
      nuc -= ρ
      m = ρ
   end
   
   return Q, Z, νr[1:i], nc, nuc
end
"""
    sklf_rightfin2!(A, E, B, m1, C; fast = true, atol1 = 0, atol2 = 0,  
                   rtol, withQ = true, withZ = true) -> (Q, Z, νr, nc, nuc)

Reduce the partitioned full row rank matrix pencil 

      M - λN = [ B | A-λE ] n  = [ B1  B2 | A-λE ] n
                 m    n            m1 m-m1   n
  
with `A-λE` regular, to an equivalent form `F - λG = Q'*(M - λN)*diag(I,Z)` 
using orthogonal or unitary transformation matrices `Q` and `Z` 
such that `M - λN` is transformed into a staircase form `[ Bt | At-λEt ]` exhibiting 
the separation of its finite eigenvalues:
 
                         [  Bc | Ac-λEc     *    ] nc
      [ Bt | At-λEt ] =  [-----|-----------------]
                         [  0  |  0     Auc-λEuc ] nuc
                            m     nc      nuc      

`Bt = Q'*B`, `At = Q'*A*Z`and `Et = Q'*E*Z` are returned in `B`, `A` and `E`, respectively, and `C*Z` is returned in `C` (unless `C = missing`). 
The resulting `Et` is upper triangular. If `E` is already upper triangular, then 
the preliminary reduction of `E` to upper triangular form is not performed.                

The subpencil `[ Bc | Ac-λEc ]` has full row rank `nc` for all finite values of `λ`, with
`[ Bc | Ac ] = [ Bc1 Bc2 | Ac]` in a staircase form

                        m1  m-m1  νr[1] νr[2] . . .  νr[p-2] 
                      [ B11 B12 | A11   A12   . . .  A1,p-2  A1,p-1  A1p ]  νr[1]
                      [  0  B22 | A21   A22   . . .  A2,p-2  A2,p-1  A2p ]  νr[2]
                      [  0   0  | A31   A32   . . .  A3,p-2  A3,p-1  A3p ]  νr[3]
                      [  0   0  |  0    A42   . . .  A4,p-2  A4,p-1  A4p ]  νr[4]
    [ Bc1 Bc2 | Ac] = [  0   0  |  .     .    . . .    .       .      .  ]   .
                      [  0   0  |  .     .      . .    .       .      .  ]   .
                      [  0   0  |  .     .        .    .       .      .  ]   .
                      [  0   0  |  0     0    . . .  Ap,p-2  Ap,p-1  App ]  νr[p]

where the blocks  `B11`, `B22`, `A31`, ..., `Ap,p-2`  have full row ranks.
The `p`-dimensional vector `νr`, with `p = 2nr` even,  contains the row dimensions of the 
blocks.  The difference `νr[2i-1]+νr[2i]-νr[2i+1]-νr[2i+2]` for `i = 1, 2, ..., nr` is the 
number of elementary Kronecker blocks of size `(i-1) x i`.

The `nuc x nuc` subpencil `Auc-λEuc` contains the finite eigenvalues of `M - λN` (also called the uncontrollable finite eigenvalues of `A - λE`). 
If E is singular, `Auc-λEuc` may also contain a part of the infinite eigenvalues of `M - λN` (also called the uncontrollable infinite eigenvalues of `A - λE`).

The keyword arguments `atol1`, `atol2` and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of `A`, the absolute tolerance for the nonzero elements of `B`,  
and the relative tolerance for the nonzero elements of `A` and `B`.  
The reduction is performed using rank decisions based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.

The performed orthogonal or unitary left transformations are accumulated in the matrix `Q` if `withQ = true`. 
Otherwise, `Q` is set to `nothing`.   
The performed orthogonal or unitary right transformations are accumulated in the matrix `Z` if `withZ = true`. 
Otherwise, `Z` is set to `nothing`.   

Method: The implemented algorithm [1] represents a specialization of the 
controllability staircase algorithm of [2] to the special structure of the input matrix `B = [B1,B2]`.

References

[1] Varga, A. Reliable algorithms for computing minimal dynamic covers for descriptor systems.
    Proc. of MTNS'04, Leuven, Belgium, 2004.

[2] Varga, A. Computation of Irreducible Generalized State-Space Realizations.
    Kybernetika, vol. 26, pp. 89-106, 1990.

"""
function sklf_rightfin2!(A::AbstractMatrix{T}, E::AbstractMatrix{T}, B::AbstractMatrix{T}, m1::Int, C::Union{AbstractMatrix{T},Missing}; 
                        fast::Bool = true, atol1::Real = zero(real(T)), atol2::Real = zero(real(T)), 
                        rtol::Real = ((size(A,1)+1)*eps(real(float(one(T)))))*iszero(max(atol1,atol2)), 
                        withQ::Bool = true, withZ::Bool = true) where T <: BlasFloat
   n = LinearAlgebra.checksquare(A)
   n == LinearAlgebra.checksquare(E) || throw(DimensionMismatch("A and E must have the same dimensions"))          
   n1, m = size(B)
   n == n1 || throw(DimensionMismatch("A and B must have the same number of rows"))
   (!ismissing(C) && n != size(C,2)) && throw(DimensionMismatch("A and C must have the same number of columns"))
   (m1 <= m && m1 >= 0) || throw(DimensionMismatch("B1 must have at most $m columns"))
   
   withQ ? Q = Matrix{T}(I,n,n) : Q = nothing
   withZ ? Z = Matrix{T}(I,n,n) : Z = nothing

   # Reduce E to upper triangular form if necessary
   n == 0 || istriu(E) || _qrE!(A, E, Q, B; withQ = withQ) 
   
   # fast returns for null dimensions
   if m == 0 && n == 0
      νr = Int[]
      return Q, Z, νr, 0, 0
   elseif n == 0
      νr = [0, 0]
      return Q, Z, νr, 0, 0
   elseif m == 0
      νr = Int[]
      return Q, Z, νr, 0, n
   end

   νr = Vector{Int}(undef,2*n)
    
   tolA = max(atol1, rtol*opnorm(A,1))
   tolB = max(atol2, rtol*opnorm(B,1))

   i = 0
   coff = 0
   nc = 0
   nuc = n
   m2 = m-m1
   tol = tolB 
   while m > 0 && nc < n
      # Step 3: Particular case of the standard algorithm PREDUCE
      ρ1 = _sreduceBAE2!(nuc, m1, A, E, B, C, Q, Z, tol, fast = fast, coff = coff, withQ = withQ, withZ = withZ)
      nuc -= ρ1
      ρ2 = _sreduceBAE2!(nuc, m2, A, E, view(B,1:n,m1+1:m), C, Q, Z, tol, coff = coff, fast = fast, withQ = withQ, withZ = withZ)
      ρ = ρ1+ρ2
      ρ == 0 && break
      i += 2
      νr[i-1] = ρ1
      νr[i] = ρ2
      m = ρ
      nc += ρ
      nuc -= ρ2
      B = view(A,1:n,coff+1:coff+m)
      coff += ρ
      m1 = ρ1
      m2 = ρ2
      tol = tolA
   end   
   return Q, Z, νr[1:i], nc, nuc
end
"""
    sklf_left!(A, E, C, B; fast = true, atol1 = 0, atol2 = 0, atol3 = 0, rtol, withQ = true, withZ = true) -> (Q, Z, μl, no, nfuo, niuo)

Reduce the partitioned full column rank matrix pencil 

      M - λN = | A-λE | n
               |  C   | p
                  n     
  
to an equivalent form `F - λG = diag(Q',I)*(M - λN)*Z` using orthogonal or unitary transformation matrices `Q`  and `Z` 
such that `M - λN` is transformed into a Kronecker-like form  exhibiting its 
infinite, finite  and left Kronecker structures (also known as the generalized observability staircase form):
 
                    | Aiuo-λEiuo     *       |   *    | niuo
                    |    0       Afuo-λEfuo  |   *    | nfuo
      | At-λEt |    |    0           0       | Ao-λEo | no
      |--------| =  |------------------------|--------|
      |  Ct    |    |    0           0       |   Co   | p
                        niuo        nfuo         no

`Ct = C*Z`, `At = Q'*A*Z` and `Et = Q'*E*Z` are returned in `C`, `A` and `E`, respectively, 
and `Q'*B` is returned in `B` (unless `B = missing`).                 

The subpencil `| Ao-λEo |` has full column rank `no`, is in a staircase form, and contains the left Kronecker indices of `M - λN`. 
              `|   Co   |`
The `nl`-dimensional vector `μl` contains the row and column dimensions of the blocks
of the staircase form such that `i`-th block has dimensions `μl[nl-i] x μl[nl-i+1]` (with μl[0] = p) and 
has full column rank. The difference `μl[nl-i]-μl[nl-i+1]` for `i = 1, 2, ..., nl` is the number of elementary Kronecker blocks
of size `i x (i-1)`.

The `niuo x niuo` subpencil `Aiuo-λEiuo` contains the infinite eigenvalues of `M - λN` (also called the unobservable infinite eigenvalues of `A-λE`).  

The `nfuo x nfuo` subpencil `Afuo-λEfuo` contains the finite eigenvalues of `M - λN` (also called the unobservable finite eigenvalues of `A-λE`).  

The keyword arguments `atol1`, `atol2`, `atol3` and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of `A`, the absolute tolerance for the nonzero elements of `E`, 
the absolute tolerance for the nonzero elements of `C`,  and 
the relative tolerance for the nonzero elements of `A`, `E` and `C`.  
The reduction is performed using rank decisions based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.

The performed orthogonal or unitary left transformations are accumulated in the matrix `Q` if `withQ = true`. 
Otherwise, `Q` is set to `nothing`.   
The performed orthogonal or unitary right transformations are accumulated in the matrix `Z` if `withZ = true`. 
Otherwise, `Z` is set to `nothing`.   
"""
function sklf_left!(A::AbstractMatrix{T}, E::AbstractMatrix{T}, C::AbstractMatrix{T}, B::Union{AbstractVecOrMat{T},Missing}; fast::Bool = true, 
                   atol1::Real = zero(real(T)), atol2::Real = zero(real(T)), atol3::Real = zero(real(T)), 
                   rtol::Real = ((size(A,1)+1)*eps(real(float(one(T)))))*iszero(max(atol1,atol2,atol3)), 
                   withQ::Bool = true, withZ::Bool = true) where T <: BlasFloat
   n = LinearAlgebra.checksquare(A)
   n == LinearAlgebra.checksquare(E) || throw(DimensionMismatch("A and E must have the same dimensions"))          
   p, n1 = size(C)
   n == n1 || throw(DimensionMismatch("A and C must have the same number of columns"))
   (!ismissing(B) && n != size(B,1)) && throw(DimensionMismatch("A and B must have the same number of rows"))
   
   maxpn = max(p,n)
   μl = Vector{Int}(undef,maxpn)
   nfu = 0
   niu = 0
   tol1 = atol1
   withQ ? Q = Matrix{T}(I,n,n) : Q = nothing
   withZ ? Z = Matrix{T}(I,n,n) : Z = nothing
 
   # fast returns for null dimensions
   if p == 0 && n == 0
      return Q, Z, μl, n, nfu, niu
   elseif n == 0
      μl[1] = 0
      return Q, Z, μl[1:1], n, nfu, niu
   end

   tolA = max(atol1, rtol*opnorm(A,1))
   tolC = max(atol3, rtol*opnorm(C,Inf))

   ρ1 = _sreduceC!(A, E, C, Z, tolC; fast = fast, withZ = withZ)
   ρ1 == n && (return Q, Z, [ρ1], n, nfu, niu)
   
   # reduce to basic form
   n1, m1, p1 = _preduceBF!(A, E, Q, Z, B, missing; ctrail = ρ1, atol = atol2, rtol = rtol, fast = fast, withQ = withQ, withZ = withZ) 
   
   mrinf = 0
   nrinf = 0
   roff = 0
   coff = 0
   rtrail = 0
   ctrail = ρ1
   niu = 0
   while m1 > 0
      # Steps 1 & 2: Standard algorithm PREDUCE
      τ, ρ = _preduce1!(n1, m1, p1, A, E, Q, Z, tolA, B, missing; fast = fast, 
                        roff = roff, coff = coff, ctrail = ctrail, withQ = withQ, withZ = withZ)
      ρ+τ == m1 || error("| C' | A'-λE' | has no full row rank")
      roff += m1
      coff += m1
      niu += m1
      n1 -= ρ
      m1 = ρ
      p1 -= τ 
   end
  
   if ρ1 > 0
      j = 1 
      μl[1] = ρ1
   else
      return Q, Z, μl[1:0], 0, n1, niu
   end
   
   no = ρ1
   nfu = n1
   while p1 > 0
      # Step 3: Particular form of the dual algorithm PREDUCE
      ρ = _preduce4!(nfu, 0, p1, A, E, Q, Z, tolA, B, missing; fast = fast, roff = roff, coff = coff, rtrail = rtrail, ctrail = ctrail, 
                     withQ = withQ, withZ = withZ) 
      ρ == 0 && break
      j += 1
      μl[j] = ρ
      rtrail += p1
      ctrail += ρ
      no += ρ
      nfu -= ρ
      p1 = ρ
   end
 
   return Q, Z, reverse(μl[1:j]), no, nfu, niu
end
"""
    sklf_leftfin!(A, E, C, B; fast = true, atol1 = 0, atol2 = 0,  
                  rtol, withQ = true, withZ = true) -> (Q, Z, μl, no, nuo)

Reduce the partitioned full column rank matrix pencil 

      M - λN = | A-λE | n
               |  C   | p
                  n     
  
with `A-λE` regular, to an equivalent form `F - λG = Q'*(M - λN)*diag(I,Z)` 
using orthogonal or unitary transformation matrices `Q` and `Z` 
such that `M - λN` is transformed into a staircase form exhibiting the separation of its finite
eigenvalues:
 
                    | Auo-λEuo  |   *    | nuo
      | At-λEt |    |   0       | Ao-λEo | no
      |--------| =  |-----------|--------|
      |  Ct    |    |   0       |   Co   | p
                        nuo         no

`Ct = C*Z`, `At = Q'*A*Z` and `Et = Q'*E*Z` are returned in `C`, `A` and `E`, respectively, 
and `Q'*B` is returned in `B` (unless `B = missing`).   
The resulting `Et` is upper triangular. If `E` is already upper triangular, then 
the preliminary reduction of `E` to upper triangular form is not performed.                

The subpencil `| Ao-λEo |` has full column rank `no` for all finite values of `λ`, is in a staircase form, 
              `|   Co   |`
and, if `E` is nonsingular, contains the left Kronecker indices of `M - λN`. The `nl`-dimensional vector `μl` contains the row and column dimensions of the blocks
of the staircase form such that `i`-th block has dimensions `μl[nl-i] x μl[nl-i+1]` (with μl[0] = p) and 
has full column rank. If `E` is nonsingular, the difference `μl[nl-i]-μl[nl-i+1]` for `i = 1, 2, ..., nl` is the number of elementary Kronecker blocks
of size `i x (i-1)`.

The keyword arguments `atol1`, `atol2` and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of `A`, the absolute tolerance for the nonzero elements of `C`,  
and the relative tolerance for the nonzero elements of `A` and `C`.  
The reduction is performed using rank decisions based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.

The performed orthogonal or unitary left transformations are accumulated in the matrix `Q` if `withQ = true`. 
Otherwise, `Q` is set to `nothing`.   
The performed orthogonal or unitary right transformations are accumulated in the matrix `Z` if `withZ = true`. 
Otherwise, `Z` is set to `nothing`.   

`Note:` This function, called with reversed input parameters `E` and `A` (i.e., instead `A` and `E`), performs the 
separation all infinite and nonzero finite eigenvalues of the pencil `M - λN`.
"""
function sklf_leftfin!(A::AbstractMatrix{T}, E::AbstractMatrix{T}, C::AbstractMatrix{T}, B::Union{AbstractVecOrMat{T},Missing}; 
                       fast::Bool = true, atol1::Real = zero(real(T)), atol2::Real = zero(real(T)), 
                       rtol::Real = ((size(A,1)+1)*eps(real(float(one(T)))))*iszero(max(atol1,atol2)), 
                       withQ::Bool = true, withZ::Bool = true) where T <: BlasFloat

   n = LinearAlgebra.checksquare(A)
   n == LinearAlgebra.checksquare(E) || throw(DimensionMismatch("A and E must have the same dimensions"))          
   p, n1 = size(C)
   n == n1 || throw(DimensionMismatch("A and C must have the same number of columns"))
   (!ismissing(B) && n != size(B,1)) && throw(DimensionMismatch("A and B must have the same number of rows"))

   μl = Vector{Int}(undef,max(n,1))
   withQ ? Q = Matrix{T}(I,n,n) : Q = nothing
   withZ ? Z = Matrix{T}(I,n,n) : Z = nothing

   # fast returns for null dimensions
   p == 0 && n == 0 && (return Q, Z, μl[1:0], 0, 0)
   n == 0 && (return Q, Z, [0], 0, 0)
 
   # Reduce E to upper triangular form if necessary
   istriu(E) || _qrE!(A, E, Q, B; withQ = withQ) 
   p == 0 && (return Q, Z, μl[1:0], 0, n)


   tolA = max(atol1, rtol*opnorm(A,1))
   tolC = max(atol2, rtol*opnorm(C,Inf))
      
   i = 0
   init = true
   rtrail = 0
   ctrail = 0
   no = 0
   nuo = n
   while p > 0 && no < n
      init = (i == 0)
      init ? tol = tolC : tol = tolA
      # Step 3: Particular case of the standard algorithm PREDUCE
      ρ = _sreduceAEC!(nuo, p, A, E, C, B, Q, Z, tol, fast = fast, init = init, 
                       rtrail = rtrail, ctrail = ctrail, withQ = withQ, withZ = withZ)
      ρ == 0 && break
      i += 1
      μl[i] = ρ
      ctrail += ρ
      init || (rtrail += p)
      no += ρ
      nuo -= ρ
      p = ρ
   end

   return Q, Z, reverse(μl[1:i]), no, nuo
end
"""
    sklf_right!(A, B, C; fast = true, atol1 = 0, atol2 = 0, rtol, withQ = true) -> (Q, νr, nc, nuc)

Reduce the partitioned full row rank matrix pencil 

      M - λN = | B | A-λI | n
                 m    n     
  
to an equivalent form `F - λG = Q'*(M - λN)*diag(I,Q)` using orthogonal or unitary transformation matrix `Q`  
such that `M - λN` is transformed into a Kronecker-like form `| Bt | At-λI |` exhibiting its 
right and finite Kronecker structures (also known as the controllability staircase form):
 
                        |  Bc | Ac-λI     *    | nc
      | Bt | At-λI | =  |-----|----------------|
                        |  0  |  0     Auc-λI  | nuc
                           m     nc      nuc

`Bt = Q'*B` and `At = Q'*A*Q` are returned in `B` and `A`, respectively, and `C*Q` is returned in `C` (unless `C = missing`).                 

The subpencil `| Bc | Ac-λI |` has full row rank `nc`, is in a staircase form, and contains the right Kronecker indices of `M - λN`. 
The `nr`-dimensional vector `νr` contains the row and column dimensions of the blocks
of the staircase form  `| Bc | Ac-λI |` such that `i`-th block has dimensions `νr[i] x νr[i-1]` (with `νr[0] = m`) and 
has full row rank. The difference `νr[i-1]-νr[i]` for `i = 1, 2, ..., nr` is the number of elementary Kronecker blocks
of size `(i-1) x i`.

The `nuc x nuc` matrix `Auc` contains the (finite) eigenvalues of `M - λN` (also called the uncontrollable eigenvalues of `A`).  

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of `A`, the absolute tolerance for the nonzero elements of `B`,  and the relative tolerance 
for the nonzero elements of `A` and `B`.  
The reduction is performed using rank decisions based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.

The performed orthogonal or unitary transformations are accumulated in the matrix `Q` if `withQ = true`. 
Otherwise, `Q` is set to `nothing`.   
"""
function sklf_right!(A::AbstractMatrix{T}, B::AbstractVecOrMat{T}, C::Union{AbstractMatrix{T},Missing}; fast::Bool = true, 
                     atol1::Real = zero(real(T)), atol2::Real = zero(real(T)), 
                     rtol::Real = ((size(A,1)+1)*eps(real(float(one(T)))))*iszero(max(atol1,atol2)), 
                     withQ::Bool = true) where T <: BlasFloat
   n = LinearAlgebra.checksquare(A)
   n1, m = typeof(B) <: AbstractVector ? (length(B),1) : size(B)
   n == n1 || throw(DimensionMismatch("A and B must have the same number of rows"))
   (!ismissing(C) && n != size(C,2)) && throw(DimensionMismatch("A and C must have the same number of columns"))
   
   νr = Vector{Int}(undef,max(n,1))
   nu = 0
   tol1 = atol1
   withQ ? Q = Matrix{T}(I,n,n) : Q = nothing
 
   # fast returns for null dimensions
   if m == 0 && n == 0
      return Q, νr[1:0], 0, 0
   elseif n == 0
      νr[1] = 0
      return Q, νr[1:1], 0, 0
   elseif m == 0
      return Q, νr[1:0], 0, n
   end

   tolA = max(atol1, rtol*opnorm(A,1))
   typeof(B) <: AbstractVector ? tolB = max(atol2, rtol*norm(B,1)) : tolB = max(atol2, rtol*opnorm(B,1))
      
   i = 0
   init = true
   roff = 0
   coff = 0
   nc = 0
   nu = n
   while m > 0 && nc < n
      init = (i == 0)
      init ? tol = tolB : tol = tolA
      # Step 3: Particular case of the standard algorithm PREDUCE
      ρ = _sreduceBA!(nu, m, A, B, C, Q, tol, fast = fast, init = init, roff = roff, coff = coff, withQ = withQ)
      ρ == 0 && break
      i += 1
      νr[i] = ρ
      roff += ρ
      init || (coff += m)
      nc += ρ
      nu -= ρ
      m = ρ
   end

   return Q, νr[1:i], nc, nu
end
"""
    sklf_right2!(A, B, m1, C; fast = true, atol1 = 0, atol2 = 0, rtol, withQ = true) -> (Q, νr, nc, nuc)

Reduce the partitioned full row rank matrix pencil 

      M - λN = [ B | A-λI ] n  = [ B1  B2 | A-λI ] n
                 m    n            m1 m-m1   n
  
to an equivalent form `F - λG = Q'*(M - λN)*diag(I,Q)` using orthogonal or unitary transformation matrix `Q`  
such that `M - λN` is transformed into a Kronecker-like form `[ Bt | At-λI ]` exhibiting its 
right and finite Kronecker structures (also known as the controllability staircase form):
 
                        [  Bc | Ac-λI     *    ] nc
      [ Bt | At-λI ] =  [-----|----------------]
                        [  0  |  0     Auc-λI  ] nuc
                           m     nc      nuc

`Bt = Q'*B` and `At = Q'*A*Q` are returned in `B` and `A`, respectively, and `C*Q` is returned in `C` (unless `C = missing`).                 

The subpencil `[ Bc | Ac-λI ]` has full row rank `nc` with `[ Bc | Ac ] = [ Bc1 Bc2 | Ac]` 
in a staircase form

                        m1  m-m1  νr[1] νr[2] . . .  νr[p-2] 
                      [ B11 B12 | A11   A12   . . .  A1,p-2  A1,p-1  A1p ]  νr[1]
                      [  0  B22 | A21   A22   . . .  A2,p-2  A2,p-1  A2p ]  νr[2]
                      [  0   0  | A31   A32   . . .  A3,p-2  A3,p-1  A3p ]  νr[3]
                      [  0   0  |  0    A42   . . .  A4,p-2  A4,p-1  A4p ]  νr[4]
    [ Bc1 Bc2 | Ac] = [  0   0  |  .     .    . . .    .       .      .  ]   .
                      [  0   0  |  .     .      . .    .       .      .  ]   .
                      [  0   0  |  .     .        .    .       .      .  ]   .
                      [  0   0  |  0     0    . . .  Ap,p-2  Ap,p-1  App ]  νr[p]

where the blocks  `B11`, `B22`, `A31`, ..., `Ap,p-2`  have full row ranks.
The `p`-dimensional vector `νr`, with `p = 2nr` even,  contains the row dimensions of the 
blocks.  The difference `νr[2i-1]+νr[2i]-νr[2i+1]-νr[2i+2]` for `i = 1, 2, ..., nr` is the 
number of elementary Kronecker blocks of size `(i-1) x i`.

The `nuc x nuc` matrix `Auc` contains the (finite) eigenvalues of `M - λN` (also called the uncontrollable eigenvalues of `A`).  

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of `A`, the absolute tolerance for the nonzero elements of `B`,  and the relative tolerance 
for the nonzero elements of `A` and `B`.  
The reduction is performed using rank decisions based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.

The performed orthogonal or unitary transformations are accumulated in the matrix `Q` if `withQ = true`. 
Otherwise, `Q` is set to `nothing`.  

Method: The implemented algorithm [1] represents a specialization of the controllability staircase algorithm of [2] 
to the special structure of the input matrix `B = [B1,B2]`.

References:

[1] Varga, A. Reliable algorithms for computing minimal dynamic covers. Proc. CDC'2003, Hawaii, 2003.

[2] Varga, A. Numerically stable algorithm for standard controllability form determination.
    Electronics Letters, vol. 17, pp. 74-75, 1981.
"""
function sklf_right2!(A::AbstractMatrix{T}, B::AbstractMatrix{T}, m1::Int, C::Union{AbstractMatrix{T},Missing}; fast::Bool = true, 
                     atol1::Real = zero(real(T)), atol2::Real = zero(real(T)), 
                     rtol::Real = ((size(A,1)+1)*eps(real(float(one(T)))))*iszero(max(atol1,atol2)), 
                     withQ::Bool = true) where T <: BlasFloat
   n = LinearAlgebra.checksquare(A)
   n1, m = size(B)
   n == n1 || throw(DimensionMismatch("A and B must have the same number of rows"))
   (!ismissing(C) && n != size(C,2)) && throw(DimensionMismatch("A and C must have the same number of columns"))
   (m1 <= m && m1 >= 0) || throw(DimensionMismatch("B1 must have at most $m columns"))

   withQ ? Q = Matrix{T}(I,n,n) : Q = nothing

   # fast returns for null dimensions
   if m == 0 && n == 0
      νr = Int[]
      return Q, νr, 0, 0
   elseif n == 0
      νr = [0, 0]
      return Q, νr, 0, 0
   elseif m == 0
      νr = Int[]
      return Q, νr, 0, n
   end

   νr = Vector{Int}(undef,2*n)
   
   tolA = max(atol1, rtol*opnorm(A,1))
   tolB = max(atol2, rtol*opnorm(B,1))
      
   i = 0
   nc = 0
   nu = n
   m2 = m-m1
   tol = tolB 
   coff = 0
   while m > 0 && nc < n
      # Step 3: Particular case of the standard algorithm PREDUCE
      ρ1 = _sreduceBA2!(nu, m1, A, B, C, Q, tol, coff = coff, fast = fast, withQ = withQ)
      nu -= ρ1
      ρ2 = _sreduceBA2!(nu, m2, A, view(B,1:n,m1+1:m), C, Q, tol, coff = coff, fast = fast, withQ = withQ)
      ρ = ρ1+ρ2
      ρ == 0 && break
      i += 2
      νr[i-1] = ρ1
      νr[i] = ρ2
      nc += ρ
      nu -= ρ2
      m = ρ
      B = view(A,1:n,coff+1:coff+m)
      coff += ρ
      m1 = ρ1
      m2 = ρ2
      tol = tolA
   end
   return Q, νr[1:i], nc, nu
end

"""
    sklf_left!(A, C, B; fast = true, atol1 = 0, atol2 = 0, rtol, withQ = true) -> (Q, μl, no, nuo)

Reduce the partitioned full column rank matrix pencil 

      M - λN = | A-λI | n
               |  C   | p
                  n     
  
to an equivalent form `F - λL = diag(Q',I)*(M - λN)*Q` using orthogonal or unitary transformation matrix `Q`  
such that `M - λN` is transformed into a Kronecker-like form  exhibiting its 
finite  and left Kronecker structures (also known as the observability staircase form):
 
                   | Auo-λI  |  *    | nuo
      | At-λI |    |    0    | Ao-λI | no
      |-------| =  |---------|-------|
      |  Ct   |    |    0    |   Co  | p
                       nuo       no

`Ct = C*Q` and `At = Q'*A*Q` are returned in `C` and `A`, respectively, and `Q'*B` is returned in `B` (unless `B = missing`).                 

The subpencil `| Ao-λI; Co |` has full column rank `no`, is in a staircase form, 
and contains the left Kronecker indices of `M - λN`. 
   
The `nl`-dimensional vector `μl` contains the row and column dimensions of the blocks
of the staircase form such that `i`-th block has dimensions `μl[nl-i] x μl[nl-i+1]` (with μl[0] = p) and 
has full column rank. The difference `μl[nl-i]-μl[nl-i+1]` for `i = 1, 2, ..., nl` is the number of elementary Kronecker blocks
of size `i x (i-1)`.

The `nuo x nuo` matrix `Auo` contains the (finite) eigenvalues of `M - λN` (also called the unobservable eigenvalues of `A`).  

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of `A`, the absolute tolerance for the nonzero elements of `C`,  and the relative tolerance 
for the nonzero elements of `A` and `C`.  
The reduction is performed using rank decisions based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.

The performed orthogonal or unitary transformations are accumulated in the matrix `Q` if `withQ = true`. 
Otherwise, `Q` is set to `nothing`.   
"""
function sklf_left!(A::AbstractMatrix{T}, C::AbstractMatrix{T}, B::Union{AbstractVecOrMat{T},Missing}; fast::Bool = true, atol1::Real = zero(real(T)), atol2::Real = zero(real(T)), 
                   rtol::Real = ((size(A,1)+1)*eps(real(float(one(T)))))*iszero(max(atol1,atol2)), 
                   withQ::Bool = true) where T <: BlasFloat
   n = LinearAlgebra.checksquare(A)
   p, n1 = size(C)
   n == n1 || throw(DimensionMismatch("A and C must have the same number of columns"))
   (!ismissing(B) && n != size(B,1)) && throw(DimensionMismatch("A and B must have the same number of rows"))
   
   μl = Vector{Int}(undef,max(n,p))
   nu = 0
   tol1 = atol1
   withQ ? Q = Matrix{T}(I,n,n) : Q = nothing
 
   # fast returns for null dimensions
   if p == 0 && n == 0
      return Q, μl[1:0], 0, 0
   elseif n == 0
      μl[1] = 0
      return Q, μl[1:1], 0, 0
   elseif p == 0
      return Q, μl[1:0], 0, n
   end

   tolA = max(atol1, rtol*opnorm(A,1))
   tolC = max(atol2, rtol*opnorm(C,Inf))
      
   i = 0
   init = true
   rtrail = 0
   ctrail = 0
   no = 0
   nu = n
   while p > 0 && no < n
      init = (i == 0)
      init ? tol = tolC : tol = tolA
      # Step 3: Particular case of the standard algorithm PREDUCE
      ρ = _sreduceAC!(nu, p, A, C, B, Q, tol, fast = fast, init = init, rtrail = rtrail, ctrail = ctrail, withQ = withQ)
      ρ == 0 && break
      i += 1
      μl[i] = ρ
      ctrail += ρ
      init || (rtrail += p)
      no += ρ
      nu -= ρ
      p = ρ
      #return Q, reverse(μl[1:i]), no, nu
   end

   return Q, reverse(μl[1:i]), no, nu
end
"""
    gsklf(A, E, B, C, D; disc = false, jobopt = "none", offset = sqrt(ϵ), fast = true, atol1 = 0, atol2 = 0, rtol, 
         withQ = true, withZ = true) -> (F, G, Q, Z, dimsc, nmszer, nizer)

Reduce the structured matrix pencil `M - λN` 

               | A-λE | B | 
      M - λN = |------|---|
               |  C   | D |  

with the `n x n` pencil `A-λE` regular and `[E B]` of full row rank `n`, 
to an equivalent form `F - λG = diag(Q',I)*(M - λN)*Z` using orthogonal or unitary transformation 
matrices `Q` and `Z` such that the transformed matrices `F` and `G` are in the
following special, row partition preserving, Kronecker-like form

                | Ar-λEr    *     *   *   |
                |   0     Al-λEl  Bl  *   |
       F - λG = |   0       0     0   Bn  | ,    
                |-------------------------|
                |   0       Cl    Dl  *   |
         
where:
(1) `Ar-λEr` is a `mr x nr` full row rank pencil, which contains the right Kronecker structure  
     of `M - λN` and that part of its zeros which lie outside of the domain of interest `Cb` of 
     the complex plane defined by the keyword arguments `disc` and `jobopt`; 
(2)  `Al-lambda*El` is an `nl x nl` regular square pencil with `El` invertible and upper triangular;
(3)  `Bl` is an `nl x ml` matrix, where `ml` is the normal rank 
     of the transfer function matrix `C*inv(λE-A)*B+D`; 
(4)  `Bn` is an invertible matrix of order `nsinf`. 

The column dimensions `(nr, nl, ml, nsinf)` are returned in `dimsc`. 

The full column rank subpencil 

                | Al-λEl | Bl | 
    Ml - λNl := |--------|----|
                |   Cl   | Dl |  

contains the left Kronecker structure of `M - λN` and its zeros lying in the domain of interest `Cb`,
and has the property that `rank [Al-λEl Bl] = nl` provided `rank [A-λE B] = n` for all finite `λ ∈ Cb`
   (finite stabilizability with respect to `Cb`). 
The number of finite zeros lying on the boundary of `Cb`, if available, is returned in `nmszer`, 
otherwise `nmszer` is set to `missing`.  
If `disc = false`, the number of infinite zeros, if available, is returned in `nizer`, 
otherwise `nizer` is set to `missing`. If `disc = true`, `nizer` is set to 0. 
The boundary offset  `β` to be used to assess the zeros on the boundary of `Cb` is specified 
by the keyword parameter `offset`. Accordingly, if `disc = false`, then the boundary of `Cb` contains
the complex numbers with real parts within the interval `[-β,β]`, while if `disc = true`, 
then the boundary of `Cb` contains
the complex numbers with moduli within the interval `[1-β,1+β]`. The default value used for `β` is `sqrt(ϵ)`
 (the machine precision of the element type of `A`). 

The domain of interest `Cb` for the zeros of the pencil `Ml - λNl` is defined as follows:

if `jobopt = none`, then `Cb` is the empty set and no zeros of `M - λN` are included in `Ml - λNl`;
   
if  `jobopt = all`, then `Cb` is the extended complex plane (including infinity) and 
   all zeros of `M - λN` are included in `Ml - λNl`;

if `jobopt = finite`, then `Cb` is the complex plane (without infinity) and 
   all finite zeros of `M - λN` are included in `Ml - λNl`;

if `jobopt = infinite`, then `Cb` is the point at infinity and 
   all infinite zeros of `M - λN` are included in `Ml - λNl`;

if `jobopt = stable`, then, for `disc = false`, `Cb` is the set of complex numbers 
   with real parts less than `-β`, while for `disc = true`, `Cb` is the set of complex numbers 
   with moduli less than `1-β` and all finite zeros of `M - λN` in `Cb` are included in `Ml - λNl`;

if `jobopt = unstable`, then, for `disc = false`, `Cb` is the set of complex numbers 
   with real parts greater than `β` or infinite, while for `disc = true`, `Cb` is the set of complex numbers 
   with moduli greater than `1+β` or infinite and all finite and infinite zeros of `M - λN` in `Cb` are included in `Ml - λNl`;

if `jobopt = s-unstable`, then, for `disc = false`, `Cb` is the set of complex numbers 
   with real parts greater than `-β` or infinite, while for `disc = true`, `Cb` is the set of complex numbers 
   with moduli greater than `1-β` or infinite and all finite and infinite zeros of `M - λN` in `Cb` are included in `Ml - λNl`.

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of `M`, the absolute tolerance for the nonzero elements of `N`,  and the relative tolerance 
for the nonzero elements of `M` and `N`. 
The reduction is performed using rank decisions based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.

The performed left orthogonal or unitary transformations are accumulated in the `n x n` matrix `Q` if `withQ = true`. 
Otherwise, `Q` is set to `nothing`.   
The performed right orthogonal or unitary transformations are accumulated in the matrix `Z` if `withZ = true`. 
Otherwise, `Z` is set to `nothing`.  

Method: The reduction algorithm of [1] has been adapted to deal with
several zero selection options. 

References

[1] Oara, C. 
    Constructive solutions to spectral and inner–outer factorizations 
    with respect to the disk. Automatica, 41:1855–1866, 2005.
"""
function gsklf(A::AbstractMatrix, E::Union{AbstractMatrix,UniformScaling{Bool}}, 
               B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix; jobopt = "none", 
               disc::Bool = false, offset::Real = sqrt(eps(float(real(eltype(A))))), 
               atol1::Real = 0, atol2::Real = 0,  
               rtol::Real = ((min(size(A)...)+1)*eps(real(float(one(eltype(A))))))*iszero(min(atol1,atol2)),
               fast::Bool = true, withQ = true, withZ = true)

   n = LinearAlgebra.checksquare(A) 
   eident = (E == I);
   eident || LinearAlgebra.checksquare(E) == n || throw(DimensionMismatch("E must be a $n x $n matrix"))

   p, m = size(D)
   (n,m) == size(B) || throw(DimensionMismatch("A, B and D must have compatible dimensions"))
   (p,n) == size(C) || throw(DimensionMismatch("A, C and D must have compatible dimensions"))
   T = promote_type(eltype(A), eltype(B), eltype(C), eltype(D))
   eident || (T = promote_type(T,eltype(E)))
   T <: BlasFloat || (T = promote_type(Float64,T))
   A1 = copy_oftype(A,T)   
   eident || (E1 = copy_oftype(E,T))
   B1 = copy_oftype(B,T)
   C1 = copy_oftype(C,T)
   D1 = copy_oftype(D,T)

   
   opts = ("none","unstable","all","infinite","finite","stable","s-unstable")

   job = -1
   for i = 1:length(opts)
       isequal(jobopt,opts[i]) && (job = i-1; break)
   end
   job < 0 && error("No such zero selection option")
   
   
   #  Step 0: Compute orthogonal Q and U0 such that 
   #                                       
   #                 ( A-s*E | B )        ( A11-s*E11  A12-s*E12  A13-s*E13 )
   #    diag(Q',I) * (-------|---) * U0 = (    0          0         B2      )
   #                 (   C   | D )        (    C1         C2        D1      )
   #      
   #           with E11(n1,n1) and B2(n-n1,n-n1) invertible.

   nm = n+m
   np = n+p
   
   withQ ? Q = Matrix{T}(I,n,n) : Q = nothing
   withZ ? Z = Matrix{T}(I,nm,nm) : Z = nothing
   i1 = 1:n

   if eident
      At = [ A1 B1; C1 D1]; Et = [ I zeros(T,n,m); zeros(T,m,nm)] 
      AB = view(At,i1,:); CD = view(At,n+1:np,:); EF = view(Et,i1,:)
      n1 = n; nsinf = 0;   
   else
      n1, rA22 = _svdlikeAE!(A1, E1, Q, view(Z,i1,i1), B1, C1; svdA = false, fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol, withQ = withQ, withZ = withZ) 
      nsinf = n-n1; 
      if nsinf > 0
         i1 = 1:n1; i2 = n1+1:n
         rank(view(B1,i2,:), atol = atol1, rtol = rtol*opnorm(B1,1)) == nsinf || 
              error("The pair (A-λE,B) is not infinite controllable")
         m1 = m+nsinf
         At = [ A1 B1; C1 D1]; Et = [ E1 zeros(T,n,m); zeros(T,p,nm)] 
         AB = view(At,1:n,:); CD = view(At,n+1:np,:); EF = view(Et,1:n,:)
         # compress columns using an RQ-decomposition based reduction
         T <: Complex ? tran = 'C' : tran = 'T'
         Ak = view(At,i2,:)
         tau = similar(At,nsinf)
         LinearAlgebra.LAPACK.gerqf!(Ak,tau)
         LinearAlgebra.LAPACK.ormrq!('R',tran,Ak,tau,view(At,i1,:))
         LinearAlgebra.LAPACK.ormrq!('R',tran,Ak,tau,view(Et,i1,:))
         LinearAlgebra.LAPACK.ormrq!('R',tran,Ak,tau,CD)
         withZ && LinearAlgebra.LAPACK.ormrq!('R',tran,Ak,tau,Z) 
         At[i2,1:nm-nsinf] = zeros(T,nsinf,nm-nsinf) 
         triu!(view(At,i2,nm-nsinf+1:nm))
      else
         withQ ? Q = Matrix{T}(I,n,n) : Q = nothing
         withZ ? Z = Matrix{T}(I,nm,nm) : Z = nothing
         At = copy_oftype([A B; C D],T); Et = [ E zeros(T,n,m); zeros(T,m,nm)] 
         AB = view(At,i1,:); CD = view(At,n+1:n+p,:); EF = view(Et,i1,:)
         n1 = n; nsinf = 0;   
      end
   end

   #  Step 1: Compress [C1 C2] to [C1 C2]*U1 = [0 D2] with D2(p,rcd) monic.
   #
   #          Compute  [ A11-s*E11  A12-s*E12 ]*U1 = [ A1-s*E1   B1-s*F1 ]

   n1m = n1+m; j1 = 1:n1m;
   tolCD = max(atol1, rtol*opnorm(CD,1))
   rcd = _sreduceC!(view(At,i1,j1),view(Et,i1,j1),view(CD,:,j1),view(Z,:,j1), tolCD; 
                    fast = fast, withZ = withZ) 
   
   # Step 2: Compute the  Kronecker-like staircase form of A1-s*E1 such that
   #
   #                               (A11-s*E11    X         X      |    X   )
   # Q2'*(A1-s*E1 | B1-s*F1 )*Z2 = (    0     A22-s*E22    X      |    X   )
   #                               (    0        0      A33-s*E33 | B3-s*F3)

   n3 = n1m-rcd;
   j1 = 1:n3; j2 = n3+1:n1m; jt = n3+1:nm

   if n3 > 0
      if job == 0 || job == 3 
         # compute the KLF with (Arf,Ainf,Al) diagonal blocks 
         AB[i1,j1], EF[i1,j1], q, z, nr, mr, νi, νl, μl = 
               klf_leftinf(view(AB,i1,j1),view(EF,i1,j1); fast = fast,  
               atol1 = atol1, atol2 = atol2, rtol = rtol, withQ = true, withZ = true) 
         nf = 0                              
         #mc = nr+pr; nc = nr+mr;
         mc = nr+mr; nc = nr;
      else
         if job == 1
            # compute the KLF with (Ar,Af,Ainf,Al) diagonal blocks if disc = true, or 
            # compute the KLF with (Ar,Ainf,Af,Al) diagonal blocks if disc = false
            finite_infinite = disc
         elseif job == 2 || job == 4 || job == 5
            # compute the KLF with (Ar,Ainf,Af,Al) diagonal blocks
            finite_infinite = false
         else # job = 6
            # compute the KLF with (Ar,Af,Ainf,Al) diagonal blocks
            finite_infinite = true
         end
         AB[i1,j1], EF[i1,j1], q, z, νr, μr, νi, nf, νl, μl = 
               klf(view(AB,i1,j1),view(EF,i1,j1); fast = fast, finite_infinite = finite_infinite, 
               atol1 = atol1, atol2 = atol2, rtol = rtol, withQ = true, withZ = true) 
         nc = sum(νr); mc = sum(μr);
      end
      AB[i1,jt] = q'*view(AB,i1,jt);
      EF[i1,jt] = q'*view(EF,i1,jt);
      withQ && (Q[:,i1] = view(Q,:,i1)*q)
      withZ && (Z[:,j1] = view(Z,:,j1)*z)
      ni = sum(νi); 
      nl = sum(νl); 
   else
      nc = 0; mc = 0; ni = 0; nf = 0; nl = n1;  
   end
   
   nmszer = missing; nizer = missing;
   if job == 0
      # no zeros included
      nus = 0; ns = ni; nizer = ni;
   elseif job == 1 
      # include all unstable zeros excluding those on the boundary of the
      # stability region
      if nf == 0
         if disc
            nus = ni; ns = 0; nmszer = 0; nizer = 0;
         else
            nus = 0; ns = ni; nmszer = 0; nizer = ni;
         end
      else    
         if disc
            # the finite zeros are in the leading block
            izf = nc+1:nc+nf; jzf = mc+1:mc+nf;
         else
            # the finite zeros are in the trailing block
            izf = nc+ni+1:nc+ni+nf; jzf = mc+ni+1:mc+ni+nf;
         end
         S = schur(view(AB,izf,jzf),view(EF,izf,jzf))
         if disc
            select = abs.(S.α) .<= (1+offset)*abs.(S.β);
            nizer = 0; 
            nmszer = count(t -> t == true, abs.(S.α[select]) .>= (1-offset)*abs.(S.β[select]))
         else
            select = (real.(S.α ./ S.β) .<= offset);
            nizer = ni;
            nmszer = count(t -> t == true, real(S.α[select] ./ S.β[select]) .>= -offset)
         end
         ns = count(t -> t == true, select)  
          AB[izf,jzf], EF[izf,jzf], q, z = ordschur!(S, select)
      
         if disc
            # apply q to the trailing columns including infinite zeros
            j4 = mc+nf+1:nm;  
            izf1 = 1:nc;
            nus = nf-ns+ni;   # include infinite zeros among unstable zeros
         else
            # apply q to the trailing columns without infinite zeros
            j4 = mc+ni+nf+1:nm;  
            izf1 = 1:(nc+ni);
            nus = nf-ns;      # include only the finite unstable zeros
            ns = ns+ni;       # include infinite zeros among stable zeros
         end
         AB[izf,j4] = q'*view(AB,izf,j4); EF[izf,j4] = q'*view(EF,izf,j4)
         AB[izf1,jzf] = view(AB,izf1,jzf)*z; EF[izf1,jzf] = view(EF,izf1,jzf)*z;
         withQ && ( Q[:,izf] = view(Q,:,izf)*q )
         withZ && ( Z[:,jzf] = view(Z,:,jzf)*z )
      end
   elseif job == 2   
      # include all zeros 
      nus = nf+ni; ns = 0;
   elseif job == 3   
      # include all infinite zeros 
      nus = ni; ns = 0; nizer = ni; 
   elseif job == 4   
      # include all finite zeros 
      nus = nf; ns = ni;
   elseif job == 5 
      # include all stable zeros 
      if nf == 0
         nus = 0; ns = ni;
         if disc
            nmszer = 0; nizer = 0;
         else
            nmszer = 0; nizer = ni;
         end
      else    
         # separate stable/unstable zeros
         # the finite zeros are in the trailing block
         izf = nc+ni+1:nc+ni+nf; jzf = mc+ni+1:mc+ni+nf;
         S = schur(view(AB,izf,jzf),view(EF,izf,jzf))
         if disc
            select = abs.(S.α) .>= (1-offset)*abs.(S.β);
            nizer = 0; 
            nmszer = count(t -> t == true, abs.(S.α[select]) .<= (1+offset)*abs.(S.β[select]))
         else
            select = (real.(S.α ./ S.β) .>= -offset);
            nizer = ni;
            nmszer = count(t -> t == true, real(S.α[select] ./ S.β[select]) .<= offset)
         end
         nus = count(t -> t == false, select) 
         AB[izf,jzf], EF[izf,jzf], q, z = ordschur!(S, select)
         ns = nf-nus+ni; 
         izf1 = 1:(nc+ni);
         j4 = mc+ni+nf+1:nm; 
         AB[izf,j4] = q'*view(AB,izf,j4); EF[izf,j4] = q'*view(EF,izf,j4)
         AB[izf1,jzf] = view(AB,izf1,jzf)*z; EF[izf1,jzf] = view(EF,izf1,jzf)*z;
         withQ && ( Q[:,izf] = view(Q,:,izf)*q )
         withZ && ( Z[:,jzf] = view(Z,:,jzf)*z )
      end
   elseif job == 6 
      # include all strictly unstable zeros and infinite zeros
      if nf == 0
         nus = ni; ns = 0;
         if disc
            nmszer = 0; nizer = 0;
         else
            nmszer = 0; nizer = ni;
         end
      else    
         # the finite zeros are in the leading block
         izf = nc+1:nc+nf; jzf = mc+1:mc+nf;
         S = schur(view(AB,izf,jzf),view(EF,izf,jzf))
         if disc
            select = abs.(S.α) .<= (1-offset)*abs.(S.β);
            nizer = 0;
            nmszer = count(t -> t == true, abs.(S.α[.!select]) .<= (1+offset)*abs.(S.β[.!select]))
         else
            select = (real.(S.α ./ S.β) .<= -offset);
            nizer = ni;
            nmszer = count(t -> t == true, real(S.α[.!select] ./ S.β[.!select]) .<= offset)
         end
         ns = count(t -> t == true, select) 
         AB[izf,jzf], EF[izf,jzf], q, z = ordschur!(S, select)
      
         # apply q to the trailing columns including infinite zeros
         j4 = mc+nf+1:nm;  
         izf1 = 1:nc;
         nus = nf-ns+ni;   # include infinite zeros among unstable zeros
         AB[izf,j4] = q'*view(AB,izf,j4); EF[izf,j4] = q'*view(EF,izf,j4)
         AB[izf1,jzf] = view(AB,izf1,jzf)*z; EF[izf1,jzf] = view(EF,izf1,jzf)*z;
         withQ && ( Q[:,izf] = view(Q,:,izf)*q )
         withZ && ( Z[:,jzf] = view(Z,:,jzf)*z )
      end
   end
   
   # Step 3: Compress [E33 F3] to [E33 F3]*U2 = [E2 0] with E2(nl+nus,nl+nus) invertible.
   #         Compute  [A33 B3]*U2 = [A2 B2].
   #         Compute  [0 D1]*U2 = [C2 D2]  with D2 monic.
   mrg = mc+ns; 
   ir = nc+ns+1:n1; jr = mrg+1:n1m;
   nbl = nl+nus;
   mbl = n1m-mrg-nbl; # normal rank
   if nbl > 0
      T <: Complex ? tran = 'C' : tran = 'T'
      ir1 = 1:(nc+ns); 
      E2 = view(Et,ir,jr)
      _, τ = LinearAlgebra.LAPACK.geqrf!(E2)
      jt = mrg+1:nm
      j4 = n1m+1:nm
      # A <- Q1'*A
      LinearAlgebra.LAPACK.ormqr!('L',tran,E2,τ,view(At,ir,jt))
      LinearAlgebra.LAPACK.ormqr!('L',tran,E2,τ,view(Et,ir,j4))
      # Q <- Q*Q1
      withQ && LinearAlgebra.LAPACK.ormqr!('R','N',E2,τ,view(Q,:,ir))
      j1 = mrg+1:mrg+nbl
      # compute in-place the complete orthogonal decomposition E2*Z1 = [E11 0; 0 0] with E11 nonsingular and UT
      _, tau = LinearAlgebra.LAPACK.tzrzf!(E2)
      # A <- A*Z1
      LinearAlgebra.LAPACK.ormrz!('R',tran,E2,tau,view(At,1:n1,jr))
      LinearAlgebra.LAPACK.ormrz!('R',tran,E2,tau,view(Et,ir1,jr))
      LinearAlgebra.LAPACK.ormrz!('R',tran,E2,tau,view(CD,:,jr))
      withZ && LinearAlgebra.LAPACK.ormrz!('R',tran,E2,tau,view(Z,:,jr))
      EF[ir,jr] = [ triu(EF[ir,j1]) zeros(T,nbl,mbl) ]
   end 
    
   
   # if withQ && withZ
   #    println(" res1 = $(norm(Q'*[A B]*Z-AB))") 
   #    println(" res2 = $(norm(Q'*[E zeros(n,m)]*Z-EF))")
   #    println(" res3 = $(norm([C D]*Z-CD))") 
   # end
   
   dimsc = (mrg, nbl, mbl, nsinf)  # column dimensions of blocks
   return At, Et, Q, Z, dimsc, nmszer, nizer   

  
   # end GSKLF
end

