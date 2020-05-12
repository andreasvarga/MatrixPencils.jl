"""
    sreduceBF(A, E, B, C, D; fast = true, atol = 0, rtol, withQ = true, withZ = true) -> F, G, Q, Z, n, m, p

Reduce the partitioned linear pencil `M - λN` 

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

The performed orthogonal or unitary transformations are accumulated in `Q`, if `withQ = true`, and 
`Z`, if `withZ = true`. 

If `fast = true`, `E1` is determined upper triangular using a rank revealing QR-decomposition with column pivoting of `E` 
and `n` is evaluated as the number of nonzero diagonal elements of the R factor, whose magnitudes are greater than 
`tol = max(atol,abs(R[1,1])*rtol)`. 
If `fast = false`,  `E1` is determined diagonal using a rank revealing SVD-decomposition of `E` and 
`n` is evaluated as the number of singular values greater than `tol = max(atol,smax*rtol)`, where `smax` 
is the largest singular value. 
The rank decision based on the SVD-decomposition is generally more reliable, but the involved computational effort is higher.
"""
function sreduceBF(A::Union{AbstractMatrix,Missing}, E::Union{AbstractMatrix,UniformScaling{Bool},Missing}, 
                   B::Union{AbstractMatrix,Missing}, C::Union{AbstractMatrix,Missing}, D::Union{AbstractMatrix,Missing}; 
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
      p, m = size(D)
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
      ndx == size(B,1) || throw(DimensionMismatch("A and B must have the same number of rows"))
      T = promote_type(eltype(A), eltype(E), eltype(B), eltype(E), eltype(A), eltype(E) )
      T <: BlasFloat || (T = promote_type(Float64,T))
      ny = 0
      C = zeros(T,ny,nx)
      nu = size(B,2)
      D = zeros(T,ny,nu)
   else
      ndx, nx = size(A)
      T = promote_type(eltype(A), eltype(B), eltype(C), eltype(D))
      eident || (ndx,nx) == size(E) || throw(DimensionMismatch("A and E must have the same dimensions"))
      ny, nu = size(D)
      (ndx,nu) == size(B) || throw(DimensionMismatch("A, B and D must have compatible dimensions"))
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
    sklf(A, E, B, C, D; fast = true, finite_infinite = false, atol1 = 0, atol2 = 0, rtol, withQ = true, withZ = true) -> (F, G, Q, Z, νr, μr, νi, nf, νl, μl)

Reduce the structured linear pencil `M - λN` 

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
              B::Union{AbstractMatrix,Missing}, C::Union{AbstractMatrix,Missing}, D::Union{AbstractMatrix,Missing}; 
              atol1::Real = ismissing(A) ? zero(real(eltype(D))) : zero(real(eltype(A))), 
              atol2::Real = ismissing(A) ? zero(real(eltype(D))) : zero(real(eltype(A))), 
              rtol::Real = (ismissing(A) ? 1 : min(size(A)...))*eps(real(float(ismissing(A) ? one(real(eltype(D))) : one(real(eltype(A))))))*iszero(min(atol1,atol2)), 
              fast::Bool = true, finite_infinite::Bool = false, withQ::Bool = true, withZ::Bool = true) 
   
   # Step 0: Reduce to the standard form
   M, N, Q, Z, n, m, p = sreduceBF(A, E, B, C, D, atol = atol2, rtol = rtol, fast = fast, withQ = withQ, withZ = withZ)
   mM, nM = size(M)
   T = eltype(M)

   if finite_infinite
      
      # Reduce M-λN to a KLF exhibiting the right and finite structures
      #                  [ Mr - λ Nr  |   *        |     *        ]
      #      M1 - λ N1 = [    0       | Mf -  λ Nf |     *        ]
      #                  [    0       |    0       | Mli -  λ Nli ]
      
      νr, μr, nf, ν, μ, tol1 = klf_right!(n, m, p, M, N, Q, Z, atol = atol1, rtol = rtol,  
                                          withQ = withQ, withZ = withZ, fast = fast)
      (mM == 0 || nM == 0) && (return  M, N, Q, Z, νr, μr, ν[1:0], nf, ν, μ)
    
      # Reduce Mli-λNli to a KLF exhibiting the infinite and left structures and update M1 - λ N1 to
      #                  [ Mr - λ Nr  |   *        |     *      |    *       ]
      #     M2 -  λ N2 = [    0       | Mf -  λ Nf |     *      |    *       ]
      #                  [    0       |    0       | Mi -  λ Ni |    *       ]
      #                  [    0       |    0       |    0       | Ml -  λ Nl ]

      mr = sum(νr)+nf
      nr = sum(μr)+nf
      jM2 = nr+1:nr+sum(μ)
      M2 = view(M,:,jM2)
      N2 = view(N,:,jM2)
      withZ ? (Z2 = view(Z,:,jM2)) : (Z2 = nothing)
      νi, νl, μl = klf_left_refine!(ν, μ, M2, N2, Q, Z2, tol1, roff = mr,    
                                    withQ = withQ, withZ = withZ, fast = fast)
   else

      # Reduce M-λN to a KLF exhibiting the left and finite structures
      #                  [ Mri - λ Nri  |   *        |     *      ]
      #      M1 - λ N1 = [     0        | Mf -  λ Nf |     *      ]
      #                  [     0        |    0       | Ml -  λ Nl ]

      ν, μ, nf, νl, μl, tol1 = klf_left!(n, m, p, M, N, Q, Z, atol = atol1, rtol = rtol,  
                                         withQ = withQ, withZ = withZ, fast = fast)
      (mM == 0 || nM == 0) && (return  M, N, Q, Z, ν, μ, ν[1:0], nf, νl, μl)
   
      # Reduce Mri-λNri to a KLF exhibiting the right and infinite structures and update M1 - λ N1 to
      #                  [ Mr - λ Nr  |   *        |     *      |    *       ]
      #     M2 -  λ N2 = [    0       | Mi -  λ Ni |     *      |    *       ]
      #                  [    0       |    0       | Mf -  λ Nf |    *       ]
      #                  [    0       |    0       |    0       | Ml -  λ Nl ]

      iM11 = 1:sum(ν)
      M1 = view(M,iM11,:)
      N1 = view(N,iM11,:)
      νr, μr, νi = klf_right_refine!(ν, μ, M1, N1, Q, Z, tol1, ctrail = nM-sum(μ),   
                                     withQ = withQ, withZ = withZ, fast = fast)
   end
   return  M, N, Q, Z, νr, μr, νi, nf, νl, μl                                             
end
"""
    sklf_right(A, E, B, C, D; fast = true, atol1 = 0, atol2 = 0, rtol, withQ = true, withZ = true) -> (F, G, Q, Z, νr, μr, nf, ν, μ)

Reduce the structured linear pencil `M - λN` 

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
                |    O       |     0      | Mli-λNli |            

The full row rank pencil `Mr-λNr`, in a staircase form, contains the right Kronecker indices of `M-λN` and has the form

         Mr-λNr  = | Br | Ar-λEr |,
                   
where `Er` is upper triangular and nonsingular. 
The `nr`-dimensional vectors `νr` and `μr` contain the row and, respectively, column dimensions of the blocks
of the staircase form  `Mr-λNr` such that `i`-th block has dimensions `νr[i] x μr[i]` and 
has full row rank. The difference `μr[i]-νr[i]` for `i = 1, 2, ..., nr` is the number of elementary Kronecker blocks
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
function sklf_right(A::Union{AbstractMatrix,Missing}, E::Union{AbstractMatrix,UniformScaling{Bool},Missing}, 
   B::Union{AbstractMatrix,Missing}, C::Union{AbstractMatrix,Missing}, D::Union{AbstractMatrix,Missing}; 
   atol1::Real = ismissing(A) ? zero(real(eltype(D))) : zero(real(eltype(A))), 
   atol2::Real = ismissing(A) ? zero(real(eltype(D))) : zero(real(eltype(A))), 
   rtol::Real = (ismissing(A) ? 1 : min(size(A)...))*eps(real(float(ismissing(A) ? one(real(eltype(D))) : one(real(eltype(A))))))*iszero(min(atol1,atol2)), 
   fast::Bool = true, withQ::Bool = true, withZ::Bool = true) 
   
   # Step 0: Reduce to the standard form
   M, N, Q, Z, n, m, p = sreduceBF(A, E, B, C, D, atol = atol2, rtol = rtol, fast = fast, withQ = withQ, withZ = withZ)

   νr, μr, nf, ν, μ, tol1 = klf_right!(n, m, p, M, N, Q, Z, atol = atol1, rtol = rtol,  
                                       withQ = withQ, withZ = withZ, fast = fast)

   return M, N, Q, Z, νr, μr, nf, ν, μ

end
"""
    sklf_left(A, E, B, C, D; fast = true, atol1 = 0, atol2 = 0, rtol, withQ = true, withZ = true) -> (F, G, Q, Z, ν, μ, nf, νl, μl)

Reduce the structured linear pencil `M - λN` 

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
function sklf_left(A::Union{AbstractMatrix,Missing}, E::Union{AbstractMatrix,UniformScaling{Bool},Missing}, 
   B::Union{AbstractMatrix,Missing}, C::Union{AbstractMatrix,Missing}, D::Union{AbstractMatrix,Missing}; 
   atol1::Real = ismissing(A) ? zero(real(eltype(D))) : zero(real(eltype(A))), 
   atol2::Real = ismissing(A) ? zero(real(eltype(D))) : zero(real(eltype(A))), 
   rtol::Real = (ismissing(A) ? 1 : min(size(A)...))*eps(real(float(ismissing(A) ? one(real(eltype(D))) : one(real(eltype(A))))))*iszero(min(atol1,atol2)), 
   fast::Bool = true, withQ::Bool = true, withZ::Bool = true) 
   
   # Step 0: Reduce to the standard form
   M, N, Q, Z, n, m, p = sreduceBF(A, E, B, C, D, atol = atol2, rtol = rtol, fast = fast, withQ = withQ, withZ = withZ)

   ν, μ, nf, νl, μl, tol1 = klf_left!(n, m, p, M, N, Q, Z, atol = atol1, rtol = rtol,  
                                      withQ = withQ, withZ = withZ, fast = fast)

   return M, N, Q, Z, ν, μ, nf, νl, μl
end
"""
    sklf_right!(A, E, B, F, C, G, D, H; fast = true, atol1 = 0, atol2 = 0, atol3 = 0, rtol, withQ = true, withZ = true) -> (Q, Z, nc)

Reduce the partitioned full row rank linear pencil 

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
(unless `C`, `D', `G` and `H` are set to 'missing').


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
function sklf_right!(A::AbstractMatrix{T1}, E::AbstractMatrix{T1}, B::AbstractMatrix{T1}, F::AbstractMatrix{T1}, 
                     C::Union{AbstractMatrix{T1},T2}, G::Union{AbstractMatrix{T1},T2},
                     D::Union{AbstractMatrix{T1},T2}, H::Union{AbstractMatrix{T1},T2}; 
                     fast::Bool = true, atol1::Real = zero(real(eltype(A))), atol2::Real = zero(real(eltype(E))), atol3::Real = zero(real(eltype(B))), 
                     rtol::Real = ((size(A,1)+2)*eps(real(float(one(eltype(A))))))*iszero(max(atol1,atol2,atol3)), 
                     withQ::Bool = true, withZ::Bool = true) where {T1 <: BlasFloat, T2 <: Missing}
   n = LinearAlgebra.checksquare(A)
   n == LinearAlgebra.checksquare(E) || throw(DimensionMismatch("A and E must have the same dimensions"))          
   n1, m = size(B)
   n == n1 || throw(DimensionMismatch("A and B must have the same number of rows"))
   (n,m) == size(F) || throw(DimensionMismatch("B and F must have the same dimensions"))
   if !ismissing(C) 
      (p,n2) = size(C)
      n == n2 || throw(DimensionMismatch("A and C must have the same number of columns"))
      (p,n) == size(G) || throw(DimensionMismatch("C and G must have the same dimensions"))
      (p,m) == size(D) || throw(DimensionMismatch("D must have the same row dimension as C and same column dimension as B"))
      (p,m) == size(H) || throw(DimensionMismatch("D and H must have the same dimensions"))
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
    sklf_left!(A, E, C, B, F, G, D, H; fast = true, atol1 = 0, atol2 = 0, atol3 = 0, rtol, withQ = true, withZ = true) -> (Q, Z, no)

Reduce the partitioned full row rank linear pencil 

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
(unless `B`, `D', `F` and `H` are set to 'missing').

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
                     B::Union{AbstractMatrix{T1},T2}, F::Union{AbstractMatrix{T1},T2},
                     D::Union{AbstractMatrix{T1},T2}, H::Union{AbstractMatrix{T1},T2}; 
                     fast::Bool = true, atol1::Real = zero(real(eltype(A))), atol2::Real = zero(real(eltype(E))), atol3::Real = zero(real(eltype(B))), 
                     rtol::Real = ((size(A,1)+2)*eps(real(float(one(eltype(A))))))*iszero(max(atol1,atol2,atol3)), 
                     withQ::Bool = true, withZ::Bool = true) where {T1 <: BlasFloat, T2 <: Missing}
   n = LinearAlgebra.checksquare(A)
   n == LinearAlgebra.checksquare(E) || throw(DimensionMismatch("A and E must have the same dimensions"))          
   p, n1 = size(C)
   n == n1 || throw(DimensionMismatch("A and C must have the same number of columns"))
   (p,n) == size(G) || throw(DimensionMismatch("C and G must have the same dimensions"))
   if !ismissing(C) 
      n2, m = size(B)
      n == n2 || throw(DimensionMismatch("A and B must have the same number of rows"))
      (n,m) == size(F) || throw(DimensionMismatch("B and F must have the same dimensions"))
      (p,m) == size(D) || throw(DimensionMismatch("D must have the same row dimension as C and same column dimension as B"))
      (p,m) == size(H) || throw(DimensionMismatch("D and H must have the same dimensions"))
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

Reduce the partitioned full row rank linear pencil 

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

`Bt = Q'*B`, `At = Q'*A*Z`and `Et = Q'*E*Z` are returned in `B`, `A` and `E`, respectively, and `C*Z` is returned in `C` (unless `C = missing').                 

The subpencil `| Bc | Ac-λEc |` has full row rank `nc`, is in a staircase form, and contains the right Kronecker indices of `M - λN`. 
The `nr`-dimensional vector `νr` contains the row and column dimensions of the blocks
of the staircase form  `| Bc | Ac-λI |` such that `i`-th block has dimensions `νr[i] x νr[i-1]` (with `νr[0] = m`) and 
has full row rank. The difference `νr[i-1]-νr[i]` for `i = 1, 2, ..., nr` is the number of elementary Kronecker blocks
of size `(i-1) x i`.

The `nfuc x nfuc` subpencil `Afuc-λEfuc` contains the finite eigenvalues of `M - λN` (also called the uncontrollable finite eigenvalues of `A - λE`).  

The `niuc x niuc` subpencil `Aiuc-λEiuc` contains the infinite eigenvalues of `M - λN` (also called the uncontrollable infinite eigenvalues of `A - λE`).  

The keyword arguments `atol1`, `atol2`, , `atol3`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of `A`, the absolute tolerance for the nonzero elements of `E`, the absolute tolerance for the nonzero elements of `B`,  
and the relative tolerance for the nonzero elements of `A`, `E` and `B`.  
The reduction is performed using rank decisions based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.
The performed orthogonal or unitary left transformations are accumulated in the matrix `Q` if `withQ = true`. 
Otherwise, `Q` is set to `nothing`.   
The performed orthogonal or unitary right transformations are accumulated in the matrix `Z` if `withZ = true` or 
`C` is provided. Otherwise, `Z` is set to `nothing`.   
"""
function sklf_right!(A::AbstractMatrix{T1}, E::AbstractMatrix{T1}, B::AbstractMatrix{T1}, C::Union{AbstractMatrix{T1},Missing}; fast::Bool = true, 
   atol1::Real = zero(real(eltype(A))), atol2::Real = zero(real(eltype(E))), atol3::Real = zero(real(eltype(B))), 
                   rtol::Real = ((size(A,1)+1)*eps(real(float(one(eltype(A))))))*iszero(max(atol1,atol2,atol3)), 
                   withQ::Bool = true, withZ::Bool = true) where T1 <: BlasFloat
   n = LinearAlgebra.checksquare(A)
   n == LinearAlgebra.checksquare(E) || throw(DimensionMismatch("A and E must have the same dimensions"))          
   n1, m = size(B)
   n == n1 || throw(DimensionMismatch("A and B must have the same number of rows"))
   (!ismissing(C) && n !== size(C,2)) && throw(DimensionMismatch("A and C must have the same number of columns"))
   
   maxmn = max(m,n)
   νr = Vector{Int}(undef,maxmn)
   nfu = 0
   niu = 0
   tol1 = atol1
   withQ ? Q = Matrix{eltype(A)}(I,n,n) : Q = nothing
   withZ ? Z = Matrix{eltype(A)}(I,n,n) : Z = nothing
 
   # fast returns for null dimensions
   if m == 0 && n == 0 
      return Q, Z, νr, n, nfu, niu
   elseif n == 0 
      νr[1] = 0
      return Q, Z, νr[1:1], n, nfu, niu
   end

   tolA = max(atol1, rtol*opnorm(A,1))
   tolB = max(atol3, rtol*opnorm(B,1))

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

Reduce the partitioned full row rank linear pencil 

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

`Bt = Q'*B`, `At = Q'*A*Z`and `Et = Q'*E*Z` are returned in `B`, `A` and `E`, respectively, and `C*Z` is returned in `C` (unless `C = missing'). 
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

Note: This function, called with reversed input parameters `E` and `A` (i.e., instead `A` and `E`), performs the 
separation all infinite and nonzero finite eigenvalues of the pencil `M - λN`.
"""
function sklf_rightfin!(A::AbstractMatrix{T1}, E::AbstractMatrix{T1}, B::AbstractMatrix{T1}, C::Union{AbstractMatrix{T1},Missing}; 
                        fast::Bool = true, atol1::Real = zero(real(T1)), atol2::Real = zero(real(T1)), 
                        rtol::Real = ((size(A,1)+1)*eps(real(float(one(T1)))))*iszero(max(atol1,atol2)), 
                        withQ::Bool = true, withZ::Bool = true) where T1 <: BlasFloat
   n = LinearAlgebra.checksquare(A)
   n == LinearAlgebra.checksquare(E) || throw(DimensionMismatch("A and E must have the same dimensions"))          
   n1, m = size(B)
   n == n1 || throw(DimensionMismatch("A and B must have the same number of rows"))
   (!ismissing(C) && n !== size(C,2)) && throw(DimensionMismatch("A and C must have the same number of columns"))
   
   νr = Vector{Int}(undef,max(n,1))
   withQ ? Q = Matrix{eltype(A)}(I,n,n) : Q = nothing
   withZ ? Z = Matrix{eltype(A)}(I,n,n) : Z = nothing

   # fast returns for null dimensions
   m == 0 && n == 0 && (return Q, Z, νr[1:0], 0, 0)
   n == 0 && (return Q, Z, [0], 0, 0)

   # Reduce E to upper triangular form if necessary
   istriu(E) || _qrE!(A, E, Q, B; withQ = withQ) 
   m == 0 && (return Q, Z, νr[1:0], 0, n)
  
   tolA = max(atol1, rtol*opnorm(A,1))
   tolB = max(atol2, rtol*opnorm(B,1))

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
    sklf_left!(A, E, C, B; fast = true, atol1 = 0, atol2 = 0, atol3 = 0, rtol, withQ = true, withZ = true) -> (Q, Z, μl, no, nfuo, niuo)

Reduce the partitioned full column rank linear pencil 

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
and `Q'*B` is returned in `B` (unless `B = missing').                 

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
function sklf_left!(A::AbstractMatrix{T1}, E::AbstractMatrix{T1}, C::AbstractMatrix{T1}, B::Union{AbstractMatrix{T1},Missing}; fast::Bool = true, 
                   atol1::Real = zero(real(eltype(A))), atol2::Real = zero(real(eltype(E))), atol3::Real = zero(real(eltype(C))), 
                   rtol::Real = ((size(A,1)+1)*eps(real(float(one(eltype(A))))))*iszero(max(atol1,atol2,atol3)), 
                   withQ::Bool = true, withZ::Bool = true) where T1 <: BlasFloat
   n = LinearAlgebra.checksquare(A)
   n == LinearAlgebra.checksquare(E) || throw(DimensionMismatch("A and E must have the same dimensions"))          
   p, n1 = size(C)
   n == n1 || throw(DimensionMismatch("A and C must have the same number of columns"))
   (!ismissing(B) && n !== size(B,1)) && throw(DimensionMismatch("A and B must have the same number of rows"))
   
   maxpn = max(p,n)
   μl = Vector{Int}(undef,maxpn)
   nfu = 0
   niu = 0
   tol1 = atol1
   withQ ? Q = Matrix{eltype(A)}(I,n,n) : Q = nothing
   withZ ? Z = Matrix{eltype(A)}(I,n,n) : Z = nothing
 
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

Reduce the partitioned full column rank linear pencil 

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
and `Q'*B` is returned in `B` (unless `B = missing').   
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

Note: This function, called with reversed input parameters `E` and `A` (i.e., instead `A` and `E`), performs the 
separation all infinite and nonzero finite eigenvalues of the pencil `M - λN`.
"""
function sklf_leftfin!(A::AbstractMatrix{T1}, E::AbstractMatrix{T1}, C::AbstractMatrix{T1}, B::Union{AbstractMatrix{T1},Missing}; 
                       fast::Bool = true, atol1::Real = zero(real(T1)), atol2::Real = zero(real(T1)), 
                       rtol::Real = ((size(A,1)+1)*eps(real(float(one(T1)))))*iszero(max(atol1,atol2)), 
                       withQ::Bool = true, withZ::Bool = true) where T1 <: BlasFloat

   n = LinearAlgebra.checksquare(A)
   n == LinearAlgebra.checksquare(E) || throw(DimensionMismatch("A and E must have the same dimensions"))          
   p, n1 = size(C)
   n == n1 || throw(DimensionMismatch("A and C must have the same number of columns"))
   (!ismissing(B) && n !== size(B,1)) && throw(DimensionMismatch("A and B must have the same number of rows"))

   μl = Vector{Int}(undef,max(n,1))
   withQ ? Q = Matrix{eltype(A)}(I,n,n) : Q = nothing
   withZ ? Z = Matrix{eltype(A)}(I,n,n) : Z = nothing

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

Reduce the partitioned full row rank linear pencil 

      M - λN = | B | A-λI | n
                 m    n     
  
to an equivalent form `F - λG = Q'*(M - λN)*diag(I,Q)` using orthogonal or unitary transformation matrix `Q`  
such that `M - λN` is transformed into a Kronecker-like form `| Bt | At-λI |` exhibiting its 
right and finite Kronecker structures (also known as the controllability staircase form):
 
                        |  Bc | Ac-λI     *    | nc
      | Bt | At-λI | =  |-----|----------------|
                        |  0  |  0     Auc-λI  | nuc
                           m     nc      nuc

`Bt = Q'*B` and `At = Q'*A*Q` are returned in `B` and `A`, respectively, and `C*Q` is returned in `C` (unless `C = missing').                 

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
function sklf_right!(A::AbstractMatrix{T1}, B::AbstractMatrix{T1}, C::Union{AbstractMatrix{T1},Missing}; fast::Bool = true, 
                     atol1::Real = zero(real(eltype(A))), atol2::Real = zero(real(eltype(A))), 
                     rtol::Real = ((size(A,1)+1)*eps(real(float(one(eltype(A))))))*iszero(max(atol1,atol2)), 
                     withQ::Bool = true) where T1 <: BlasFloat
   n = LinearAlgebra.checksquare(A)
   n1, m = size(B)
   n == n1 || throw(DimensionMismatch("A and B must have the same number of rows"))
   (!ismissing(C) && n !== size(C,2)) && throw(DimensionMismatch("A and C must have the same number of columns"))
   
   νr = Vector{Int}(undef,max(n,1))
   nu = 0
   tol1 = atol1
   withQ ? Q = Matrix{eltype(A)}(I,n,n) : Q = nothing
 
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
   tolB = max(atol2, rtol*opnorm(B,1))
      
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
    sklf_left!(A, C, B; fast = true, atol1 = 0, atol2 = 0, rtol, withQ = true) -> (Q, μl, no, nuo)

Reduce the partitioned full column rank linear pencil 

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

`Ct = C*Q` and `At = Q'*A*Q` are returned in `C` and `A`, respectively, and `Q'*B` is returned in `B` (unless `B = missing').                 

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
function sklf_left!(A::AbstractMatrix{T1}, C::AbstractMatrix{T1}, B::Union{AbstractMatrix{T1},Missing}; fast::Bool = true, atol1::Real = zero(real(eltype(A))), atol2::Real = zero(real(eltype(A))), 
                   rtol::Real = ((size(A,1)+1)*eps(real(float(one(eltype(A))))))*iszero(max(atol1,atol2)), 
                   withQ::Bool = true) where T1 <: BlasFloat
   n = LinearAlgebra.checksquare(A)
   p, n1 = size(C)
   n == n1 || throw(DimensionMismatch("A and C must have the same number of columns"))
   (!ismissing(B) && n !== size(B,1)) && throw(DimensionMismatch("A and B must have the same number of rows"))
   
   μl = Vector{Int}(undef,max(n,p))
   nu = 0
   tol1 = atol1
   withQ ? Q = Matrix{eltype(A)}(I,n,n) : Q = nothing
 
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
   end

   return Q, reverse(μl[1:i]), no, nu
end
