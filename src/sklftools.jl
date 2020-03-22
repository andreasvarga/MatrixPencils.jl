"""
    sreduceBF(A, E, B, C, D; fast = true, atol = 0, rtol, withQ = true, withZ = true) -> F, G, N, Q, Z, n, m, p

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
                   atol::Real = ismissing(A) ? zero(real(eltype(D))) : zero(real(eltype(A))), 
                   rtol::Real = (ismissing(A) ? 1 : min(size(A)...))*eps(real(float(ismissing(A) ? one(real(eltype(D))) : one(real(eltype(A))))))*iszero(atol), 
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
      eltype(D) !== T && (D = convert(Matrix{T},D))
      F = copy(D)
      G = zeros(T,p,m)
      return F, G,  withQ ? Matrix{T}(I,p,p) : nothing, withZ ? Matrix{T}(I,m,m) : nothing, 0, m, p
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
      ndx == size(B,1) || throw(DimensionMismatch("A and B must have the same number of rows"))
      T = promote_type(eltype(A), eltype(E), eltype(B), eltype(E), eltype(A), eltype(E), )
      T <: BlasFloat || (T = promote_type(Float64,T))
      ny = 0
      C = zeros(T,ny,nx)
      nu = size(B,2)
      D = zeros(T,ny,nu)
   else
      isa(A,Adjoint) && (A = copy(A))
      isa(E,Adjoint) && (E = copy(E))
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
      E = Matrix{T}(I,nx,nx)
   end


   (!ismissing(A) && eltype(A) !== T) && (A = convert(Matrix{T},A))
   (!eident && !ismissing(E) && eltype(E) !== T) && (E = convert(Matrix{T},E))
   (!ismissing(B) && eltype(B) !== T) && (B = convert(Matrix{T},B))
   (!ismissing(C) && eltype(C) !== T) && (C = convert(Matrix{T},C))
   (!ismissing(D) && eltype(D) !== T) && (D = convert(Matrix{T},D))

   withQ ? (Q = Matrix{T}(I,ndx+ny,ndx+ny)) : (Q = nothing)
   withZ ? (Z = Matrix{T}(I,nx+nu,nx+nu)) : (Z = nothing)
   if eident
      n1 = nx; m1 = 0; p1 = 0;
   else
      n1, m1, p1 = _preduceBF!(A, E, Q, Z, B, C; atol = atol, rtol = rtol, fast = fast, withQ = withQ, withZ = withQ) 
   end
   F = [ B A; D C]
   G = [ zeros(T,ndx,nu) E; zeros(T,ny,nx+nu)]
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
      if mM == 0 || nM == 0
          return  M, N, Q, Z, νr, μr, ν[1:0], nf, ν, μ
      end

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
      if mM == 0 || nM == 0
         return  M, N, Q, Z, ν, μ, ν[1:0], nf, νl, μl
      end

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
