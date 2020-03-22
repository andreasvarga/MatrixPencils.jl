"""
     lsequal(A1, E1, B1, C1, D1, A2, E2, B2, C2, D2; fastrank = true, atol1 = 0, atol2 = 0, rtol = min(atol1,atol2)>0 ? 0 : n*ϵ) -> flag::Bool

Check if two linearizations `(A1-λE1,B1,C1,D1)` and `(A2-λE2,B2,C2,D2)` satisfy the equivalence condition  
   
                -1                      -1
     C1*(A1-λE1)  *B1 + D1 = C2*(A2-λE2)  *B2 + D2

The ckeck is performed by computing the normal rank `n` of the structured linear matrix pencil `M - λN`

              | A1-λE1   0    |   B1  | 
              |   0    A2-λE2 |   B2  | 
     M - λN = |---------------|-------|
              |   C1     -C2  | D1-D2 |  

and verifying that `n = n1+n2`, where `n1` and `n2` are the orders of the square matrices `A1` and `A2`, respectively.

If `fastrank = true`, the rank is evaluated by counting how many singular values of `M - γ N` have magnitude 
greater than `max(max(atol1,atol2), rtol*σ₁)`, where `σ₁` is the largest singular value of `M - γ N` and 
`γ` is a randomly generated value [1]. 
If `fastrank = false`, the rank is evaluated as `nr + ni + nf + nl`, where `nr` and `nl` are the sums 
of right and left Kronecker indices, respectively, while `ni` and `nf` are the number of finite and 
infinite eigenvalues, respectively. The sums `nr+ni` and  `nf+nl`, are determined from an 
appropriate Kronecker-like form (KLF) exhibiting the spliting of the right and left structures 
of the pencil `M - λN`. For efficiency purpose, the reduction to the relevant KLF is only partially performed 
using rank decisions based on rank revealing SVD-decompositions. 

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of `M`, the absolute tolerance for the nonzero elements of `N`,  and the relative tolerance 
for the nonzero elements of `M` and `N`. The default relative tolerance is `k*ϵ`, where `k` is the size of 
the smallest dimension of `M`, and `ϵ` is the machine epsilon of the element type of `M`. 

[1] A. Varga, On Checking Null Rank Conditions of Rational Matrices, https://arxiv.org/abs/1812.11396, 2018.
"""
function lsequal(A1::AbstractMatrix, E1::Union{AbstractMatrix,UniformScaling{Bool}}, 
   B1::AbstractMatrix, C1::AbstractMatrix, D1::AbstractMatrix,
   A2::AbstractMatrix, E2::Union{AbstractMatrix,UniformScaling{Bool}}, 
   B2::AbstractMatrix, C2::AbstractMatrix, D2::AbstractMatrix; 
   atol1::Real = zero(real(eltype(A1))), atol2::Real = zero(real(eltype(A1))), 
   rtol::Real =  min(size(A1)...)*eps(real(float(one(real(eltype(A1))))))*iszero(max(atol1,atol2)), 
   fastrank::Bool = true)

   # implicit dimensional checks are performed using the try-catch scheme 
   try
      T = promote_type(eltype(A1), eltype(A2))
      n1 = size(A1,1)
      n2 = size(A2,1)
      A = [A1  zeros(T,n1,n2);
           zeros(T,n2,n1) A2]
      B = [B1; B2;]
      C = [C1 -C2;]
      D = [D1-D2;]
      if E1 == I && E2 == I
         E = I
      else
         E = [E1  zeros(T,n1,n2);
              zeros(T,n2,n1) E2]
      end
      return (sprank(A,E,B,C,D,atol1 = atol1, atol2 = atol2, rtol = rtol, fastrank = fastrank) == n1+n2)
   catch
      return false
   end
end
"""
    lsminreal2(A, E, B, C, D; fast = true, atol1 = 0, atol2 = 0, rtol, finite = true, infinite = true, contr = true, obs = true, noseig = true) 
               -> Ar, Er, Br, Cr, Dr, nuc, nuo, nse

Reduce the linearization `(A-λE,B,C,D)` of a rational matrix to a reduced form `(Ar-λEr,Br,Cr,Dr)` such that

             -1                    -1
     C*(A-λE)  *B + D = Cr*(Ar-λEr)  *Br + Dr
     
with the least possible order `nr` of `Ar-λEr` if `finite = true`, `infinite = true`, 
`contr = true`, `obs = true` and `nseig = false`. 
The reduced order linearization satisfies:

     (1) rank[Br Ar-λEr] = nr for all finite λ (finite controllability)

     (2) rank[Br Er] = nr (infinite controllability)

     (3) rank[Ar-λEr; Cr] = nr for all finite λ (finite observability)

     (4) rank[Er; Cr] = nr (infinite observability)

     (5) Ar-λEr has no simple eigenvalues

The achieved dimensional reductions to fulfill conditions (1) and (2), conditions (3) and (4), and 
respectively, condition (5) are returned in `nuc`, `nuo`, `nse`. 

Some reduction steps can be skipped by appropriately selecting the keyword arguments
`contr`, `obs`, `finite`, `infinite` and `nseig`. 

If `contr = false`, then the controllability conditions (1) and (2) are not enforced. 
If `contr = true` and `finite = true`, then the finite controllability condition (1) is enforced. 
If `contr = true` and `infinite = true`, then the infinite controllability condition (2) is enforced. 

If `obs = false`, then observability condition (3) and (4) are not enforced.
If `obs = true` and `finite = true`, then the finite observability condition (3) is enforced.
If `obs = true` and `infinite = true`, then the infinite observability condition (4) is enforced.

If `nseig = false`, then condition (5) on the lack of simple eigenvalues is not enforced. 

To enforce conditions (1)-(4), the `Procedure GIR` in `[1, page 328]` is employed, which performs 
orthogonal similarity transformations on the matrices of the original linearization `(A-λE,B,C,D)` 
to obtain an irreducible linearization using structured pencil reduction algorithms. 
To enforce condition (5), residualization formulas (see, e.g., `[1, page 329]`) are employed which
involves matrix inversions. 

The underlying pencil manipulation algorithms employ rank determinations based on either the use of 
rank revealing QR-decomposition with column pivoting, if `fast = true`, or in the SVD-decomposition.
The rank decision based on the SVD-decomposition is generally more reliable, but the involved computational effort is higher.

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of matrices `A`, `B`, `C`, `D`, the absolute tolerance for the nonzero elements of `E`,  
and the relative tolerance for the nonzero elements of `A`, `B`, `C`, `D` and `E`. 

[1] A. Varga, Solving Fault Diagnosis Problems - Linear Synthesis Techniques, Springer Verlag, 2017. 
"""
function lsminreal2(A::AbstractMatrix, E::Union{AbstractMatrix,UniformScaling{Bool}}, 
                    B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix; 
                    atol1::Real = zero(real(eltype(A))), atol2::Real = zero(real(eltype(A))), 
                    rtol::Real =  min(size(A)...)*eps(real(float(one(real(eltype(A))))))*iszero(max(atol1,atol2)), 
                    fast::Bool = true, finite::Bool = true, infinite::Bool = true, 
                    contr::Bool = true, obs::Bool = true, noseig::Bool = true)

    eident = (typeof(E) == UniformScaling{Bool}) || isequal(E,I) 
    emat = (typeof(E) <: AbstractMatrix)
    isa(A,Adjoint) && (A = copy(A))
    isa(E,Adjoint) && (E = copy(E))
    n = LinearAlgebra.checksquare(A)
    emat && (n,n) !== size(E) && throw(DimensionMismatch("A and E must have the same dimensions"))
    p, m = size(D)
    (n,m) == size(B) || throw(DimensionMismatch("A, B and D must have compatible dimensions"))
    (p,n) == size(C) || throw(DimensionMismatch("A, C and D must have compatible dimensions"))
    T = promote_type(eltype(A), eltype(B), eltype(C), eltype(D))
    eident || (T = promote_type(T,eltype(E)))
    T <: BlasFloat || (T = promote_type(Float64,T))        

    eltype(A) !== T && (A = convert(Matrix{T},A))
    (!eident && eltype(E) !== T) && (E = convert(Matrix{T},E))
    eltype(B) !== T && (B = convert(Matrix{T},B))
    eltype(C) !== T && (C = convert(Matrix{T},C))
    eltype(D) !== T && (D = convert(Matrix{T},D))
    

    (n == 0 || m == 0 || p == 0) && (return A, E, B, C, D, 0, 0, 0)
    # save system matrices
    Ar = copy(A)
    Br = copy(B)
    Cr = copy(C)
    Dr = copy(D)
    Er = copy(E)
    ir = 1:n
    if eident
        if contr
           _, _, nr, nuc = sklf_right!(Ar, Br, Cr; fast = fast, atol1 = atol1, atol2 = atol1, rtol = rtol, withQ = false) 
           if nuc > 0
              ir = 1:nr
              # save intermediary results
              A = Ar[ir,ir]
              B = Br[ir,:]
              C = Cr[:,ir]
           else
              # restore original matrices 
              Ar = copy(A)
              Br = copy(B)
              Cr = copy(C)
           end
        else
           nuc = 0
        end
        if obs
            _, _, no, nuo = sklf_left!(view(Ar,ir,ir), view(Cr,:,ir), view(Br,ir,:); fast = fast, atol1 = atol1, atol2 = atol1, rtol = rtol, withQ = false) 
            if nuo > 0
               ir = ir[end-no+1:end]
            else
               # restore saved matrices
               Ar[ir,ir] = A
               Br[ir,:] = B
               Cr[:,ir] = C
            end
        else
           nuo = 0
        end
        return Ar[ir,ir], emat ? Er[ir,ir] : I, Br[ir,:], Cr[:,ir], Dr, nuc, nuo, 0
    else
        if finite
           if contr  
              _, _, _, nr, nfuc = sklf_rightfin!(Ar, Er, Br, Cr; fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol, withQ = false, withZ = false) 
              if nfuc > 0
                 ir = 1:nr
                 # save intermediary results
                 A = Ar[ir,ir]
                 E = Er[ir,ir]
                 B = Br[ir,:]
                 C = Cr[:,ir]
              else
                 # restore original matrices 
                 Ar = copy(A)
                 Er = copy(E)
                 Br = copy(B)
                 Cr = copy(C)
              end
           else
              nfuc = 0
           end
           if obs 
               _, _, _, no, nfuo = sklf_leftfin!(view(Ar,ir,ir), view(Er,ir,ir), view(Cr,:,ir), view(Br,ir,:); 
                                                fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol, withQ = false, withZ = false) 
               if nfuo > 0
                  ir = ir[end-no+1:end]
                  # save intermediary results
                  A = Ar[ir,ir]
                  E = Er[ir,ir]
                  B = Br[ir,:]
                  C = Cr[:,ir]
               else
                  # restore saved matrices
                  Ar[ir,ir] = A
                  Er[ir,ir] = E
                  Br[ir,:] = B
                  Cr[:,ir] = C
               end
           else
               nfuo = 0
           end
         else
            nfuc = 0
            nfuo = 0
         end
         if infinite
            if contr  
               _, _, _, nr, niuc = sklf_rightfin!(view(Er,ir,ir), view(Ar,ir,ir), view(Br,ir,:), view(Cr,:,ir); 
                                                 fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol, 
                                                 withQ = false, withZ = false) 
               if niuc > 0
                  ir = ir[1:nr]
                  # save intermediary results
                  A = Ar[ir,ir]
                  E = Er[ir,ir]
                  B = Br[ir,:]
                  C = Cr[:,ir]
               else
                  # restore original matrices 
                  Ar[ir,ir] = A
                  Er[ir,ir] = E
                  Br[ir,:] = B
                  Cr[:,ir] = C
               end
            else
               niuc = 0
            end
            if obs 
                _, _, _, no, niuo = sklf_leftfin!(view(Er,ir,ir), view(Ar,ir,ir), view(Cr,:,ir), view(Br,ir,:); 
                                                 fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol, 
                                                 withQ = false, withZ = false) 
                if niuo > 0
                   ir = ir[end-no+1:end]
                   # save intermediary results
                   A = Ar[ir,ir]
                   E = Er[ir,ir]
                   B = Br[ir,:]
                   C = Cr[:,ir]
                else
                   # restore saved matrices
                   Ar[ir,ir] = A
                   Er[ir,ir] = E
                   Br[ir,:] = B
                   Cr[:,ir] = C
                end
            else
                niuo = 0
            end
         else
            niuc = 0
            niuo = 0
          end
          nuc = nfuc+niuc
          nuo = nfuo+niuo
          if noseig
            rE, rA22  = _svdlikeAE!(view(Ar,ir,ir), view(Er,ir,ir), nothing, nothing, view(Br,ir,:), view(Cr,:,ir), 
                        fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol, withQ = false, withZ = false)
            if rA22 > 0
               i1 = ir[1:rE]
               i2 = ir[rE+1:rE+rA22]
               # make A22 = I
               fast ? (A22 = UpperTriangular(Ar[i2,i2])) : (A22 = Diagonal(Ar[i2,i2]))
               ldiv!(A22,view(Ar,i2,i1))
               ldiv!(A22,view(Br,i2,:))
               # apply simplified residualization formulas
               Dr -= Cr[:,i2]*Br[i2,:]
               Br[i1,:] -= Ar[i1,i2]*Br[i2,:]
               Cr[:,i1] -= Cr[:,i2]*Ar[i2,i1]
               Ar[i1,i1] -= Ar[i1,i2]*Ar[i2,i1]
               ir = [i1; ir[rE+rA22+1:end]]
            else
               # restore saved matrices
               Ar[ir,ir] = A
               Er[ir,ir] = E
               Br[ir,:] = B
               Cr[:,ir] = C
            end
            return Ar[ir,ir], Er[ir,ir], Br[ir,:], Cr[:,ir], Dr, nuc, nuo, rA22
         else
            return Ar[ir,ir], Er[ir,ir], Br[ir,:], Cr[:,ir], Dr, nuc, nuo, 0
         end
     end
end
"""
    lsminreal(A, E, B, C, D; fast = true, atol1 = 0, atol2, rtol, contr = true, obs = true, noseig = true) -> Ar, Er, Br, Cr, Dr, nuc, nuo, nse

Reduce the linearization `(A-λE,B,C,D)` of a rational matrix to a reduced form `(Ar-λEr,Br,Cr,Dr)` such that

             -1                    -1
     C*(A-λE)  *B + D = Cr*(Ar-λEr)  *Br + Dr
     
with the least possible order `nr` of `Ar-λEr` if `contr = true`, `obs = true` and `nseig = false`. 
The reduced order linearization satisfies:

     (1) rank[Br Ar-λEr] = nr for all finite λ (finite controllability)

     (2) rank[Br Er] = nr (infinite controllability)

     (3) rank[Ar-λEr; Cr] = nr for all finite λ (finite observability)

     (4) rank[Er; Cr] = nr (infinite observability)

     (5) Ar-λEr has no simple eigenvalues

The achieved dimensional reductions to fulfill conditions (1) and (2), conditions (3) and (4), and 
respectively, condition (5) are returned in `nuc`, `nuo`, `nse`. 

Some reduction steps can be skipped by appropriately selecting the keyword arguments
`contr`, `obs` and `nseig`. 

If `contr = false`, then the controllability conditions (1) and (2) are not enforced. 

If `obs = false`, then observability condition (3) and (4) are not enforced.

If `nseig = false`, then condition (5) on the lack of simple eigenvalues is not enforced. 

To enforce conditions (1)-(4), orthogonal similarity transformations are performed on 
the matrices of the original linearization `(A-λE,B,C,D)` to obtain an irreducible linearization using
structured pencil reduction algorithms, as the fast versions of the reduction techniques of the 
full row rank pencil [B A-λE] and full column rank pencil [A-λE;C] proposed in [1]. 
To enforce condition (5), residualization formulas (see, e.g., `[2, page 329]`) are employed which
involves matrix inversions. 

The underlying pencil manipulation algorithms employ rank determinations based on either the use of 
rank revealing QR-decomposition with column pivoting, if `fast = true`, or in the SVD-decomposition.
The rank decision based on the SVD-decomposition is generally more reliable, but the involved computational effort is higher.

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of matrices `A`, `B`, `C`, `D`, the absolute tolerance for the nonzero elements of `E`,  
and the relative tolerance for the nonzero elements of `A`, `B`, `C`, `D` and `E`. 

[1] P. Van Dooreen, The generalized eigenstructure problem in linear system theory, 
IEEE Transactions on Automatic Control, vol. AC-26, pp. 111-129, 1981.

[2] A. Varga, Solving Fault Diagnosis Problems - Linear Synthesis Techniques, Springer Verlag, 2017. 
"""
function lsminreal(A::AbstractMatrix, E::Union{AbstractMatrix,UniformScaling{Bool}}, 
                   B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix; 
                   atol1::Real = zero(real(eltype(A))), atol2::Real = zero(real(eltype(A))), 
                   rtol::Real =  min(size(A)...)*eps(real(float(one(real(eltype(A))))))*iszero(max(atol1,atol2)), 
                   fast::Bool = true, contr::Bool = true, obs::Bool = true, noseig::Bool = true)

    eident = (typeof(E) == UniformScaling{Bool}) || isequal(E,I) 
    emat = (typeof(E) <: AbstractMatrix)
    isa(A,Adjoint) && (A = copy(A))
    isa(E,Adjoint) && (E = copy(E))
    n = LinearAlgebra.checksquare(A)
    emat && (n,n) !== size(E) && throw(DimensionMismatch("A and E must have the same dimensions"))
    p, m = size(D)
    (n,m) == size(B) || throw(DimensionMismatch("A, B and D must have compatible dimensions"))
    (p,n) == size(C) || throw(DimensionMismatch("A, C and D must have compatible dimensions"))
    T = promote_type(eltype(A), eltype(B), eltype(C), eltype(D))
    eident || (T = promote_type(T,eltype(E)))
    T <: BlasFloat || (T = promote_type(Float64,T))        

    eltype(A) !== T && (A = convert(Matrix{T},A))
    (!eident && eltype(E) !== T) && (E = convert(Matrix{T},E))
    eltype(B) !== T && (B = convert(Matrix{T},B))
    eltype(C) !== T && (C = convert(Matrix{T},C))
    eltype(D) !== T && (D = convert(Matrix{T},D))
    

    (n == 0 || m == 0 || p == 0) && (return A, E, B, C, D, 0, 0, 0)
    # save system matrices
    Ar = copy(A)
    Br = copy(B)
    Cr = copy(C)
    Dr = copy(D)
    Er = copy(E)
    ir = 1:n
    if eident
        if contr
           _, _, nr, nuc = sklf_right!(Ar, Br, Cr; fast = fast, atol1 = atol1, atol2 = atol1, rtol = rtol, withQ = false) 
           if nuc > 0
              ir = 1:nr
              # save intermediary results
              A = Ar[ir,ir]
              B = Br[ir,:]
              C = Cr[:,ir]
           else
              # restore original matrices 
              Ar = copy(A)
              Br = copy(B)
              Cr = copy(C)
           end
        else
           nuc = 0
        end
        if obs
            _, _, no, nuo = sklf_left!(view(Ar,ir,ir), view(Cr,:,ir), view(Br,ir,:); fast = fast, atol1 = atol1, atol2 = atol1, rtol = rtol, withQ = false) 
            if nuo > 0
               ir = ir[end-no+1:end]
            else
               # restore saved matrices
               Ar[ir,ir] = A
               Br[ir,:] = B
               Cr[:,ir] = C
            end
        else
           nuo = 0
        end
        return Ar[ir,ir], emat ? Er[ir,ir] : I, Br[ir,:], Cr[:,ir], Dr, nuc, nuo, 0
    else
        if contr  
            _, _, _, nr, nfuc, niuc = sklf_right!(Ar, Er, Br, Cr; fast = fast, atol1 = atol1, atol2 = atol2, atol3 = atol1, rtol = rtol, withQ = false, withZ = false) 
            nuc = nfuc+niuc
            if nuc > 0
               ir = 1:nr
               # save intermediary results
               A = Ar[ir,ir]
               E = Er[ir,ir]
               B = Br[ir,:]
               C = Cr[:,ir]
            else
               # restore original matrices 
               Ar = copy(A)
               Er = copy(E)
               Br = copy(B)
               Cr = copy(C)
            end
         else
            nuc = 0
         end
         if obs 
            _, _, _, no, nfuo, niuo = sklf_left!(view(Ar,ir,ir), view(Er,ir,ir), view(Cr,:,ir), view(Br,ir,:); 
                                               fast = fast, atol1 = atol1, atol2 = atol2, atol3 = atol1, 
                                               rtol = rtol, withQ = false, withZ = false) 
             nuo = nfuo+niuo
             if nuo > 0
                ir = ir[end-no+1:end]
                # save intermediary results
                A = Ar[ir,ir]
                E = Er[ir,ir]
                B = Br[ir,:]
                C = Cr[:,ir]
             else
                # restore saved matrices
                Ar[ir,ir] = A
                Er[ir,ir] = E
                Br[ir,:] = B
                Cr[:,ir] = C
             end
         else
             nuo = 0
         end
         if noseig
            rE, rA22  = _svdlikeAE!(view(Ar,ir,ir), view(Er,ir,ir), nothing, nothing, view(Br,ir,:), view(Cr,:,ir), 
                        fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol, withQ = false, withZ = false)
            if rA22 > 0
               i1 = ir[1:rE]
               i2 = ir[rE+1:rE+rA22]
               # make A22 = I
               fast ? (A22 = UpperTriangular(Ar[i2,i2])) : (A22 = Diagonal(Ar[i2,i2]))
               ldiv!(A22,view(Ar,i2,i1))
               ldiv!(A22,view(Br,i2,:))
               # apply simplified residualization formulas
               Dr -= Cr[:,i2]*Br[i2,:]
               Br[i1,:] -= Ar[i1,i2]*Br[i2,:]
               Cr[:,i1] -= Cr[:,i2]*Ar[i2,i1]
               Ar[i1,i1] -= Ar[i1,i2]*Ar[i2,i1]
               ir = [i1; ir[rE+rA22+1:end]]
            else
               # restore saved matrices
               Ar[ir,ir] = A
               Er[ir,ir] = E
               Br[ir,:] = B
               Cr[:,ir] = C
            end
            return Ar[ir,ir], Er[ir,ir], Br[ir,:], Cr[:,ir], Dr, nuc, nuo, rA22
         else
            return Ar[ir,ir], Er[ir,ir], Br[ir,:], Cr[:,ir], Dr, nuc, nuo, 0
         end
     end
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
                   rtol::Real = (min(size(A)...)*eps(real(float(one(eltype(A))))))*iszero(max(atol1,atol2,atol3)), 
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
                       rtol::Real = (min(size(A)...)*eps(real(float(one(T1)))))*iszero(max(atol1,atol2)), 
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
   if p == 0 && n == 0
      return Q, Z, μl[1:0], 0, 0
   elseif n == 0
      μl[1] = 0
      return Q, Z, μl[1:1], 0, 0
   end

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
                   rtol::Real = (min(size(A)...)*eps(real(float(one(eltype(A))))))*iszero(max(atol1,atol2,atol3)), 
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
                        rtol::Real = (min(size(A)...)*eps(real(float(one(T1)))))*iszero(max(atol1,atol2)), 
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
   if m == 0 && n == 0
      return Q, Z, νr[1:0], 0, 0
   elseif n == 0
      νr[1] = 0
      return Q, Z, νr[1:1], 0, 0
   end

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
                     rtol::Real = (min(size(A)...)*eps(real(float(one(eltype(A))))))*iszero(max(atol1,atol2)), 
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
                   rtol::Real = (min(size(A)...)*eps(real(float(one(eltype(A))))))*iszero(max(atol1,atol2)), 
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
