"""
    lseval(A, E, B, C, D, val) 

Evaluate the rational matrix `G(λ) = C*inv(λE-A)*B+D` for `λ = val`. 
"""
function lseval(A::AbstractMatrix, E::Union{AbstractMatrix,UniformScaling{Bool}}, 
                B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix, val::Number)
   return C*((val*E-A)\B)+D
end
"""
    lpseval(A, E, B, F, C, G, D, H, val) 

Evaluate the rational matrix `G(λ) = (C-λG)*inv(λE-A)*(B-λF)+D-λH` for `λ = val`. 
"""
function lpseval(A::AbstractMatrix, E::Union{AbstractMatrix,UniformScaling{Bool}}, 
                 B::AbstractMatrix, F::AbstractMatrix, C::AbstractMatrix, G::AbstractMatrix, 
                 D::AbstractMatrix, H::AbstractMatrix, val::Number)
   return (C-val*G)*((val*E-A)\(B-val*F))+D-val*H
end
"""
     lsequal(A1, E1, B1, C1, D1, A2, E2, B2, C2, D2; 
             fastrank = true, atol1 = 0, atol2 = 0, rtol = min(atol1,atol2)>0 ? 0 : n*ϵ) -> flag::Bool

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
of right and left Kronecker indices, respectively, while `ni` and `nf` are the number of infinite and 
finite eigenvalues, respectively. The sums `nr+ni` and  `nf+nl`, are determined from an 
appropriate Kronecker-like form (KLF) exhibiting the spliting of the right and left structures 
of the pencil `M - λN`. For efficiency purpose, the reduction to the relevant KLF is only partially performed 
using rank decisions based on rank revealing SVD-decompositions. 

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of `M`, the absolute tolerance for the nonzero elements of `N` and the relative tolerance 
for the nonzero elements of `M` and `N`. The default relative tolerance is `k*ϵ`, where `k` is the size of 
the smallest dimension of `M`, and `ϵ` is the machine epsilon of the element type of `M`. 

[1] A. Varga, On Checking Null Rank Conditions of Rational Matrices, [https://arxiv.org/abs/1812.11396](https://arxiv.org/abs/1812.11396), 2018.
"""
function lsequal(A1::AbstractMatrix, E1::Union{AbstractMatrix,UniformScaling{Bool}}, 
   B1::AbstractMatrix, C1::AbstractMatrix, D1::AbstractMatrix,
   A2::AbstractMatrix, E2::Union{AbstractMatrix,UniformScaling{Bool}}, 
   B2::AbstractMatrix, C2::AbstractMatrix, D2::AbstractMatrix; 
   atol1::Real = zero(real(eltype(A1))), atol2::Real = zero(real(eltype(A1))), 
   rtol::Real =  (size(A1,1)+size(A2,1)+2)*eps(real(float(one(real(eltype(A1))))))*iszero(max(atol1,atol2)), 
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
   catch err
      println("$err")
      return false
   end
end
"""
     lpsequal(A1, E1, B1, F1, C1, G1, D1, H1, 
              A2, E2, B2, F2, C2, G2, D2, H2; fastrank = true, atol1 = 0, atol2 = 0, rtol = min(atol1,atol2)>0 ? 0 : n*ϵ) 
              -> flag::Bool

Check if two linear pencil linearizations `(A1-λE1,B1-λF1,C1-λG1,D1-λH1)` and `(A2-λE2,B2-λF2,C2-λG2,D2-λH2)` 
satisfy the equivalence condition  
   
                      -1                                      -1
     (C1-λG1)*(A1-λE1)  *(B1-λF1) + D1-λH1 = (C2-λG2)*(A2-λE2)  *(B2-λF2) + D2-λH2

The ckeck is performed by computing the normal rank `n` of the structured linear matrix pencil `M - λN`

              | A1-λE1    0    |    B1-λF1      | 
              |   0     A2-λE2 |    B2-λF2      | 
     M - λN = |----------------|----------------|
              | C1-λG1 -C2+λG2 | D1-D2-λ(H1-H2) |  

and verifying that `n = n1+n2`, where `n1` and `n2` are the orders of the square matrices `A1` and `A2`, respectively.

If `fastrank = true`, the rank is evaluated by counting how many singular values of `M - γ N` have magnitude 
greater than `max(max(atol1,atol2), rtol*σ₁)`, where `σ₁` is the largest singular value of `M - γ N` and 
`γ` is a randomly generated value [1]. 
If `fastrank = false`, the rank is evaluated as `nr + ni + nf + nl`, where `nr` and `nl` are the sums 
of right and left Kronecker indices, respectively, while `ni` and `nf` are the number of infinite and 
finite eigenvalues, respectively. The sums `nr+ni` and  `nf+nl`, are determined from an 
appropriate Kronecker-like form (KLF) exhibiting the spliting of the right and left structures 
of the pencil `M - λN`. For efficiency purpose, the reduction to the relevant KLF is only partially performed 
using rank decisions based on rank revealing SVD-decompositions. 

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of `M`, the absolute tolerance for the nonzero elements of `N`,  and the relative tolerance 
for the nonzero elements of `M` and `N`. The default relative tolerance is `k*ϵ`, where `k` is the size of 
the smallest dimension of `M`, and `ϵ` is the machine epsilon of the element type of `M`. 

[1] A. Varga, On Checking Null Rank Conditions of Rational Matrices, [https://arxiv.org/abs/1812.11396](https://arxiv.org/abs/1812.11396), 2018.
"""
function lpsequal(A1::AbstractMatrix, E1::Union{AbstractMatrix,UniformScaling{Bool}}, 
   B1::AbstractMatrix, F1::AbstractMatrix, C1::AbstractMatrix, G1::AbstractMatrix, 
   D1::AbstractMatrix, H1::AbstractMatrix,
   A2::AbstractMatrix, E2::Union{AbstractMatrix,UniformScaling{Bool}}, 
   B2::AbstractMatrix, F2::AbstractMatrix, C2::AbstractMatrix, G2::AbstractMatrix, 
   D2::AbstractMatrix, H2::AbstractMatrix; 
   atol1::Real = zero(real(eltype(A1))), atol2::Real = zero(real(eltype(A1))), 
   rtol::Real =  (size(A1,1)+size(A2,1)+4)*eps(real(float(one(real(eltype(A1))))))*iszero(max(atol1,atol2)), 
   fastrank::Bool = true)

   # implicit dimensional checks are performed using the try-catch scheme 
   try
      T = promote_type(eltype(A1), eltype(A2))
      n1 = size(A1,1)
      n2 = size(A2,1)
      A = [A1  zeros(T,n1,n2);
           zeros(T,n2,n1) A2]
      B = [B1; B2;]
      F = [F1; F2;]
      C = [C1 -C2;]
      G = [G1 -G2;]
      D = [D1-D2;]
      H = [H1-H2;]
      if E1 == I && E2 == I
         E = I
      else
         E = [E1  zeros(T,n1,n2);
              zeros(T,n2,n1) E2]
      end
      return (prank([A B;C D],[E F;G H], atol1 = atol1, atol2 = atol2, rtol = rtol, fastrank = fastrank) == n1+n2)
   catch err
      println("$err")
      return false
   end
end
"""
    lsminreal2(A, E, B, C, D; fast = true, atol1 = 0, atol2 = 0, rtol, finite = true, infinite = true, contr = true, obs = true, noseig = true) 
               -> (Ar, Er, Br, Cr, Dr, nuc, nuo, nse)

Reduce the linearization `(A-λE,B,C,D)` of a rational matrix to a reduced form `(Ar-λEr,Br,Cr,Dr)` such that

             -1                    -1
     C*(A-λE)  *B + D = Cr*(Ar-λEr)  *Br + Dr
     
with the least possible order `nr` of `Ar-λEr` if `finite = true`, `infinite = true`, 
`contr = true`, `obs = true` and `nseig = true`. Such a realization is called `minimal` and satisfies:

     (1) rank[Br Ar-λEr] = nr for all finite λ (finite controllability);

     (2) rank[Br Er] = nr (infinite controllability);

     (3) rank[Ar-λEr; Cr] = nr for all finite λ (finite observability);

     (4) rank[Er; Cr] = nr (infinite observability);

     (5) Ar-λEr has no simple infinite eigenvalues.

A realization satisfying only conditions (1)-(4) is called `irreducible`. 

The achieved dimensional reductions to fulfill conditions (1) and (2), conditions (3) and (4), and 
respectively, condition (5) are returned in `nuc`, `nuo`, `nse`. 

Some reduction steps can be skipped by appropriately selecting the keyword arguments
`contr`, `obs`, `finite`, `infinite` and `nseig`. 

If `contr = false`, then the controllability conditions (1) and (2) are not enforced. 
If `contr = true` and `finite = true`, then the finite controllability condition (1) is enforced. 
If `contr = true` and `finite = false`, then the finite controllability condition (1) is not enforced. 
If `contr = true` and `infinite = true`, then the infinite controllability condition (2) is enforced. 
If `contr = true` and `infinite = false`, then the infinite controllability condition (2) is not enforced. 

If `obs = false`, then observability condition (3) and (4) are not enforced.
If `obs = true` and `finite = true`, then the finite observability condition (3) is enforced.
If `obs = true` and `finite = false`, then the finite observability condition (3) is not enforced.
If `obs = true` and `infinite = true`, then the infinite observability condition (4) is enforced.
If `obs = true` and `infinite = false`, then the infinite observability condition (4) is not enforced.

If `nseig = false`, then condition (5) on the lack of simple infinite eigenvalues is not enforced. 

To enforce conditions (1)-(4), the `Procedure GIR` in `[1, page 328]` is employed, which performs 
orthogonal similarity transformations on the matrices of the original linearization `(A-λE,B,C,D)` 
to obtain an irreducible linearization using structured pencil reduction algorithms. 
To enforce condition (5), residualization formulas (see, e.g., `[1, page 329]`) are employed which
involves matrix inversions. 

The underlying pencil manipulation algorithms employ rank determinations based on either the use of 
rank revealing QR-decomposition with column pivoting, if `fast = true`, or the SVD-decomposition.
The rank decision based on the SVD-decomposition is generally more reliable, but the involved computational effort is higher.

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of matrices `A`, `B`, `C`, `D`, the absolute tolerance for the nonzero elements of `E`,  
and the relative tolerance for the nonzero elements of `A`, `B`, `C`, `D` and `E`. 

[1] A. Varga, Solving Fault Diagnosis Problems - Linear Synthesis Techniques, Springer Verlag, 2017. 
"""
function lsminreal2(A::AbstractMatrix, E::Union{AbstractMatrix,UniformScaling{Bool}}, 
                    B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix; 
                    atol1::Real = zero(real(eltype(A))), atol2::Real = zero(real(eltype(A))), 
                    rtol::Real =  (size(A,1)+1)*eps(real(float(one(real(eltype(A))))))*iszero(max(atol1,atol2)), 
                    fast::Bool = true, finite::Bool = true, infinite::Bool = true, 
                    contr::Bool = true, obs::Bool = true, noseig::Bool = true)

    emat = (typeof(E) <: AbstractMatrix)
    eident = !emat || isequal(E,I) 
    n = LinearAlgebra.checksquare(A)
    emat && (n,n) !== size(E) && throw(DimensionMismatch("A and E must have the same dimensions"))
    p, m = size(D)
    (n,m) == size(B) || throw(DimensionMismatch("A, B and D must have compatible dimensions"))
    (p,n) == size(C) || throw(DimensionMismatch("A, C and D must have compatible dimensions"))
    T = promote_type(eltype(A), eltype(B), eltype(C), eltype(D))
    eident || (T = promote_type(T,eltype(E)))
    T <: BlasFloat || (T = promote_type(Float64,T))        

    A1 = copy_oftype(A,T)   
    eident ? E1 = copy(E) : E1 = copy_oftype(E,T)
    B1 = copy_oftype(B,T)
    C1 = copy_oftype(C,T)
    D1 = copy_oftype(D,T)  

    (n == 0 || m == 0 || p == 0) && (return A1, E1, B1, C1, D1, 0, 0, 0)
    # save system matrices
    Ar = copy(A1)
    Br = copy(B1)
    Cr = copy(C1)
    Dr = copy(D1)
    Er = copy(E1)
    ir = 1:n
    if eident
        if contr
           _, _, nr, nuc = sklf_right!(Ar, Br, Cr; fast = fast, atol1 = atol1, atol2 = atol1, rtol = rtol, withQ = false) 
           if nuc > 0
              ir = 1:nr
              # save intermediary results
              A1 = Ar[ir,ir]
              B1 = Br[ir,:]
              C1 = Cr[:,ir]
           else
              # restore original matrices 
              Ar = copy(A1)
              Br = copy(B1)
              Cr = copy(C1)
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
               Ar[ir,ir] = A1
               Br[ir,:] = B1
               Cr[:,ir] = C1
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
                 A1 = Ar[ir,ir]
                 E1 = Er[ir,ir]
                 B1 = Br[ir,:]
                 C1 = Cr[:,ir]
              else
                 # restore original matrices 
                 Ar = copy(A1)
                 Er = copy(E1)
                 Br = copy(B1)
                 Cr = copy(C1)
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
                  A1 = Ar[ir,ir]
                  E1 = Er[ir,ir]
                  B1 = Br[ir,:]
                  C1 = Cr[:,ir]
               else
                  # restore saved matrices
                  Ar[ir,ir] = A1
                  Er[ir,ir] = E1
                  Br[ir,:] = B1
                  Cr[:,ir] = C1
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
                  A1 = Ar[ir,ir]
                  E1 = Er[ir,ir]
                  B1 = Br[ir,:]
                  C1 = Cr[:,ir]
               else
                  # restore original matrices 
                  Ar[ir,ir] = A1
                  Er[ir,ir] = E1
                  Br[ir,:] = B1
                  Cr[:,ir] = C1
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
                   A1 = Ar[ir,ir]
                   E1 = Er[ir,ir]
                   B1 = Br[ir,:]
                   C1 = Cr[:,ir]
                else
                   # restore saved matrices
                   Ar[ir,ir] = A1
                   Er[ir,ir] = E1
                   Br[ir,:] = B1
                   Cr[:,ir] = C1
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
               Ar[ir,ir] = A1
               Er[ir,ir] = E1
               Br[ir,:] = B1
               Cr[:,ir] = C1
            end
            return Ar[ir,ir], Er[ir,ir], Br[ir,:], Cr[:,ir], Dr, nuc, nuo, rA22
         else
            return Ar[ir,ir], Er[ir,ir], Br[ir,:], Cr[:,ir], Dr, nuc, nuo, 0
         end
     end
end
"""
    lsminreal(A, E, B, C, D; fast = true, atol1 = 0, atol2, rtol, contr = true, obs = true, noseig = true) 
              -> (Ar, Er, Br, Cr, Dr, nuc, nuo, nse)

Reduce the linearization `(A-λE,B,C,D)` of a rational matrix to a reduced form `(Ar-λEr,Br,Cr,Dr)` such that

             -1                    -1
     C*(A-λE)  *B + D = Cr*(Ar-λEr)  *Br + Dr
     
with the least possible order `nr` of `Ar-λEr` if `contr = true`, `obs = true` and `nseig = true`. 
Such a realization is called `minimal` and satisfies:

     (1) rank[Br Ar-λEr] = nr for all finite λ (finite controllability)

     (2) rank[Br Er] = nr (infinite controllability)

     (3) rank[Ar-λEr; Cr] = nr for all finite λ (finite observability)

     (4) rank[Er; Cr] = nr (infinite observability)

     (5) Ar-λEr has no simple infinite eigenvalues

A realization satisfying only conditions (1)-(4) is called `irreducible`. 

The achieved dimensional reductions to fulfill conditions (1) and (2), conditions (3) and (4), and 
respectively, condition (5) are returned in `nuc`, `nuo`, `nse`. 

Some reduction steps can be skipped by appropriately selecting the keyword arguments
`contr`, `obs` and `nseig`. 

If `contr = false`, then the controllability conditions (1) and (2) are not enforced. 

If `obs = false`, then observability condition (3) and (4) are not enforced.

If `nseig = false`, then condition (5) on the lack of simple infinite eigenvalues is not enforced. 

To enforce conditions (1)-(4), orthogonal similarity transformations are performed on 
the matrices of the original linearization `(A-λE,B,C,D)` to obtain an irreducible linearization using
structured pencil reduction algorithms, as the fast versions of the reduction techniques of the 
full row rank pencil [B A-λE] and full column rank pencil [A-λE;C] proposed in [1]. 
To enforce condition (5), residualization formulas (see, e.g., `[2, page 329]`) are employed which
involves matrix inversions. 

The underlying pencil manipulation algorithms employ rank determinations based on either the use of 
rank revealing QR-decomposition with column pivoting, if `fast = true`, or the SVD-decomposition.
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
                   rtol::Real =  (size(A,1)+1)*eps(real(float(one(real(eltype(A))))))*iszero(max(atol1,atol2)), 
                   fast::Bool = true, contr::Bool = true, obs::Bool = true, noseig::Bool = true)

    emat = (typeof(E) <: AbstractMatrix)
    eident = !emat || isequal(E,I) 
    n = LinearAlgebra.checksquare(A)
    emat && (n,n) !== size(E) && throw(DimensionMismatch("A and E must have the same dimensions"))
    p, m = size(D)
    (n,m) == size(B) || throw(DimensionMismatch("A, B and D must have compatible dimensions"))
    (p,n) == size(C) || throw(DimensionMismatch("A, C and D must have compatible dimensions"))
    T = promote_type(eltype(A), eltype(B), eltype(C), eltype(D))
    eident || (T = promote_type(T,eltype(E)))
    T <: BlasFloat || (T = promote_type(Float64,T))        

    A1 = copy_oftype(A,T)   
    eident ? E1 = copy(E) : E1 = copy_oftype(E,T)
    B1 = copy_oftype(B,T)
    C1 = copy_oftype(C,T)
    D1 = copy_oftype(D,T)  

    (n == 0 || m == 0 || p == 0) && (return A1, E1, B1, C1, D1, 0, 0, 0)
    # save system matrices
    Ar = copy(A1)
    Br = copy(B1)
    Cr = copy(C1)
    Dr = copy(D1)
    Er = copy(E1)
    ir = 1:n
    if eident
        if contr
           _, _, nr, nuc = sklf_right!(Ar, Br, Cr; fast = fast, atol1 = atol1, atol2 = atol1, rtol = rtol, withQ = false) 
           if nuc > 0
              ir = 1:nr
              # save intermediary results
              A1 = Ar[ir,ir]
              B1 = Br[ir,:]
              C1 = Cr[:,ir]
           else
              # restore original matrices 
              Ar = copy(A1)
              Br = copy(B1)
              Cr = copy(C1)
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
               Ar[ir,ir] = A1
               Br[ir,:] = B1
               Cr[:,ir] = C1
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
               A1 = Ar[ir,ir]
               E1 = Er[ir,ir]
               B1 = Br[ir,:]
               C1 = Cr[:,ir]
            else
               # restore original matrices 
               Ar = copy(A1)
               Er = copy(E1)
               Br = copy(B1)
               Cr = copy(C1)
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
                A1 = Ar[ir,ir]
                E1 = Er[ir,ir]
                B1 = Br[ir,:]
                C1 = Cr[:,ir]
             else
                # restore saved matrices
                Ar[ir,ir] = A1
                Er[ir,ir] = E1
                Br[ir,:] = B1
                Cr[:,ir] = C1
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
               Ar[ir,ir] = A1
               Er[ir,ir] = E1
               Br[ir,:] = B1
               Cr[:,ir] = C1
            end
            return Ar[ir,ir], Er[ir,ir], Br[ir,:], Cr[:,ir], Dr, nuc, nuo, rA22
         else
            return Ar[ir,ir], Er[ir,ir], Br[ir,:], Cr[:,ir], Dr, nuc, nuo, 0
         end
     end
end
"""
    lpsminreal(A, E, B, F, C, G, D, H; fast = true, atol1 = 0, atol2, rtol, contr = true, obs = true)  
               -> (Ar, Er, Br, Fr, Cr, Gr, Dr, Hr, V, W, nuc, nuo)

Reduce the linearization `(A-λE,B-λF,C-λG,D-λH)` of a rational matrix to a reduced form 
`(Ar-λEr,Br-λFr,Cr-λGr,Dr-λHr)` such that, for appropriate 
invertible upper triangular matrices `V` and `W`, 

                      -1                                     -1
     V'*((C-λG)*(A-λE)  *(B-λF) + D-λH)*W = (Cr-λGr)*(Ar-λEr)  *(Br-λFr) + Dr-λHr
     
with the least possible order `nr` of `Ar-λEr` if `contr = true` and `obs = true`.
Such a realization is called `strongly minimal` and satisfies:

     (1) rank[Br-λFr Ar-λEr] = nr for all finite and infinite λ (strong controllability)

     (2) rank[Ar-λEr; Cr-λGr] = nr for all finite and infinite λ (strong observability)

The achieved dimensional reductions to fulfill conditions (1) and (2) are 
returned in `nuc` and `nuo`, respectively. 

If `contr = true`, then the strong controllability condition (1) is enforced and `W` is an invertible upper triangular matrix or 
`W = I` if `nuc = 0`.
If `contr = false`, then the strong controllability condition (1) is not enforced and `W = I`. 

If `obs = true`, then the strong observability condition (2) is enforced and `V` is an invertible upper triangular matrix or 
`V = I` if `nuo = 0`. 
If `obs = false`, then the strong observability condition (2) is not enforced and `V = I`.

To enforce conditions (1) and (2), orthogonal similarity transformations are performed on 
the matrices of the original linearization `(A-λE,B-λF,C-λG,D-λH)` to obtain a strongly minimal linearization 
using structured pencil reduction algorithms [1]. The resulting realization `(Ar-λEr,Br-λFr,Cr-λGr,Dr-λHr)`
fulfills the strong controllability and strong observability conditions established in [2]. 

The underlying pencil manipulation algorithms employ rank determinations based on either the use of 
rank revealing QR-decomposition with column pivoting, if `fast = true`, or the SVD-decomposition, if `fast = false`.
The rank decision based on the SVD-decomposition is generally more reliable, but the involved computational effort is higher.

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of matrices `A`, `B`, `C`, `D`, the absolute tolerance for the nonzero elements of `E`, `F`, `G`, `H`  
and the relative tolerance for the nonzero elements of `A`, `B`, `C`, `D` and `E`, `F`, `G`, `H`. 

[1] F.M. Dopico, M.C. Quintana and P. Van Dooren, Linear system matrices of rational transfer functions, 
to appear in "Realization and Model Reduction of Dynamical Systems", A Festschrift to honor the 70th birthday of Thanos Antoulas", 
Springer-Verlag. [arXiv: 1903.05016](https://arxiv.org/pdf/1903.05016.pdf)

[2] G. Verghese, Comments on ‘Properties of the system matrix of a generalized state-space system’,
Int. J. Control, Vol.31(5) (1980) 1007–1009.
"""
function lpsminreal(A::AbstractMatrix, E::AbstractMatrix, B::AbstractMatrix, F::AbstractMatrix, C::AbstractMatrix, G::AbstractMatrix,
                   D::AbstractMatrix, H::AbstractMatrix; 
                   atol1::Real = zero(real(eltype(A))), atol2::Real = zero(real(eltype(A))), 
                   rtol::Real =  (size(A,1)+2)*eps(real(float(one(real(eltype(A))))))*iszero(min(atol1,atol2)), 
                   fast::Bool = true, contr::Bool = true, obs::Bool = true, noseig::Bool = true)

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

    A1 = copy_oftype(A,T)   
    B1 = copy_oftype(B,T)
    C1 = copy_oftype(C,T)
    D1 = copy_oftype(D,T)  
    E1 = copy_oftype(E,T)
    F1 = copy_oftype(F,T)
    G1 = copy_oftype(G,T)
    H1 = copy_oftype(H,T)  

    (n == 0 || m == 0 || p == 0) && (return A1, E1, B1, F1, C1, G1, D1, H1, I, I, 0, 0)
    # save system matrices
    Ar = copy(A1)
    Br = copy(B1)
    Cr = copy(C1)
    Dr = copy(D1)
    Er = copy(E1)
    Fr = copy(F1)
    Gr = copy(G1)
    Hr = copy(H1)
    ir = 1:n
    nr = n
    if contr  
      _, W, nr = sklf_right!(Ar, Er, Br, Fr, Cr, Gr, Dr, Hr; fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol, withQ = false, withZ = false) 
      nuc = n-nr
      if nuc > 0
         ir = 1:nr
         # save intermediary results
         A1 = copy(Ar[ir,ir])
         E1 = copy(Er[ir,ir])
         B1 = copy(Br[ir,:])
         F1 = copy(Fr[ir,:])
         C1 = copy(Cr[:,ir])
         G1 = copy(Gr[:,ir])
         D1 = copy(Dr)
         H1 = copy(Hr)
         W = UpperTriangular(W)
      else
         # restore original matrices 
         Ar = copy(A1)
         Br = copy(B1)
         Cr = copy(C1)
         Dr = copy(D1)
         Er = copy(E1)
         Fr = copy(F1)
         Gr = copy(G1)
         Hr = copy(H1)
         W = I
      end
   else
      W = I
      nuc = 0
   end
   if obs 
      V, _, no = sklf_left!(view(Ar,ir,ir), view(Er,ir,ir), view(Cr,:,ir), view(Gr,:,ir), view(Br,ir,:), view(Fr,ir,:), Dr, Hr; 
                                         fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol, withQ = false, withZ = false) 
       nuo = nr-no
       if nuo > 0
          ir = ir[end-no+1:end]
          V = UpperTriangular(V)
       else
          # restore saved matrices
          Ar[ir,ir] = A1
          Er[ir,ir] = E1
          Br[ir,:] = B1
          Fr[ir,:] = F1
          Cr[:,ir] = C1
          Gr[:,ir] = G1
          Dr = copy(D1)
          Hr = copy(H1)
          V = I
         end
   else
      V = I
      nuo = 0
   end
   return Ar[ir,ir], Er[ir,ir], Br[ir,:], Fr[ir,:], Cr[:,ir], Gr[:,ir], Dr, Hr, V, W, nuc, nuo
end
