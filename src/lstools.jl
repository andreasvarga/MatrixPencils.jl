function rcond(A::DenseMatrix{T},tola::Real = 0) where {T <: Union{BlasInt,BlasFloat}}
   T1 = promote_type(T,Float64)
   nrmA = opnorm(A,1)
   nrmA <= tola && return zero(real(T1))
   istriu(A) ? (return LinearAlgebra.LAPACK.trcon!('1','U','N',copy_oftype(A,T1))) : 
       (return LinearAlgebra.LAPACK.gecon!('1', LinearAlgebra.LAPACK.getrf!(copy_oftype(A,T1))[1],real(T1)(nrmA)) ) 
end
rcond(A::DenseMatrix{T},tola::Real = 0) where {T} = 1/cond(A,1)
"""
    lseval(A, E, B, C, D, val; atol1, atol2, rtol, fast = true) -> Gval

Evaluate `Gval`, the value of the rational matrix `G(λ) = C*inv(λE-A)*B+D` for `λ = val`. 
The computed `Gval` has infinite entries if `val` is a pole (finite or infinite) of `G(λ)`.
If `val` is finite and `val*E-A` is singular or if `val = Inf` and `E` is singular, 
then the entries of `Gval` are evaluated separately for minimal realizations of each input-output channel.

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of matrices `A`, `B`, `C`, `D`, the absolute tolerance for the nonzero elements of `E`,  
and the relative tolerance for the nonzero elements of `A`, `B`, `C`, `D` and `E`. 

The computation of minimal realizations of individual input-output channels relies on pencil manipulation algorithms,
which employ rank determinations based on either the use of 
rank revealing QR-decomposition with column pivoting, if `fast = true`, or the SVD-decomposition.
The rank decision based on the SVD-decomposition is generally more reliable, but the involved computational effort is higher.
"""
function lseval(A::AbstractMatrix, E::Union{AbstractMatrix,UniformScaling{Bool}}, 
                B::AbstractVecOrMat, C::AbstractMatrix, D::AbstractVecOrMat, val::Number;  
                atol1::Real = zero(real(eltype(A))), atol2::Real = zero(real(eltype(A))), 
                rtol::Real =  (size(A,1)+1)*eps(float(real(eltype(A))))*iszero(max(atol1,atol2)), fast::Bool = true)

   T = promote_type(eltype(A), eltype(B), eltype(C), eltype(D),typeof(val))
   E == I || (T = promote_type(T,eltype(E)))
   T <: BlasFloat || (T = promote_type(Float64,T))  
   # check dimensions
   n = LinearAlgebra.checksquare(A)
   if typeof(E) <: AbstractMatrix
      n == LinearAlgebra.checksquare(E) || error("A and E must have the same size")
   end
   n == 0 && (return T.(D))

   n1, m = typeof(B) <: AbstractVector ? (length(B),1) : size(B)
   n1 == n ||  error("B must have the same row size as A")
   p, n1 = size(C)
   n1 == n ||  error("C must have the same column size as A")
   p1, m1 = typeof(D) <: AbstractVector ? (length(D),1) : size(D)
   m1 == m ||  error("D must have the same column size as B")
   p1 == p ||  error("D must have the same row size as C")


   toleps = (size(A,1)+1)*eps(real(T))
   if abs(val) < Inf 
      LUF = lu!(T(val)*E-A;check = false)
      tol = max(atol1,atol2)
      if rcond(LUF.U, tol) < toleps
         G = zeros(T,p,m)
         for i = 1:p
            At1, Et1, Bt1, Ct1, Dt1, = lsminreal2(A, E, B, view(C,i:i,:), view(D,i:i,:); infinite = false, noseig = false, contr = false, fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol) 
            for j = 1:m
                At11, Et11, Bt11, Ct11, Dt11, = lsminreal2(At1, Et1, view(Bt1,:,j:j), Ct1, view(Dt1,:,j:j); infinite = false, noseig = false, obs = false, fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol) 
                LUF = lu!(T(val)*Et11-At11;check = false)
                rcond(LUF.U,tol) < toleps ? G[i,j] = T(Inf) : G[i,j] = (Ct11*ldiv!(LUF,copy_oftype(Bt11,T)) + Dt11)[1,1]
            end
         end
         return G
      else
         return C*ldiv!(LUF,copy_oftype(B,T)) + D
      end
   else
      (E == I || rcond(E,atol2) > toleps) && (return T.(D))
      At, Et, Bt, Ct, Dt, = lsminreal2(A, E, B, C, D; finite = false, fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol) 
      if size(At,1) > 0 && rcond(Et,atol2) < toleps 
         G = zeros(T,p,m)
         for i = 1:p
            At1, Et1, Bt1, Ct1, Dt1, = lsminreal2(At, Et, Bt, view(Ct,i:i,:), view(Dt,i:i,:); finite = false, contr = false, noseig = false, fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol) 
            for j = 1:m
                At11, Et11, Bt11, Ct11, Dt11, = lsminreal2(At1, Et1, view(Bt1,:,j:j), Ct1, view(Dt1,:,j:j); finite = false, obs = false, noseig = true, fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol) 
                size(At11,1) > 0 && rcond(Et11,atol2) < toleps ? G[i,j] = T(Inf) : G[i,j] = Dt11[1,1]
            end
         end
         return G
      else
         return Dt
      end
   end
end
"""
     lps2ls(A, E, B, F, C, G, D, H; compacted = false, atol1 = 0, atol2 = 0, atol3 = 0, rtol = min(atol1,atol2,atol3)>0 ? 0 : n*ϵ) -> (Ad,Ed,Bd,Cd,Dd)

Construct an input-output equivalent descriptor system linearizations `(Ad-λdE,Bd,Cd,Dd)` to a pencil based linearization 
`(A-λE,B-λF,C-λG,D-λH)` satisfying 

                -1                        -1
     Cd*(λEd-Ad)  *Bd + Dd = (C-λG)*(λE-A)  *(B-λF) + D-λH .

If `compacted = true`, a compacted linearization is determined by exploiting possible rank defficiencies of the
matrices `F`, `G`, and `H`.  

The keyword arguments `atol1`, `atol2`, `atol3`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of `F`, the absolute tolerance for the nonzero elements of `G`, 
the absolute tolerance for the nonzero elements of `H` and the relative tolerance 
for the nonzero elements of `F`, `G` and `H`. The default relative tolerance is `n*ϵ`, where `n` is the size of 
 `A`, and `ϵ` is the machine epsilon of the element type of `A`. 
"""
function lps2ls(A::AbstractMatrix, E::Union{AbstractMatrix,UniformScaling}, B::AbstractVecOrMat, F::Union{AbstractVecOrMat,Missing},
                C::AbstractMatrix, G::Union{AbstractMatrix,Missing}, D::AbstractVecOrMat, H::Union{AbstractVecOrMat,Missing}; 
                compacted::Bool = false, atol1::Real = zero(real(eltype(A))), atol2::Real = zero(real(eltype(A))), 
                atol3::Real = zero(real(eltype(A))), rtol::Real = (min(size(A)...)*eps(real(float(one(eltype(A))))))*iszero(min(atol1,atol2,atol3))) 

   T = promote_type(eltype(A), E == I ? Bool : eltype(E), eltype(B), eltype(C), eltype(D))
   T = promote_type(T, ismissing(F) ? T : eltype(F), ismissing(G) ? T : eltype(G), ismissing(H) ? T :  eltype(H))
   ismissing(F) && ismissing(G) && ismissing(H) && (return A, E, B, C, D)

   n = size(A,1)
   p, m = typeof(D) <: AbstractVector ? (length(D),1) : size(D)
   nm = n+m
   D1 = copy_oftype(D,T)

   (ismissing(F) || iszero(F)) ? mF = 0 : mF = m
   (ismissing(G) || iszero(G)) ? pG = 0 : pG = p
   if compacted
       if ismissing(G) || iszero(G) 
           pG = 0 
           C11 = zeros(T,p,pG)
           E12 = zeros(T,pG,n)
       else
          S = svd(G,full=true)
          pG = count(S.S .> max(atol2,rtol*S.S[1]))
          if pG == p
             C11 = -I
             E12 = view(G,:,:)
          else
             C11 = -S.U[:,1:pG]
             E12 = Diagonal(view(S.S,1:pG))*view(S.Vt,1:pG,:)
          end
       end
       if ismissing(F) || iszero(F) 
           mF = 0 
           B22 = zeros(T,mF,m)
           E23 = zeros(T,n,mF)
       else
          # svd is not working on a vector
          typeof(F) <: AbstractVector ? S = svd(reshape(F,n,1),full=true) : S = svd(F,full=true)
          mF = count(S.S .> max(atol1,rtol*S.S[1]))
          if mF == m
             B22 = -I
             E23 = view(F,:,:)
          else
             B22 = -S.Vt[1:mF,:]
             E23 = view(S.U,:,1:mF)*Diagonal(view(S.S,1:mF))
          end
       end
       A1 = [I zeros(T,pG,n+mF); zeros(T,n,pG) A zeros(T,n,mF); zeros(T,mF,pG+n) I]
       E1 = [zeros(T,pG,pG) E12 zeros(T,pG,mF); zeros(T,n,pG) E E23; zeros(T,mF,pG+n+mF)]
       B1 = [zeros(T,pG,m); B; B22]
       C1 = [C11 C zeros(T,p,mF)]
   else
       A1 = [I zeros(T,pG,n+mF); zeros(T,n,pG) A zeros(T,n,mF); zeros(T,mF,pG+n) I]
       E1 = [zeros(T,pG,pG) ismissing(G) ? zeros(T,pG,n) : G[1:pG,:] zeros(T,pG,mF); zeros(T,n,pG) E ismissing(F) ? zeros(T,n,mF) : F[:,1:mF]; zeros(T,mF,pG+n+mF)]
       B1 = [zeros(T,pG,m); B; mF > 0 ? -I : zeros(T,mF,m)]
       C1 = [pG > 0 ? -I : zeros(T,p,pG) C zeros(T,p,mF)]
   end
   (ismissing(H) || iszero(H)) &&  (return A1, E1, B1, C1, D1)

   finished = false
   if compacted
      typeof(H) <: AbstractVector ? S = svd(reshape(H,p,1),full=true) : S = svd(H,full=true)
      r = count(S.S .> max(atol3,rtol*S.S[1]))
      if r < min(p,m)
         hs2 = Diagonal(sqrt.(view(S.S,1:r)))
         A2 = Matrix{T}(I,2*r,2*r)
         E2 = [zeros(T,r,r) I; zeros(T,r,2*r)]
         B2 = [zeros(T,r,m); hs2*view(S.Vt,1:r,:)]
         C2 = [view(S.U,:,1:r)*hs2 zeros(T,p,r)]
         finished = true
      end
   end
   if !finished
      if m <= p
         A2 = Matrix{T}(I,2*m,2*m)
         E2 = [zeros(T,m,m) I; zeros(T,m,2*m)]
         B2 = [zeros(T,m,m); I]
         C2 = [H zeros(T,p,m)]
      else
         A2 = Matrix{T}(I,2*p,2*p)
         E2 = [zeros(T,p,p) I; zeros(T,p,2*p)]
         B2 = [zeros(T,p,m); -H]
         C2 = [-I zeros(T,p,p)]
      end 
   end
   return blockdiag(A1,A2), blockdiag(E1,E2), [B1; B2], [C1 C2], D1
end

# function blockdiag(mats::AbstractMatrix{T}...) where T
#     rows = Int[size(m, 1) for m in mats]
#     cols = Int[size(m, 2) for m in mats]
#     res = zeros(T, sum(rows), sum(cols))
#     m = 1
#     n = 1
#     for ind=1:length(mats)
#         mat = mats[ind]
#         i = rows[ind]
#         j = cols[ind]
#         res[m:m + i - 1, n:n + j - 1] = mat
#         m += i
#         n += j
#     end
#     return res
# end
# taken from ControlSystems.jl
@static if VERSION >= v"1.8.0-beta1"
   blockdiag(mats...) = cat(mats..., dims=Val((1,2)))
   blockdiag(mats::Union{<:Tuple, <:Base.Generator}) = cat(mats..., dims=Val((1,2)))
else
   blockdiag(mats...) = cat(mats..., dims=(1,2))
   blockdiag(mats::Union{<:Tuple, <:Base.Generator}) = cat(mats..., dims=(1,2))
end

"""
    lpseval(A, E, B, F, C, G, D, H, val; atol1, atol2, rtol, fast = true) -> Gval

Evaluate `Gval`, the value of the rational matrix `G(λ) = (C-λG)*inv(λE-A)*(B-λF)+D-λH` for `λ = val`. 
The computed `Gval` has infinite entries if `val` is a pole (finite or infinite) of `G(λ)`.
If `val` is finite and `val*E-A` is singular or `val` is infinite and `E` is singular, 
then the entries of `Gval` are evaluated separately for each input-output chanel employing descriptor system based 
minimal realizations.

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of matrices `A`, `B`, `C`, `D`, the absolute tolerance for the nonzero elements of `E`, `F`, `G`, `H`,  
and the relative tolerance for the nonzero elements of `A`, `B`, `C`, `D`, `E`, `F`, `G`, and `H`. 

The computation of minimal realizations of individual input-output chanels relies on pencil manipulation algorithms,
which employ rank determinations based on either the use of 
rank revealing QR-decomposition with column pivoting, if `fast = true`, or the SVD-decomposition.
The rank decision based on the SVD-decomposition is generally more reliable, but the involved computational effort is higher.
"""
function lpseval(A::AbstractMatrix, E::Union{AbstractMatrix,UniformScaling{Bool}}, 
                 B::AbstractVecOrMat, F::AbstractVecOrMat, C::AbstractMatrix, G::AbstractMatrix, 
                 D::AbstractVecOrMat, H::AbstractVecOrMat, val::Number; atol1::Real = zero(real(eltype(A))), atol2::Real = zero(real(eltype(A))), 
                 rtol::Real =  (size(A,1)+1)*eps(float(real(eltype(A))))*iszero(max(atol1,atol2)), fast::Bool = true)

   # check dimensions
   n = LinearAlgebra.checksquare(A)
   if typeof(E) <: AbstractMatrix
      n == LinearAlgebra.checksquare(E) || error("A and E must have the same size")
   end
   
   n1, m = typeof(B) <: AbstractVector ? (length(B),1) : size(B)
   n1 == n ||  error("B must have the same row size as A")
   n1, m1 = typeof(F) <: AbstractVector ? (length(F),1) : size(F)
   (n1, m1) == (n, m) ||  error("B and F must have the same size")
   p, n1 = size(C)
   n1 == n ||  error("C must have the same column size as A")
   (p, n) == size(G) ||  error("C and G must have the same size")
   p1, m1 = typeof(D) <: AbstractVector ? (length(D),1) : size(D)
   m1 == m ||  error("D must have the same column size as B")
   p1 == p ||  error("D must have the same row size as C")
   p1, m1 = typeof(H) <: AbstractVector ? (length(H),1) : size(H)
   (p, m) == (p1, m1) || error("D and H must have the same size")

   T = promote_type(eltype(A), eltype(B), eltype(C), eltype(D), eltype(F), eltype(G), eltype(H), typeof(val))
   E == I || (T = promote_type(T,eltype(E)))
   T <: BlasFloat || (T = promote_type(Float64,T))        

   toleps = (size(A,1)+1)*eps(real(T))
   if abs(val) < Inf 
      LUF = lu!(T(val)*E-A;check = false)
      tol = max(atol1,atol2)
      if rcond(LUF.U, tol) < toleps
         At, Et, Bt, Ct, Dt = lps2ls(A, E, B, F, C, G, D, H; compacted = true, atol1 = atol2, atol2 = atol2, atol3 = atol2, rtol = rtol)
         p, m = typeof(Dt) <: AbstractVector ? (length(D),1) : size(Dt)
         G = zeros(T,p,m)
         for i = 1:p
            At1, Et1, Bt1, Ct1, Dt1, = lsminreal2(At, Et, Bt, view(Ct,i:i,:), view(Dt,i:i,:); infinite = false, noseig = false, contr = false, fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol) 
            for j = 1:m
                At11, Et11, Bt11, Ct11, Dt11, = lsminreal2(At1, Et1, view(Bt1,:,j:j), Ct1, view(Dt1,:,j:j); infinite = false, noseig = false, obs = false, fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol) 
                LUF = lu!(T(val)*Et11-At11;check = false)
                rcond(LUF.U,tol) < toleps ? G[i,j] = T(Inf) : G[i,j] = (Ct11*ldiv!(LUF,copy_oftype(Bt11,T)) + Dt11)[1,1]
            end
         end
         return G
      else
         return (C-val*G)*ldiv!(LUF,copy_oftype(B-val*F,T)) + D-val*H
      end
   else
      return lseval(lps2ls(A, E, B, F, C, G, D, H; compacted = true, atol1 = atol2, atol2 = atol2, atol3 = atol2, rtol = rtol)...,val; 
                    atol1 = atol1, atol2 = atol2, rtol = rtol, fast = fast)
   end
end
"""
     lsequal(A1, E1, B1, C1, D1, A2, E2, B2, C2, D2; 
             fastrank = true, atol1 = 0, atol2 = 0, rtol = min(atol1,atol2)>0 ? 0 : n*ϵ) -> flag::Bool

Check if two descriptor system linearizations `(A1-λE1,B1,C1,D1)` and `(A2-λE2,B2,C2,D2)` satisfy the equivalence condition  
   
                -1                      -1
     C1*(λE1-A1)  *B1 + D1 = C2*(λE2-A2)  *B2 + D2

The ckeck is performed by computing the normal rank `n` of the structured matrix pencil `M - λN`

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

[1] A. Varga, On checking null rank conditions of rational matrices, [arXiv:1812.11396](https://arxiv.org/abs/1812.11396), 2018.
"""
function lsequal(A1::AbstractMatrix, E1::Union{AbstractMatrix,UniformScaling{Bool}}, 
   B1::AbstractVecOrMat, C1::AbstractMatrix, D1::AbstractVecOrMat,
   A2::AbstractMatrix, E2::Union{AbstractMatrix,UniformScaling{Bool}}, 
   B2::AbstractVecOrMat, C2::AbstractMatrix, D2::AbstractVecOrMat; 
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

Check if two pencil based linearizations `(A1-λE1,B1-λF1,C1-λG1,D1-λH1)` and `(A2-λE2,B2-λF2,C2-λG2,D2-λH2)` 
satisfy the equivalence condition  
   
                      -1                                      -1
     (C1-λG1)*(λE1-A1)  *(B1-λF1) + D1-λH1 = (C2-λG2)*(λE2-A2)  *(B2-λF2) + D2-λH2

The ckeck is performed by computing the normal rank `n` of the structured matrix pencil `M - λN`

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

[1] A. Varga, On checking null rank conditions of rational matrices, [arXiv:1812.11396](https://arxiv.org/abs/1812.11396), 2018.
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
     C*(λE-A)  *B + D = Cr*(λEr-Ar)  *Br + Dr
     
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
                    B::AbstractVecOrMat, C::AbstractMatrix, D::AbstractVecOrMat; 
                    atol1::Real = zero(real(eltype(A))), atol2::Real = zero(real(eltype(A))), 
                    rtol::Real =  (size(A,1)+1)*eps(real(float(one(real(eltype(A))))))*iszero(max(atol1,atol2)), 
                    fast::Bool = true, finite::Bool = true, infinite::Bool = true, 
                    contr::Bool = true, obs::Bool = true, noseig::Bool = true)

   emat = (typeof(E) <: AbstractMatrix)
   eident = !emat || isequal(E,I) 
   n = LinearAlgebra.checksquare(A)
   emat && (n,n) != size(E) && throw(DimensionMismatch("A and E must have the same dimensions"))
   p, m = typeof(D) <: AbstractVector ? (length(D),1) : size(D)
   n1, m1 = typeof(B) <: AbstractVector ? (length(B),1) : size(B)
   (n,m) == (n1, m1) || throw(DimensionMismatch("A, B and D must have compatible dimensions"))
   (p,n) == size(C) || throw(DimensionMismatch("A, C and D must have compatible dimensions"))
   T = promote_type(eltype(A), eltype(B), eltype(C), eltype(D))
   eident || (T = promote_type(T,eltype(E)))
   T <: BlasFloat || (T = promote_type(Float64,T))        

   A1 = copy_oftype(A,T)   
   eident ? E1 = copy(E) : E1 = copy_oftype(E,T)
   B1 = copy_oftype(B,T)
   C1 = copy_oftype(C,T)
   D1 = copy_oftype(D,T)  

   n == 0 && (return A1, E1, B1, C1, D1, 0, 0, 0)

   if eident
      A1, B1, C1, nuc, nuo  = lsminreal(A1, B1, C1, contr = contr, obs = obs, fast = fast, atol = atol1, rtol = rtol)
      return A1, emat ? Matrix{T}(I,size(A1)...) : I, B1, C1, D1, nuc, nuo, 0
   else
      # save system matrices
      Ar = copy(A1)
      Br = copy(B1)
      Cr = copy(C1)
      Dr = copy(D1)
      Er = copy(E1)
      ir = 1:n
      if finite
         if contr  
            m == 0 &&  (ir = 1:0; return Ar[ir,ir], Er[ir,ir], Br[ir,:], Cr[:,ir], Dr, n, 0, 0)
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
            nr = n
         end
         if obs 
            p == 0 &&  (ir = 1:0; return Ar[ir,ir], Er[ir,ir], Br[ir,:], Cr[:,ir], Dr, nfuc, nr, 0)
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
            m == 0 &&  (ir = 1:0; return Ar[ir,ir], Er[ir,ir], Br[ir,:], Cr[:,ir], Dr, n, 0, 0)
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
            p == 0 &&  (ir = 1:0; return Ar[ir,ir], Er[ir,ir], Br[ir,:], Cr[:,ir], Dr, niuc, nr, 0)
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
    lsminreal(A, B, C; fast = true, atol = 0, rtol, contr = true, obs = true, noseig = true) 
              -> (Ar, Br, Cr, nuc, nuo)

Reduce the linearization `(A-λI,B,C,0)` of a strictly proper rational matrix to a reduced form `(Ar-λI,Br,Cr,0)` such that

             -1                 -1
     C*(λI-A)  *B  = Cr*(λI-Ar)  *Br 
     
with the least possible order `nr` of `Ar` if `contr = true` and `obs = true`. 
Such a realization is called `minimal` and satisfies:

     (1) rank[Br Ar-λI] = nr for all λ (controllability);

     (2) rank[Ar-λI; Cr] = nr for all λ (observability).

The achieved dimensional reductions to fulfill conditions (1) and (2) are returned in `nuc` and `nuo`, respectively. 

Some reduction steps can be skipped by appropriately selecting the keyword arguments `contr` and `obs`. 

If `contr = false`, then the controllability condition (1) is not enforced. 

If `obs = false`, then observability condition (2) is not enforced.

To enforce conditions (1)-(2), orthogonal similarity transformations are performed on 
the matrices of the original linearization `(A-λI,B,C,0)` to obtain a minimal linearization using
structured pencil reduction algorithms, as the fast versions of the reduction techniques of the 
full row rank pencil [B A-λI] and full column rank pencil [A-λI;C] proposed in [1]. 

The underlying pencil manipulation algorithms employ rank determinations based on either the use of 
rank revealing QR-decomposition with column pivoting, if `fast = true`, or the SVD-decomposition.
The rank decision based on the SVD-decomposition is generally more reliable, but the involved computational effort is higher.

The keyword arguments `atol` and `rtol`, specify, respectively, the absolute and relative tolerances for the 
nonzero elements of matrices `A`, `B`, `C`.  

[1] P. Van Dooreen, The generalized eigenstructure problem in linear system theory, 
IEEE Transactions on Automatic Control, vol. AC-26, pp. 111-129, 1981.
"""
function lsminreal(A::AbstractMatrix, B::AbstractVecOrMat, C::AbstractMatrix; 
                   atol::Real = zero(real(eltype(A))), 
                   rtol::Real =  (size(A,1)+1)*eps(real(float(one(real(eltype(A))))))*iszero(atol), 
                   fast::Bool = true, contr::Bool = true, obs::Bool = true)
   n = LinearAlgebra.checksquare(A)
   n1, m = typeof(B) <: AbstractVector ? (length(B),1) : size(B)
   n == n1 || throw(DimensionMismatch("A and B must have the same number of rows"))
   (p,n1) = size(C)
   n1 == n || throw(DimensionMismatch("A and C must have the same number of columns"))
   T = promote_type(eltype(A), eltype(B), eltype(C))
   T <: BlasFloat || (T = promote_type(Float64,T))        

   A1 = copy_oftype(A,T)   
   B1 = copy_oftype(B,T)
   C1 = copy_oftype(C,T)

   n == 0 && (return A1, B1, C1, 0, 0)

   # save system matrices
   Ar = copy(A1)
   Br = copy(B1)
   Cr = copy(C1)
   ir = 1:n
   if contr
      m == 0 &&  (ir = 1:0; return Ar[ir,ir], Br[ir,:], Cr[:,ir], n, 0)
      _, _, nr, nuc = sklf_right!(Ar, Br, Cr; fast = fast, atol1 = atol, atol2 = atol, rtol = rtol, withQ = false)
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
      nr = n
   end
   if obs
      p == 0 &&  (ir = 1:0; return Ar[ir,ir], Br[ir,:], Cr[:,ir], nuc, nr)
      _, _, no, nuo = sklf_left!(view(Ar,ir,ir), view(Cr,:,ir), view(Br,ir,:); fast = fast, atol1 = atol, atol2 = atol, rtol = rtol, withQ = false) 
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
   return Ar[ir,ir], Br[ir,:], Cr[:,ir], nuc, nuo
end
"""
    lsminreal(A, E, B, C, D; fast = true, atol1 = 0, atol2, rtol, contr = true, obs = true, noseig = true) 
              -> (Ar, Er, Br, Cr, Dr, nuc, nuo, nse)

Reduce the linearization `(A-λE,B,C,D)` of a rational matrix to a reduced form `(Ar-λEr,Br,Cr,Dr)` such that

             -1                    -1
     C*(λE-A)  *B + D = Cr*(λEr-Ar)  *Br + Dr
     
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
                   B::AbstractVecOrMat, C::AbstractMatrix, D::AbstractVecOrMat; 
                   atol1::Real = zero(real(eltype(A))), atol2::Real = zero(real(eltype(A))), 
                   rtol::Real =  (size(A,1)+1)*eps(real(float(one(real(eltype(A))))))*iszero(max(atol1,atol2)), 
                   fast::Bool = true, contr::Bool = true, obs::Bool = true, noseig::Bool = true)

   emat = (typeof(E) <: AbstractMatrix)
   eident = !emat || isequal(E,I) 
   n = LinearAlgebra.checksquare(A)
   emat && (n,n) != size(E) && throw(DimensionMismatch("A and E must have the same dimensions"))
   p, m = typeof(D) <: AbstractVector ? (length(D),1) : size(D)
   n1, m1 = typeof(B) <: AbstractVector ? (length(B),1) : size(B)
   (n,m) == (n1, m1) || throw(DimensionMismatch("A, B and D must have compatible dimensions"))
   (p,n) == size(C) || throw(DimensionMismatch("A, C and D must have compatible dimensions"))
   T = promote_type(eltype(A), eltype(B), eltype(C), eltype(D))
   eident || (T = promote_type(T,eltype(E)))
   T <: BlasFloat || (T = promote_type(Float64,T))        

   A1 = copy_oftype(A,T)   
   eident ? E1 = copy(E) : E1 = copy_oftype(E,T)
   B1 = copy_oftype(B,T)
   C1 = copy_oftype(C,T)
   D1 = copy_oftype(D,T)  

   n == 0 && (return A1, E1, B1, C1, D1, 0, 0, 0)

   if eident
      A1, B1, C1, nuc, nuo  = lsminreal(A1, B1, C1, contr = contr, obs = obs, fast = fast, atol = atol1, rtol = rtol)
      return A1, emat ? Matrix{T}(I,size(A1)...) : I, B1, C1, D1, nuc, nuo, 0
   else
      # save system matrices
      Ar = copy(A1)
      Br = copy(B1)
      Cr = copy(C1)
      Dr = copy(D1)
      Er = copy(E1)
      ir = 1:n
      if contr  
         m == 0 &&  (ir = 1:0; return Ar[ir,ir], Er[ir,ir], Br[ir,:], Cr[:,ir], Dr, n, 0, 0)
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
         nr = n
      end
      if obs 
         p == 0 &&  (ir = 1:0; return Ar[ir,ir], Er[ir,ir], Br[ir,:], Cr[:,ir], Dr, nuc, nr, 0)
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
     V'*((C-λG)*(λE-A)  *(B-λF) + D-λH)*W = (Cr-λGr)*(λEr-Ar)  *(Br-λFr) + Dr-λHr
     
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
Springer-Verlag. [arXiv:1903.05016](https://arxiv.org/pdf/1903.05016.pdf)

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

    n == 0 && (return A1, E1, B1, F1, C1, G1, D1, H1, I, I, 0, 0)
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
      m == 0 && (ir = 1:0; return A1[ir,ir], E1[ir,ir], B1[ir,:], F1[ir,:], C1[:,ir], G1[:,ir], D1, H1, I, I, n, 0)
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
      p == 0 && (ir = 1:0; return A1[ir,ir], E1[ir,ir], B1[ir,:], F1[ir,:], C1[:,ir], G1[:,ir], D1, H1, I, I, nuc, nr)
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
"""
     lsbalance!(A, B, C; withB = true, withC = true, maxred = 16) -> D

Reduce the 1-norm of a system matrix 

             S =  ( A  B )
                  ( C  0 )

corresponding to a standard system triple `(A,B,C)` by balancing. 
This involves a diagonal similarity transformation `inv(D)*A*D` 
to make the rows and columns of 

                  diag(inv(D),I) * S * diag(D,I)
     
as close in norm as possible.
     
The balancing can be optionally performed on the following 
particular system matrices:   

        S = A, if withB = false and withC = false ,
        S = ( A  B ), if withC = false,     or    
        S = ( A ), if withB = false .
            ( C )      

The keyword argument `maxred = s` specifies the maximum allowed reduction in the 1-norm of
`S` (in an iteration) if zero rows or columns are present. 
`s` must be a positive power of `2`, otherwise  `s` is rounded to the nearest power of 2.   

_Note:_ This function is a translation of the SLICOT routine `TB01ID.f`, which 
represents an extensiom of the LAPACK family `*GEBAL.f` and also includes  
the fix proposed in [1]. 

[1]  R.James, J.Langou and B.Lowery. "On matrix balancing and eigenvector computation."
     ArXiv, 2014, http://arxiv.org/abs/1401.5766
"""
function lsbalance!(A::AbstractMatrix{T1}, B::AbstractVecOrMat{T1}, C::AbstractMatrix{T1}; 
                   withB = true, withC = true, maxred = 16) where {T1}
    n = LinearAlgebra.checksquare(A)
    n1, m = size(B,1), size(B,2)
    n1 == n || throw(DimensionMismatch("A and B must have the same number of rows"))
    n == size(C,2) || throw(DimensionMismatch("A and C must have the same number of columns"))
    T = real(T1)
    ZERO = zero(T)
    ONE = one(T)
    SCLFAC = T(2.)
    FACTOR = T(0.95); 
    maxred > ONE || throw(ArgumentError("maxred must be greater than 1, got maxred = $maxred"))
    MAXR = T(maxred)
    MAXR = T(2.) ^round(Int,log2(MAXR))
    # Compute the 1-norm of the required part of matrix S and exit if it is zero.
    SNORM = ZERO
    for j = 1:n
        CO = sum(abs,view(A,:,j))
        withC && (CO += sum(abs,view(C,:,j)))
        SNORM = max( SNORM, CO )
    end
    if withB
       for j = 1:m 
           SNORM = max( SNORM, sum(abs,view(B,:,j)) )
       end
    end
    D = fill(ONE,n)
    SNORM == ZERO && (return D)

    SFMIN1 =  T <: BlasFloat ? safemin(T) / eps(T) : safemin(Float64) / eps(Float64)
    SFMAX1 = ONE / SFMIN1
    SFMIN2 = SFMIN1*SCLFAC
    SFMAX2 = ONE / SFMIN2

    SRED = maxred
    SRED <= ZERO && (SRED = MAXR)

    MAXNRM = max( SNORM/SRED, SFMIN1 )

    # Balance the system matrix.

    # Iterative loop for norm reduction.
    NOCONV = true
    while NOCONV
        NOCONV = false
        for i = 1:n
            Aci = view(A,:,i)
            Ari = view(A,i,:)
            Ci = view(C,:,i)
            Bi = view(B,i,:)

            CO = norm(Aci)
            RO = norm(Ari)
            CA = abs(argmax(abs,Aci))
            RA = abs(argmax(abs,Ari))

            if withC
               CO = hypot(CO, norm(Ci))
               CA = max(CA,abs(argmax(abs,Ci)))
            end
            if withB
                RO = hypot(RO, norm(Bi))
                RA = max( RA, abs(argmax(abs,Bi)))
            end
            #  Special case of zero CO and/or RO.
            CO == ZERO && RO == ZERO && continue
            if CO == ZERO 
               RO <= MAXNRM && continue
               CO = MAXNRM
            end
            if RO == ZERO
               CO <= MAXNRM && continue
               RO = MAXNRM
            end

            #  Guard against zero CO or RO due to underflow.
            G = RO / SCLFAC
            F = ONE
            S = CO + RO
            while ( CO < G && max( F, CO, CA ) < SFMAX2 && min( RO, G, RA ) > SFMIN2 ) 
                F  *= SCLFAC
                CO *= SCLFAC
                CA *= SCLFAC
                G  /= SCLFAC
                RO /= SCLFAC
                RA /= SCLFAC   
            end
            G = CO / SCLFAC

            while ( G >= RO && max( RO, RA ) < SFMAX2 && min( F, CO, G, CA ) > SFMIN2 )
                F  /= SCLFAC
                CO /= SCLFAC
                CA /= SCLFAC
                G  /= SCLFAC
                RO *= SCLFAC
                RA *= SCLFAC
            end

            # Now balance.
            CO+RO >= FACTOR*S && continue
            if F < ONE && D[i] < ONE 
               F*D[i] <= SFMIN1 && continue
            end
            if F > ONE && D[i] > ONE 
               D[i] >= SFMAX1 / F  && continue
            end
            G = ONE / F
            D[i] *= F
            NOCONV = true
       
            lmul!(G,Ari)
            rmul!(Aci,F)
            lmul!(G,Bi)
            rmul!(Ci,F)
        end
    end
    return Diagonal(D)
end   
lsbalance!(A::AbstractMatrix{T1}; maxred = 16) where {T1} = lsbalance!(A,zeros(T1,size(A,1),0),zeros(T1,0,size(A,2)); maxred, withB=false, withC=false)  
"""
    qs = lsbalqual(A, B, C; SysMat = false) 

Compute the 1-norm based scaling quality of a standard system triple `(A,B,C)`.

If `SysMat = false`, the resulting `qs` is computed as 

        qs = max(qS(A),qS(B),qS(C)) ,

where `qS(⋅)` is the scaling quality measure defined in Definition 5.5 of [1] for 
nonnegative matrices. This definition has been extended to also cover matrices with
zero rows or columns.  

If `SysMat = true`, the resulting `qs` is computed as 

        qs = qS(S) ,

where `S` is the system matrix defined as        

             S =  ( A  B )
                  ( C  0 )

A large value of `qs` indicates a possible poorly scaled state-space model.   

[1] F.M.Dopico, M.C.Quintana and P. van Dooren, 
    "Diagonal scalings for the eigenstructure of arbitrary pencils", SIMAX, 43:1213-1237, 2022. 
"""
function lsbalqual(A::AbstractMatrix{T}, B::AbstractVecOrMat{T}, C::AbstractMatrix{T}; SysMat = false) where {T}
    return SysMat ? qS1([A B; C zeros(T,size(C,1),size(B,2))]) : max(qS1(A),qS1(B),qS1(C))
end 
"""
    qs = lsbalqual(A, E, B, C; SysMat = false) 

Compute the 1-norm based scaling quality of a descriptor system triple `(A-λE,B,C)`.

If `SysMat = false`, the resulting `qs` is computed as 

        qs = max(qS(A),qS(E),qS(B),qS(C)) ,

where `qS(⋅)` is the scaling quality measure defined in Definition 5.5 of [1] for 
nonnegative matrices. This definition has been extended to also cover matrices with
zero rows or columns.  

If `SysMat = true`, the resulting `qs` is computed as 

        qs = qS(S) ,

where `S` is the system matrix defined as        

             S =  ( abs(A)+abs(E)  abs(B) )
                  (    abs(C)        0    )

A large value of `qs` indicates a possible poorly scaled state-space model.   

[1] F.M.Dopico, M.C.Quintana and P. van Dooren, 
    "Diagonal scalings for the eigenstructure of arbitrary pencils", SIMAX, 43:1213-1237, 2022. 
"""
function lsbalqual(A::AbstractMatrix{T}, E::Union{AbstractMatrix{T},UniformScaling{Bool}}, B::AbstractVecOrMat{T}, C::AbstractMatrix{T}; SysMat = false) where {T}
    (!(typeof(E) <: AbstractMatrix) || isequal(E,I)) && (return lsbalqual(A,B,C; SysMat))
    return SysMat ? qS1([abs.(A).+abs.(E) B; C zeros(T,size(C,1),size(B,2))]) : 
                    max(qS1(A),qS1(E),qS1(B),qS1(C))
end 
"""
     lsbalance!(A, E, B, C; withB = true, withC = true, pow2, maxiter = 100, tol = 1) -> (D1,D2)

Reduce the 1-norm of the matrix 

             S =  ( abs(A)+abs(E)  abs(B) )
                  (    abs(C)        0    )

corresponding to a descriptor system triple `(A-λE,B,C)` by row and column balancing. 
This involves diagonal similarity transformations `D1*(A-λE)*D2` applied
iteratively to `abs(A)+abs(E)` to make the rows and columns of 
                             
                  diag(D1,I)  * S * diag(D2,I)
     
as close in norm as possible.
     
The balancing can be performed optionally on the following 
particular system matrices:   

        S = abs(A)+abs(E), if withB = false and withC = false ,
        S = ( abs(A)+abs(E)  abs(B) ), if withC = false,     or    
        S = ( abs(A)+abs(E) ), if withB = false .
            (   abs(C)     )       

The keyword argument `maxiter = k` specifies the maximum number of iterations `k` 
allowed in the balancing algorithm. 

The keyword argument `tol = τ`, with `τ ≤ 1`,  specifies the tolerance used in the stopping criterion.   

If the keyword argument `pow2 = true` is specified, then the components of the resulting 
optimal `D1` and `D2` are replaced by their nearest integer powers of 2, in which case the 
scaling of matrices is done exactly in floating point arithmetic. 
If `pow2 = false`, the optimal values `D1` and `D2` are returned.

_Note:_ This function is an extension of the MATLAB function `rowcolsums.m` of [1]. 

[1] F.M.Dopico, M.C.Quintana and P. van Dooren, 
    "Diagonal scalings for the eigenstructure of arbitrary pencils", SIMAX, 43:1213-1237, 2022. 
"""
function lsbalance!(A::AbstractMatrix{T}, E::Union{AbstractMatrix{T},UniformScaling{Bool}}, B::AbstractVecOrMat{T}, C::AbstractMatrix{T}; 
                    withB = true, withC = true, maxred = 16, maxiter = 100, tol = 1, pow2 = true) where {T}
    radix = real(T)(2.)
    emat = (typeof(E) <: AbstractMatrix)
    eident = !emat || isequal(E,I) 
    n = LinearAlgebra.checksquare(A)
    emat && (n,n) != size(E) && throw(DimensionMismatch("A and E must have the same dimensions"))
    n == size(B,1) || throw(DimensionMismatch("A and B must have compatible dimensions"))
    n == size(C,2) || throw(DimensionMismatch("A and C must have compatible dimensions"))
 
    n == 0 && (return Diagonal(T[]), Diagonal(T[]))
  
    if eident 
       D = lsbalance!(A, B, C; withB, withC, maxred)
       return inv(D), D
    else
      MA = abs.(A)+abs.(E)   
      MB = abs.(B)   
      MC = abs.(C)   
      r = fill(real(T)(n),n); c = copy(r)
      # Scale the matrix to have total sum(sum(M))=sum(c)=sum(r);
      sumcr = sum(c) 
      sumM = sum(MA) 
      withB && (sumM += sum(MB))
      withC && (sumM += sum(MC))
      sc = sumcr/sumM
      lmul!(1/sc,c); lmul!(1/sc,r)
      t = sqrt(sumcr/sumM); Dl = Diagonal(fill(t,n)); Dr = Diagonal(fill(t,n))
      # Scale left and right to make row and column sums equal to r and c
      conv = false
      for i = 1:maxiter
          conv = true
          cr = sum(MA,dims=1)
          withC && (cr .+= sum(MC,dims=1))
          dr = Diagonal(reshape(cr,n)./c)
          rdiv!(MA,dr); rdiv!(MC,dr) 
          er = minimum(dr.diag)/maximum(dr) 
          rdiv!(Dr,dr)
          cl = sum(MA,dims=2)
          withB && (cl .+= sum(MB,dims=2))
          dl = Diagonal(reshape(cl,n)./r)
          ldiv!(dl,MA); ldiv!(dl,MB)
          el = minimum(dl.diag)/maximum(dl) 
          rdiv!(Dl,dl)
         #  @show i, er, el
          max(1-er,1-el) < tol/2 && break
          conv = false
      end
      conv || (@warn "the iterative algorithm did not converge in $maxiter iterations")
      # Finally scale the two scalings to have equal maxima
      scaled = sqrt(maximum(Dr)/maximum(Dl))
      rmul!(Dl,scaled); rmul!(Dr,1/scaled)
      if pow2
         Dl = Diagonal(radix .^(round.(Int,log2.(Dl.diag))))
         Dr = Diagonal(radix .^(round.(Int,log2.(Dr.diag))))
      end
      lmul!(Dl,A); rmul!(A,Dr)
      lmul!(Dl,E); rmul!(E,Dr)
      lmul!(Dl,B)
      rmul!(C,Dr)
      return Dl, Dr  
   end
end
"""
    qs = qS1(M) 

Compute the 1-norm based scaling quality `qs = qS(abs(M))` of a matrix `M`, 
where `qS(⋅)` is the scaling quality measure defined in Definition 5.5 of [1] for 
nonnegative matrices. Nonzero rows and columns in `M` are allowed.   

[1] F.M.Dopico, M.C.Quintana and P. van Dooren, 
    "Diagonal scalings for the eigenstructure of arbitrary pencils", SIMAX, 43:1213-1237, 2022. 
"""
function qS1(M)
    (size(M,1) == 0 || size(M,2) == 0) && (return one(real(eltype(M))))
    temp = sum(abs,M,dims=1)
    tmax = maximum(temp)
    rmax = tmax == 0 ? one(real(eltype(M))) : tmax/minimum(temp[temp .!= 0])
    temp = sum(abs,M,dims=2)
    tmax = maximum(temp)
    cmax = tmax == 0 ? one(real(eltype(M))) : tmax/minimum(temp[temp .!= 0])
    return max(rmax,cmax)
end
