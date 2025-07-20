
"""
    _sreduceB!(A::AbstractMatrix{T},E::AbstractMatrix{T},B::AbstractMatrix{T},Q::Union{AbstractMatrix{T},Nothing}, tol::Real; 
                fast = true, withQ = true) -> ρ

Reduce the `n x m` matrix `B` using an orthogonal or unitary similarity transformation `Q1` to the row 
compressed form 

     BT = Q1'*B = [ B11 ] ρ
                  [  0  ] n-ρ
                     m      

where `B11` has full row rank `ρ`. `Q1'*A`, `Q1'*E` and `BT` are returned in `A`, `E` and `B`, respectively. 
The performed orthogonal or unitary transformations are accumulated in `Q` if `withQ = true`. 
The rank decisions use the absolute tolerance `tol` for the nonzero elements of `B`.
"""
function _sreduceB!(A::AbstractMatrix{T},E::AbstractMatrix{T},B::AbstractVecOrMat{T},
                    Q::Union{AbstractMatrix{T},Nothing}, tol::Real; 
                    fast::Bool = true, withQ::Bool = true) where T <: BlasFloat
   n, m = typeof(B) <: AbstractVector ? (length(B),1) : size(B)
   (m == 0 || n == 0) && (return 0)
   mA, nA = size(A) 
   ZERO = zero(T)
   if m == 1 
      b = view(B,:,1)
      n == 1 && (abs(b[1]) > tol ? (return 1) : (b[1] = ZERO; return 0))
      τ, β = larfg!(b)
      if abs(β) <= tol
         b[:] = zeros(T,n)
         return 0
      else
         larf!('L', b, conj(τ), A)  
         larf!('L', b, conj(τ), E)  
         withQ && larf!('R', b, τ, Q) 
         b[:] = [ fill(β,1); zeros(T,n-1)] 
         return 1
      end
end
   if fast
      _, τ, jpvt = LinearAlgebra.LAPACK.geqp3!(B)
      ρ = count(x -> x > tol, abs.(diag(B))) 
      T <: Complex ? tran = 'C' : tran = 'T'
      LinearAlgebra.LAPACK.ormqr!('L',tran,B,τ,A)
      LinearAlgebra.LAPACK.ormqr!('L',tran,B,τ,E)
      withQ && LinearAlgebra.LAPACK.ormqr!('R','N',B,τ,Q) 
      B[:,:] = [ triu(B[1:ρ,:])[:,invperm(jpvt)]; zeros(T,n-ρ,m) ]
   else
      if n > m
         _, τ = LinearAlgebra.LAPACK.geqrf!(B)
         T <: Complex ? tran = 'C' : tran = 'T'
         LinearAlgebra.LAPACK.ormqr!('L',tran,B,τ,A)
         LinearAlgebra.LAPACK.ormqr!('L',tran,B,τ,E)
         withQ && LinearAlgebra.LAPACK.ormqr!('R','N',B,τ,Q) 
         B[:,:] = [ triu(B[1:m,:]); zeros(T,n-m,m) ]
      end
      mn = min(n,m)
      ics = 1:mn
      jcs = 1:m
      SVD = svd(B[ics,jcs], full = true)
      ρ = count(x -> x > tol, SVD.S) 
      ρ == mn && (return ρ)
      B[ics,jcs] = [ Diagonal(SVD.S[1:ρ])*SVD.Vt[1:ρ,:]; zeros(T,mn-ρ,m) ]
      ρ == 0 && (return ρ)
      withQ && (Q[:,ics] = Q[:,ics]*SVD.U)
      A[ics,:] = SVD.U'*A[ics,:]
      E[ics,:] = SVD.U'*E[ics,:]
   end
   return ρ 
end
function _sreduceB!(A::AbstractMatrix{T},E::AbstractMatrix{T},B::AbstractVecOrMat{T},
                    Q::Union{AbstractMatrix{T},Nothing}, tol::Real; 
                    fast::Bool = true, withQ::Bool = true) where {T}
   n, m = typeof(B) <: AbstractVector ? (length(B),1) : size(B)
   (m == 0 || n == 0) && (return 0)
   mA, nA = size(A) 
   ZERO = zero(T)
   if m == 1 
      b = view(B,:,1)
      n == 1 && (abs(b[1]) > tol ? (return 1) : (b[1] = ZERO; return 0))
      #τ, β = larfg!(b)
      τ  = LinearAlgebra.reflector!(b)
      β = b[1]
      if abs(β) <= tol
         b[:] = zeros(T,n)
         return 0
      else
         # larf!('L', b, conj(τ), A)  
         # larf!('L', b, conj(τ), E)  
         LinearAlgebra.reflectorApply!(b, conj(τ), A)
         LinearAlgebra.reflectorApply!(b, conj(τ), E)
         #withQ && larf!('R', b, τ, Q) 
         withQ && reflectorApply!(Q, b, τ) 
         b[:] = [ fill(β,1); zeros(T,n-1)] 
         return 1
      end
end
   if fast
      # _, τ, jpvt = LinearAlgebra.LAPACK.geqp3!(B)
      # F = qr!(B)
      # ρ = count(x -> x > tol, abs.(diag(B))) 
      F = qr!(B,ColumnNorm())
      ρ = count(x -> x > tol, abs.(diag(B))) 
      jpvt = F.p
      # T <: Complex ? tran = 'C' : tran = 'T'
      #LinearAlgebra.LAPACK.ormqr!('L',tran,B,τ,A)
      lmul!(F.Q',A)
      #LinearAlgebra.LAPACK.ormqr!('L',tran,B,τ,E)
      lmul!(F.Q',E)
      #withQ && LinearAlgebra.LAPACK.ormqr!('R','N',B,τ,Q) 
      withQ && rmul!(Q,F.Q) 
      B[:,:] = [ triu(B[1:ρ,:])[:,invperm(jpvt)]; zeros(T,n-ρ,m) ]
   else
      if n > m
         # _, τ = LinearAlgebra.LAPACK.geqrf!(B)
         # T <: Complex ? tran = 'C' : tran = 'T'
         # LinearAlgebra.LAPACK.ormqr!('L',tran,B,τ,A)
         # LinearAlgebra.LAPACK.ormqr!('L',tran,B,τ,E)
         # withQ && LinearAlgebra.LAPACK.ormqr!('R','N',B,τ,Q) 
         F = qr!(B)
         lmul!(F.Q',A)
         lmul!(F.Q',E)
         withQ && rmul!(Q,F.Q) 
         B[:,:] = [ triu(B[1:m,:]); zeros(T,n-m,m) ]
      end
      mn = min(n,m)
      ics = 1:mn
      jcs = 1:m
      SVD = svd(B[ics,jcs], full = true)
      ρ = count(x -> x > tol, SVD.S) 
      ρ == mn && (return ρ)
      B[ics,jcs] = [ Diagonal(SVD.S[1:ρ])*SVD.Vt[1:ρ,:]; zeros(T,mn-ρ,m) ]
      ρ == 0 && (return ρ)
      withQ && (Q[:,ics] = Q[:,ics]*SVD.U)
      A[ics,:] = SVD.U'*A[ics,:]
      E[ics,:] = SVD.U'*E[ics,:]
   end
   return ρ 
end
"""
    _sreduceC!(A::AbstractMatrix{T},E::AbstractMatrix{T},C::AbstractMatrix{T},Z::Union{AbstractMatrix{T},Nothing}, tol::Real; 
                fast = true, withZ = true) -> ρ

Reduce the `p x n` matrix `C` using an orthogonal or unitary similarity transformation `Z1` to the column 
compressed form 

     CT = C*Z1 = [ 0  C11 ] p
                  n-ρ  ρ

where `C11` has full column rank `ρ`. `A*Z1`, `E*Z1` and `CT` are returned in `A`, `E` and `C`, respectively. 
The performed orthogonal or unitary transformations are accumulated in `Z` if `withZ = true`. 
The rank decisions use the absolute tolerance `tol` for the nonzero elements of `C`.
"""
function _sreduceC!(A::AbstractMatrix{T},E::AbstractMatrix{T},C::AbstractMatrix{T},
                    Z::Union{AbstractMatrix{T},Nothing}, tol::Real; 
                    fast::Bool = true, withZ::Bool = true) where T <: BlasFloat
   p, n = size(C)                
   (p == 0 || n == 0) && (return 0)
   mA, nA = size(A) 
   ZERO = zero(T)
   ia = 1:n
   if p == 1 
      c = view(C,1,:)
      n == 1 && (abs(c[1]) > tol ? (return 1) : (c[1] = ZERO; return 0))
      τ, β = larfgl!(c)
      if abs(β) <= tol
         c[:] = zeros(T,n)
         return 0
      else
         T <: Complex && (c[:] = conj(c))
         τ = conj(τ)
         larf!('R', c, τ, A)  
         larf!('R', c, τ, E)  
         withZ && larf!('R', c, τ, Z) 
         c[:] = [zeros(T,n-1); fill(β,1)]; 
         return 1
      end
end
   if fast
      # compute the RQ decomposition with row pivoting 
      temp = reverse(copy(transpose(C)),dims=1)
      _, τ, jp = LinearAlgebra.LAPACK.geqp3!(temp) 
      ρ = count(x -> x > tol, abs.(diag(temp))) 
      np = min(n,p)
      for i = 1:np
         v = reverse(temp[i:n,i]); v[end] = 1;
         T <: Complex && (v = conj(v))
         it = 1:n-i+1
         larf!('R',v, conj(τ[i]), view(A,:,it))
         larf!('R',v, conj(τ[i]), view(E,:,it))
         withZ && larf!('R',v, conj(τ[i]), view(Z,:,it))
      end
      C[:,:] = [ zeros(T,p,n-ρ) reverse(transpose(triu(temp[1:ρ,:])),dims=2)[invperm(jp),:] ]
   else
      if n > p
         # compute the RQ decomposition  
         _, τ = LinearAlgebra.LAPACK.gerqf!(C)
         T <: Complex ? tran = 'C' : tran = 'T'
         LinearAlgebra.LAPACK.ormrq!('R',tran,C,τ,A)
         LinearAlgebra.LAPACK.ormrq!('R',tran,C,τ,E)
         withZ && LinearAlgebra.LAPACK.ormrq!('R',tran,C,τ,Z) 
         C[:,:] = [ zeros(T,p,n-p) triu(C[1:p,n-p+1:n]) ]
      end
      pn = min(n,p)
      ics = 1:p
      jcs = n-pn+1:n
      SVD = svd(C[ics,jcs], full = true)
      ρ = count(x -> x > tol, SVD.S) 
      ρ == pn && (return ρ)
      C[ics,jcs] = [ zeros(T,p,pn-ρ) reverse(SVD.U,dims=2)[:,p-ρ+1:end]*Diagonal(reverse(SVD.S[1:ρ])) ]
      ρ == 0 && (return ρ)
      Z1 = reverse(SVD.V,dims=2)
      withZ && (Z[:,jcs] = Z[:,jcs]*Z1)
      A[:,jcs] = A[:,jcs]*Z1
      E[:,jcs] = E[:,jcs]*Z1
   end
   return ρ 
end
function _sreduceC!(A::AbstractMatrix{T},E::AbstractMatrix{T},C::AbstractMatrix{T},
                    Z::Union{AbstractMatrix{T},Nothing}, tol::Real; 
                    fast::Bool = true, withZ::Bool = true) where {T}
   p, n = size(C)                
   (p == 0 || n == 0) && (return 0)
   mA, nA = size(A) 
   ZERO = zero(T)
   ia = 1:n
   if p == 1 
      c = view(C,1,:)
      n == 1 && (abs(c[1]) > tol ? (return 1) : (c[1] = ZERO; return 0))
      #τ, β = larfgl!(c)
      τ = reflectorf!(c)
      β = c[end]
      if abs(β) <= tol
         c[:] = zeros(T,n)
         return 0
      else
         T <: Complex && (c[:] = conj(c))
         τ = conj(τ)
         #larf!('R', c, τ, A)  
         reflectorfApply!(A, c, τ)  
         #larf!('R', c, τ, E)  
         reflectorfApply!(E, c, τ)  
         #withZ && larf!('R', c, τ, Z) 
         withZ && reflectorfApply!(Z, c, τ)  
         c[:] = [zeros(T,n-1); fill(β,1)]; 
         return 1
      end
   end
   if fast
      # compute the RQ decomposition with row pivoting 
      temp = reverse(copy(transpose(C)),dims=1)
      # _, τ, jp = LinearAlgebra.LAPACK.geqp3!(temp) 
      # ρ = count(x -> x > tol, abs.(diag(temp))) 
      # np = min(n,p)
      F = qr!(temp,ColumnNorm())
      ρ = count(x -> x > tol, abs.(diag(temp))) 
      np = min(n,p)
      τ = F.τ
      jp = F.p
      for i = 1:np
         v = reverse(temp[i:n,i]); v[end] = 1;
         T <: Complex && (v = conj(v))
         it = 1:n-i+1
         #larf!('L',v, τ[i], view(A,it,:))
         #LinearAlgebra.reflectorApply!(v,τ[i],view(A,it,:))
         reflectorfApply!(view(A,:,it),v,conj(τ[i]))
         reflectorfApply!(view(E,:,it),v,conj(τ[i]))
         withZ && reflectorfApply!(view(Z,:,it),v,conj(τ[i]))
         # T <: Complex && (v = conj(v))
         # it = 1:n-i+1
         # larf!('R',v, conj(τ[i]), view(A,:,it))
         # larf!('R',v, conj(τ[i]), view(E,:,it))
         # withZ && larf!('R',v, conj(τ[i]), view(Z,:,it))
      end
      C[:,:] = [ zeros(T,p,n-ρ) reverse(transpose(triu(temp[1:ρ,:])),dims=2)[invperm(jp),:] ]
   else
      if n > p
         # compute the RQ decomposition  
         # _, τ = LinearAlgebra.LAPACK.gerqf!(C)
         # T <: Complex ? tran = 'C' : tran = 'T'
         # LinearAlgebra.LAPACK.ormrq!('R',tran,C,τ,A)
         # LinearAlgebra.LAPACK.ormrq!('R',tran,C,τ,E)
         # withZ && LinearAlgebra.LAPACK.ormrq!('R',tran,C,τ,Z) 
         # C[:,:] = [ zeros(T,p,n-p) triu(C[1:p,n-p+1:n]) ]
         F = qr(reverse(C,dims=1)')
         reverse!(rmul!(C,F.Q),dims=2)
         #LinearAlgebra.LAPACK.ormrq!('R',tran,C1,τ,A1)
         reverse!(rmul!(A,F.Q),dims=2)
         reverse!(rmul!(E,F.Q),dims=2)
         withZ && reverse!(rmul!(Z,F.Q),dims=2)
         C[:,:] = [ zeros(T,p,n-p) triu(C[1:p,n-p+1:n]) ]
      end
      pn = min(n,p)
      ics = 1:p
      jcs = n-pn+1:n
      SVD = svd(C[ics,jcs], full = true)
      ρ = count(x -> x > tol, SVD.S) 
      ρ == pn && (return ρ)
      C[ics,jcs] = [ zeros(T,p,pn-ρ) reverse(SVD.U,dims=2)[:,p-ρ+1:end]*Diagonal(reverse(SVD.S[1:ρ])) ]
      ρ == 0 && (return ρ)
      Z1 = reverse(SVD.V,dims=2)
      withZ && (Z[:,jcs] = Z[:,jcs]*Z1)
      A[:,jcs] = A[:,jcs]*Z1
      E[:,jcs] = E[:,jcs]*Z1
   end
   return ρ 
end

"""
    _sreduceBA!(n::Int,m::Int,A::AbstractMatrix{T},B::AbstractMatrix{T},C::Union{AbstractMatrix{T},Missing},Q::Union{AbstractMatrix{T},Nothing}, tol::Real; 
                fast = true, init = true, roff = 0, coff = 0, withQ = true) -> ρ 

Reduce for `init = true`, the pair `(A,B)` using an orthogonal or unitary similarity transformation on the matrices `A` and `B` of the form 
`H =  Q1'*A*Q1`, `G = Q1'*B`, to the form

     G = [ B11 ]  H = [ A11 A12 ] ρ
         [ 0   ]        B2  A22 ] n-ρ
           m            ρ   n-ρ    

where `B11` has full row rank `ρ`. `H`, `G` and `C*Q1` are returned in `A`, `B` and `C`, respectively. 
The performed orthogonal or unitary transformations are accumulated in `Q` if `withQ = true`. 
The rank decisions use the absolute tolerance `tol` for the nonzero elements of `B`.

Reduce for `init = false`, the matrix `A` of the form

    A = [ *   *  *  ] roff
        [ 0   B1 A1 ] n
         coff m  n 

using an orthogonal or unitary similarity transformation on the submatrices A1 and B1 of the form 
H1 =  Q1'*A1*Q1, G1 = Q1'*B1, to the form
    
                                     [ *   *    *   *   ] roff
     H = diag(I,Q1')*A*diag(I,Q1) =  [ *   B11  A11 A12 ] ρ
                                     [ 0   0    B2  A22 ] n-ρ
                                      coff m    ρ   n-ρ    

where `B11` has full row rank `ρ`. `H` and `C*diag(I,Q1)` are returned in `A` and `C`, respectively, 
and `B` is unchanged. The performed orthogonal or unitary transformations are accumulated in `Q` if `withQ = true`. 
The rank decisions use the absolute tolerance `tol` for the nonzero elements of `A`.
"""
function _sreduceBA!(n::Int,m::Int,A::AbstractMatrix{T},B::AbstractVecOrMat{T},C::Union{AbstractMatrix{T},Missing},
                     Q::Union{AbstractMatrix{T},Nothing}, tol::Real; 
                     fast::Bool = true, init::Bool = true, roff::Int = 0, coff::Int = 0, withQ::Bool = true) where T <: BlasFloat
   (m == 0 || n == 0) && (return 0)
   mA, nA = size(A) 
   ZERO = zero(T)
   ib = roff+1:roff+n
   ia = 1:roff+n
   if init 
      # coff and roff must be zero
      (coff == 0 && roff == 0) || error("coff and roff must be zero at first call")
      ja = 1:n
      B1 = view(B,ib,1:m)
      A1 = view(A,ib,ja)
   else
      (coff+m+n == nA && roff+n == mA) || error("coff and roff must have compatible values with the dimensions of A")
      ja = coff+m+1:coff+m+n
      jb = coff+1:coff+m
      B1 = view(A,ib,jb)
      A1 = view(A,ib,ja)
   end
   if m == 1 
      b = view(B1,:,1)
      n == 1 && (abs(b[1]) > tol ? (return 1) : (b[1] = ZERO; return 0))
      τ, β = larfg!(b)
      if abs(β) <= tol
         b[:] = zeros(T,n)
         return 0
      else
         #T <: Complex && (b[:] = conj(b))
         larf!('L', b, conj(τ), A1)  
         larf!('R', b, τ, view(A, ia, ja))  
         ismissing(C) || larf!('R', b, τ, view(C,:,ja))  
         withQ && larf!('R', b, τ, view(Q,:,ib)) 
         b[:] = [ fill(β,1); zeros(T,n-1)] 
         return 1
      end
   end
   if fast
      B1, τ, jpvt = LinearAlgebra.LAPACK.geqp3!(B1)
      ρ = count(x -> x > tol, abs.(diag(B1))) 
      T <: Complex ? tran = 'C' : tran = 'T'
      LinearAlgebra.LAPACK.ormqr!('L',tran,B1,τ,A1)
      LinearAlgebra.LAPACK.ormqr!('R','N',B1,τ,view(A,ia,ja))
      withQ && LinearAlgebra.LAPACK.ormqr!('R','N',B1,τ,view(Q,:,ib)) 
      ismissing(C) || LinearAlgebra.LAPACK.ormqr!('R','N',B1,τ,view(C,:,ja)) 
      B1[:,:] = [ triu(B1[1:ρ,:])[:,invperm(jpvt)]; zeros(T,n-ρ,m) ]
   else
      if n > m
         B1, τ = LinearAlgebra.LAPACK.geqrf!(B1)
         T <: Complex ? tran = 'C' : tran = 'T'
         LinearAlgebra.LAPACK.ormqr!('L',tran,B1,τ,A1)
         LinearAlgebra.LAPACK.ormqr!('R','N',B1,τ,view(A,ia,ja))
         withQ && LinearAlgebra.LAPACK.ormqr!('R','N',B1,τ,view(Q,:,ib)) 
         ismissing(C) || LinearAlgebra.LAPACK.ormqr!('R','N',B1,τ,view(C,:,ja)) 
         B1[:,:] = [ triu(B1[1:m,:]); zeros(T,n-m,m) ]
      end
      mn = min(n,m)
      ics = 1:mn
      jcs = 1:m
      SVD = svd(B1[ics,jcs], full = true)
      ρ = count(x -> x > tol, SVD.S) 
      ρ == mn && (return ρ)
      B1[ics,jcs] = [ Diagonal(SVD.S[1:ρ])*SVD.Vt[1:ρ,:]; zeros(T,mn-ρ,m) ]
      ρ == 0 && (return ρ)
      ibt = roff+1:roff+mn
      init ? (jt = coff+1:coff+mn) : (jt = coff+m+1:coff+m+mn)
      withQ && (Q[:,ibt] = Q[:,ibt]*SVD.U)
      A[ibt,ja] = SVD.U'*A[ibt,ja]
      A[ia,jt] = A[ia,jt]*SVD.U
      ismissing(C) || (C[:,jt] = C[:,jt]*SVD.U) 
   end
   return ρ 
end
function _sreduceBA!(n::Int,m::Int,A::AbstractMatrix{T},B::AbstractVecOrMat{T},C::Union{AbstractMatrix{T},Missing},
                     Q::Union{AbstractMatrix{T},Nothing}, tol::Real; 
                     fast::Bool = true, init::Bool = true, roff::Int = 0, coff::Int = 0, withQ::Bool = true) where {T}
   (m == 0 || n == 0) && (return 0)
   mA, nA = size(A) 
   ZERO = zero(T)
   ib = roff+1:roff+n
   ia = 1:roff+n
   if init 
      # coff and roff must be zero
      (coff == 0 && roff == 0) || error("coff and roff must be zero at first call")
      ja = 1:n
      B1 = view(B,ib,1:m)
      A1 = view(A,ib,ja)
   else
      (coff+m+n == nA && roff+n == mA) || error("coff and roff must have compatible values with the dimensions of A")
      ja = coff+m+1:coff+m+n
      jb = coff+1:coff+m
      B1 = view(A,ib,jb)
      A1 = view(A,ib,ja)
   end
   if m == 1 
      b = view(B1,:,1)
      n == 1 && (abs(b[1]) > tol ? (return 1) : (b[1] = ZERO; return 0))
      #τ, β = larfg!(b)
      τ = LinearAlgebra.reflector!(b)
      β = b[1]
      if abs(β) <= tol
         b[:] = zeros(T,n)
         return 0
      else
         #T <: Complex && (b[:] = conj(b))
         #larf!('L', b, conj(τ), A1)  
         #b[1] = one(T)
         LinearAlgebra.reflectorApply!(b, conj(τ),A1)
         #larf!('R', b, τ, view(A, ia, ja))  
         reflectorApply!(view(A, ia, ja), b, τ)
         #ismissing(C) || larf!('R', b, τ, view(C,:,ja))  
         ismissing(C) || reflectorApply!(view(C,:,ja), b, τ)  
         #withQ && larf!('R', b, τ, view(Q,:,ib)) 
         withQ && reflectorApply!(view(Q,:,ib), b, τ)  
         b[:] = [ fill(β,1); zeros(T,n-1)] 
         return 1
      end
   end
   if fast
      #B1, τ, jpvt = LinearAlgebra.LAPACK.geqp3!(B1)
      F = qr!(B1,ColumnNorm())
      ρ = count(x -> x > tol, abs.(diag(B1))) 
      # T <: Complex ? tran = 'C' : tran = 'T'
      # LinearAlgebra.LAPACK.ormqr!('L',tran,B1,τ,A1)
      lmul!(F.Q',A1)
      #LinearAlgebra.LAPACK.ormqr!('R','N',B1,τ,view(A,ia,ja))
      rmul!(view(A,ia,ja),F.Q)
      #withQ && LinearAlgebra.LAPACK.ormqr!('R','N',B1,τ,view(Q,:,ib)) 
      withQ && rmul!(view(Q,:,ib),F.Q)
      #ismissing(C) || LinearAlgebra.LAPACK.ormqr!('R','N',B1,τ,view(C,:,ja)) 
      ismissing(C) || rmul!(view(C,:,ja),F.Q)
      B1[:,:] = [ triu(B1[1:ρ,:])[:,invperm(F.p)]; zeros(T,n-ρ,m) ]
   else
      if n > m
         # B1, τ = LinearAlgebra.LAPACK.geqrf!(B1)
         # T <: Complex ? tran = 'C' : tran = 'T'
         F = qr!(B1)
         #LinearAlgebra.LAPACK.ormqr!('L',tran,B1,τ,A1)
         lmul!(F.Q',A1)
         #LinearAlgebra.LAPACK.ormqr!('R','N',B1,τ,view(A,ia,ja))
         rmul!(view(A,ia,ja),F.Q)
         #withQ && LinearAlgebra.LAPACK.ormqr!('R','N',B1,τ,view(Q,:,ib)) 
         withQ && rmul!(view(Q,:,ib),F.Q) 
         #ismissing(C) || LinearAlgebra.LAPACK.ormqr!('R','N',B1,τ,view(C,:,ja)) 
         ismissing(C) || rmul!(view(C,:,ja),F.Q) 
         #B1[:,:] = [ triu(B1[1:m,:]); zeros(T,n-m,m) ]
         B1[:,:] = [ F.R; zeros(T,n-m,m) ]
      end
      mn = min(n,m)
      ics = 1:mn
      jcs = 1:m
      SVD = svd(B1[ics,jcs], full = true)
      ρ = count(x -> x > tol, SVD.S) 
      ρ == mn && (return ρ)
      B1[ics,jcs] = [ Diagonal(SVD.S[1:ρ])*SVD.Vt[1:ρ,:]; zeros(T,mn-ρ,m) ]
      ρ == 0 && (return ρ)
      ibt = roff+1:roff+mn
      init ? (jt = coff+1:coff+mn) : (jt = coff+m+1:coff+m+mn)
      withQ && (Q[:,ibt] = Q[:,ibt]*SVD.U)
      A[ibt,ja] = SVD.U'*A[ibt,ja]
      A[ia,jt] = A[ia,jt]*SVD.U
      ismissing(C) || (C[:,jt] = C[:,jt]*SVD.U) 
   end
   return ρ 
end
"""
    _sreduceBA2!(n::Int,m::Int,A::AbstractMatrix{T},B::AbstractMatrix{T},C::Union{AbstractMatrix{T},Missing},
                 Q::Union{AbstractMatrix{T},Nothing}, tol::Real; 
                 fast = true, coff = 0, withQ = true) -> ρ 

Reduce the matrices `B` and `A` of the form


    B = [ *  *  ]       A = [ *    *   A12 ] 
        [ B1 B2 ] n         [ *   A21  A22 ] n
          m  k               coff       n

using an orthogonal or unitary transformation of the form `B <- G = diag(I,Q1')*B`, 
such that `G` has the form

         [  *   *  ]    
     G = [ B11 B12 ] ρ      ,
         [  0  B22 ] n-ρ    
            m  k                

where `B11` has full row rank `ρ` and the blocks `A12`, `A21` and `A22` of `A` are
updated such that `A12 <- A12*Q1`, `A21 <- Q1'*A21` and `A22 <- Q1'*A22`. 
Update  `C <- C*diag(I,Q1)` and `Q <- Q*diag(I,Q1)` if `withQ = true`.  
The rank decisions use the absolute tolerance `tol` for the nonzero elements of `B`.
"""
function _sreduceBA2!(n::Int,m::Int,A::AbstractMatrix{T},B::AbstractMatrix{T},C::Union{AbstractMatrix{T},Missing},
                     Q::Union{AbstractMatrix{T},Nothing}, tol::Real; 
                     fast::Bool = true, coff::Int = 0, withQ::Bool = true) where T <: BlasFloat
   (m == 0 || n == 0) && (return 0)
   nA, mB = size(B) 
   roff = nA - n 
   ZERO = zero(T)
   ib = roff+1:nA    # row range of [B1, B2] and [A21 A22]
   ia = 1:nA         # row range of [A12; A22]
   ja = ib           # column rabge of A2 = [A12; A22]
   ja1 = coff+1:nA   # column range of A1 = [A21 A22]
   B1 = view(B,ib,1:m)
   B2mat = (m < mB)  # B2 is present if B has more columns than m
   B2mat && (B2 = view(B,ib,m+1:mB))
   A1 = view(A,ib,ja1)  
   A2 = view(A,ia,ja)
   if m == 1 
      b = view(B1,:,1)
      n == 1 && (abs(b[1]) > tol ? (return 1) : (b[1] = ZERO; return 0))
      τ, β = larfg!(b)
      if abs(β) <= tol
         b[:] = zeros(T,n)
         return 0
      else
         larf!('L', b, conj(τ), A1)  
         B2mat &&  larf!('L', b, conj(τ), B2)  
         larf!('R', b, τ, A2)  
         ismissing(C) || larf!('R', b, τ, view(C,:,ja))  
         withQ && larf!('R', b, τ, view(Q,:,ib)) 
         b[:] = [ fill(β,1); zeros(T,n-1)] 
         return 1
      end
   end
   if fast
      B1, τ, jpvt = LinearAlgebra.LAPACK.geqp3!(B1)
      ρ = count(x -> x > tol, abs.(diag(B1))) 
      T <: Complex ? tran = 'C' : tran = 'T'
      LinearAlgebra.LAPACK.ormqr!('L',tran,B1,τ,A1)
      B2mat &&  LinearAlgebra.LAPACK.ormqr!('L', tran, B1, τ, B2)  
      LinearAlgebra.LAPACK.ormqr!('R','N',B1,τ,A2)
      withQ && LinearAlgebra.LAPACK.ormqr!('R','N',B1,τ,view(Q,:,ib)) 
      ismissing(C) || LinearAlgebra.LAPACK.ormqr!('R','N',B1,τ,view(C,:,ja)) 
      B1[:,:] = [ triu(B1[1:ρ,:])[:,invperm(jpvt)]; zeros(T,n-ρ,m) ]
   else
      if n > m
         B1, τ = LinearAlgebra.LAPACK.geqrf!(B1)
         T <: Complex ? tran = 'C' : tran = 'T'
         LinearAlgebra.LAPACK.ormqr!('L',tran,B1,τ,A1)
         B2mat && LinearAlgebra.LAPACK.ormqr!('L',tran,B1,τ,B2)
         LinearAlgebra.LAPACK.ormqr!('R','N',B1,τ,A2)
         withQ && LinearAlgebra.LAPACK.ormqr!('R','N',B1,τ,view(Q,:,ib)) 
         ismissing(C) || LinearAlgebra.LAPACK.ormqr!('R','N',B1,τ,view(C,:,ja)) 
         B1[:,:] = [ triu(B1[1:m,:]); zeros(T,n-m,m) ]
      end
      mn = min(n,m)
      ics = 1:mn
      jcs = 1:m
      SVD = svd(B1[ics,jcs], full = true)
      ρ = count(x -> x > tol, SVD.S) 
      ρ == mn && (return ρ)
      B1[ics,jcs] = [ Diagonal(SVD.S[1:ρ])*SVD.Vt[1:ρ,:]; zeros(T,mn-ρ,m) ]
      ρ == 0 && (return ρ)
      ibt = roff+1:roff+mn
      withQ && (Q[:,ibt] = Q[:,ibt]*SVD.U)
      A[ibt,ja1] = SVD.U'*A[ibt,ja1]
      B2mat &&  (B2[ics,:] = SVD.U'*B2[ics,:])  
      #jt = coff+1:coff+mn
      #A[ia,jt] = A[ia,jt]*SVD.U
      A[ia,ibt] = A[ia,ibt]*SVD.U
      #ismissing(C) || (C[:,jt] = C[:,jt]*SVD.U) 
      ismissing(C) || (C[:,ibt] = C[:,ibt]*SVD.U) 
   end
   return ρ 
end
function _sreduceBA2!(n::Int,m::Int,A::AbstractMatrix{T},B::AbstractMatrix{T},C::Union{AbstractMatrix{T},Missing},
                     Q::Union{AbstractMatrix{T},Nothing}, tol::Real; 
                     fast::Bool = true, coff::Int = 0, withQ::Bool = true) where {T} 
   (m == 0 || n == 0) && (return 0)
   nA, mB = size(B) 
   roff = nA - n 
   ZERO = zero(T)
   ib = roff+1:nA    # row range of [B1, B2] and [A21 A22]
   ia = 1:nA         # row range of [A12; A22]
   ja = ib           # column rabge of A2 = [A12; A22]
   ja1 = coff+1:nA   # column range of A1 = [A21 A22]
   B1 = view(B,ib,1:m)
   B2mat = (m < mB)  # B2 is present if B has more columns than m
   B2mat && (B2 = view(B,ib,m+1:mB))
   A1 = view(A,ib,ja1)  
   A2 = view(A,ia,ja)
   if m == 1 
      b = view(B1,:,1)
      n == 1 && (abs(b[1]) > tol ? (return 1) : (b[1] = ZERO; return 0))
      #τ, β = larfg!(b)
      τ = LinearAlgebra.reflector!(b)
      β = b[1]
      if abs(β) <= tol
         b[:] = zeros(T,n)
         return 0
      else
         #larf!('L', b, conj(τ), A1)  
         LinearAlgebra.reflectorApply!(b, conj(τ),A1)
         #B2mat &&  larf!('L', b, conj(τ), B2)  
         B2mat &&  LinearAlgebra.reflectorApply!(b, conj(τ),B2) 
         #larf!('R', b, τ, A2)  
         reflectorApply!(A2, b, τ)
         #ismissing(C) || larf!('R', b, τ, view(C,:,ja))  
         ismissing(C) || reflectorApply!(view(C,:,ja), b, τ)  
         #withQ && larf!('R', b, τ, view(Q,:,ib)) 
         withQ && reflectorApply!(view(Q,:,ib), b, τ)  
         b[:] = [ fill(β,1); zeros(T,n-1)] 
         return 1
      end
   end
   if fast
      # B1, τ, jpvt = LinearAlgebra.LAPACK.geqp3!(B1)
      # ρ = count(x -> x > tol, abs.(diag(B1))) 
      F = qr!(B1,ColumnNorm())
      ρ = count(x -> x > tol, abs.(diag(B1))) 
      jpvt = F.p
      #LinearAlgebra.LAPACK.ormqr!('L',tran,B1,τ,A1)
      lmul!(F.Q',A1)
      #B2mat &&  LinearAlgebra.LAPACK.ormqr!('L', tran, B1, τ, B2)  
      B2mat &&  lmul!(F.Q',B2) 
      #LinearAlgebra.LAPACK.ormqr!('R','N',B1,τ,A2)
      rmul!(A2,F.Q)
      #withQ && LinearAlgebra.LAPACK.ormqr!('R','N',B1,τ,view(Q,:,ib)) 
      withQ && rmul!(view(Q,:,ib),F.Q)
      #ismissing(C) || LinearAlgebra.LAPACK.ormqr!('R','N',B1,τ,view(C,:,ja)) 
      ismissing(C) || rmul!(view(C,:,ja),F.Q)
      B1[:,:] = [ triu(B1[1:ρ,:])[:,invperm(jpvt)]; zeros(T,n-ρ,m) ]
   else
      if n > m
         #B1, τ = LinearAlgebra.LAPACK.geqrf!(B1)
         F = qr!(B1)
         #LinearAlgebra.LAPACK.ormqr!('L',tran,B1,τ,A1)
         lmul!(F.Q',A1)
         #B2mat && LinearAlgebra.LAPACK.ormqr!('L',tran,B1,τ,B2)
         B2mat &&  lmul!(F.Q',B2) 
         #LinearAlgebra.LAPACK.ormqr!('R','N',B1,τ,A2)
         rmul!(A2,F.Q)
         #withQ && LinearAlgebra.LAPACK.ormqr!('R','N',B1,τ,view(Q,:,ib)) 
         withQ && rmul!(view(Q,:,ib),F.Q) 
         #ismissing(C) || LinearAlgebra.LAPACK.ormqr!('R','N',B1,τ,view(C,:,ja)) 
         ismissing(C) || rmul!(view(C,:,ja),F.Q) 
         #B1[:,:] = [ triu(B1[1:m,:]); zeros(T,n-m,m) ]
         B1[:,:] = [ F.R; zeros(T,n-m,m) ]
      end
      mn = min(n,m)
      ics = 1:mn
      jcs = 1:m
      SVD = svd(B1[ics,jcs], full = true)
      ρ = count(x -> x > tol, SVD.S) 
      ρ == mn && (return ρ)
      B1[ics,jcs] = [ Diagonal(SVD.S[1:ρ])*SVD.Vt[1:ρ,:]; zeros(T,mn-ρ,m) ]
      ρ == 0 && (return ρ)
      ibt = roff+1:roff+mn
      withQ && (Q[:,ibt] = Q[:,ibt]*SVD.U)
      A[ibt,ja1] = SVD.U'*A[ibt,ja1]
      B2mat &&  (B2[ics,:] = SVD.U'*B2[ics,:])  
      #jt = coff+1:coff+mn
      #A[ia,jt] = A[ia,jt]*SVD.U
      A[ia,ibt] = A[ia,ibt]*SVD.U
      #ismissing(C) || (C[:,jt] = C[:,jt]*SVD.U) 
      ismissing(C) || (C[:,ibt] = C[:,ibt]*SVD.U) 
   end
   return ρ 
end

"""
    _sreduceAC!(n::Int,p::Int,A::AbstractMatrix{T},C::AbstractMatrix{T},B::Union{AbstractMatrix{T},Missing},
                Q::Union{AbstractMatrix{T},Nothing}, tol::Real; 
                fast = true, init = true, rtrail = 0, ctrail = 0, withQ = true) -> ρ

Reduce for `init = true`, the pair `(A,C)` using an orthogonal or unitary similarity transformation on the matrices `A` and `C` of the form 
`H =  Q1'*A*Q1`, `L = C*Q1`, to the form

     H = [ A11 A12 ] n-ρ    L = [ 0  C11 ] p
         [ C2  A22 ] ρ           n-ρ  ρ
           n-ρ  ρ   

where `C11` has full column rank `ρ`. `H`, `L` and `Q1'*B` are returned in `A`, `C` and `B`, respectively. 
The performed orthogonal or unitary transformations are accumulated in `Q` if `withQ = true`. 
The rank decisions use the absolute tolerance `tol` for the nonzero elements of `C`.

Reduce for `init = false`, the matrix `A` of the form

        [ A1  *   ] n
    A = [ C1  *   ] p
        [ 0   *   ] rtrail
          n ctrail 

using an orthogonal or unitary similarity transformation on the submatrices A1 and C1 of the form 
H1 =  Q1'*A1*Q1, L1 = C1*Q1, to the form

                                     [ A11 A12  *   ] n-ρ
                                     [ C2  A22  *   ] ρ
     H = diag(Q1',I)*A*diag(Q1,I) =  [ 0   C11  *   ] p
                                     [ 0   0    *   ] rtrail
                                      n-ρ  ρ  ctrail    

where `C11` has full column rank `ρ`. `H` and `diag(Q1',I)*B` are returned in `A` and `B`, respectively, 
and `C` is unchanged. The performed orthogonal or unitary transformations are accumulated in `Q` if `withQ = true`. 
The rank decisions use the absolute tolerance `tol` for the nonzero elements of `A`.
"""
function _sreduceAC!(n::Int,p::Int,A::AbstractMatrix{T},C::AbstractMatrix{T},B::Union{AbstractVecOrMat{T},Missing},
                     Q::Union{AbstractMatrix{T},Nothing}, tol::Real; 
                     fast::Bool = true, init::Bool = true, rtrail::Int = 0, ctrail::Int = 0, withQ::Bool = true) where T <: BlasFloat
   (p == 0 || n == 0) && (return 0)
   mA, nA = size(A) 
   ZERO = zero(T)
   ia = 1:n
   if init 
      # coff and roff must be zero
      (ctrail == 0 && rtrail == 0) || error("rtrail and ctrail must be zero at first call")
      ic = 1:p
      C1 = view(C,ic,ia)
      A1 = view(A,ia,ia)
   else
      (ctrail+n == nA && rtrail+n+p == mA) || error("rtrail and ctrail must have compatible values with the dimensions of A")
      ic = n+1:n+p
      C1 = view(A,ic,ia)
      A1 = view(A,ia,ia)
   end
   if p == 1 
      c = view(C1,1,ia)
      n == 1 && (abs(c[1]) > tol ? (return 1) : (c[1] = ZERO; return 0))
      τ, β = larfgl!(c)
      if abs(β) <= tol
         c[:] = zeros(T,n)
         return 0
      else
         T <: Complex && (c[:] = conj(c))
         larf!('L', c, τ, view(A, ia, :))  
         ismissing(B) || larf!('L', c, τ, view(B, ia, :)) 
         τ = conj(τ)
         larf!('R', c, τ, A1)  
         withQ && larf!('R', c, τ, view(Q,:,ia)) 
         c[:] = [zeros(T,n-1); fill(β,1)]; 
         return 1
      end
   end
   if fast
      # compute the RQ decomposition with row pivoting 
      temp = reverse(copy(transpose(C1)),dims=1)
      temp, τ, jp = LinearAlgebra.LAPACK.geqp3!(temp) 
      ρ = count(x -> x > tol, abs.(diag(temp))) 
      np = min(n,p)
      for i = 1:np
         v = reverse(temp[i:n,i]); v[end] = 1;
         T <: Complex && (v = conj(v))
         it = 1:n-i+1
         larf!('L',v, τ[i], view(A,it,:))
         larf!('R',v, conj(τ[i]), view(A,:,it))
         ismissing(B) || larf!('L',v, τ[i], view(B,it,:))
         withQ && larf!('R',v, conj(τ[i]), view(Q,:,it))
      end
      C1[:,:] = [ zeros(T,p,n-ρ) reverse(transpose(triu(temp[1:ρ,:])),dims=2)[invperm(jp),:] ]
   else
      if n > p
         # compute the RQ decomposition  
         C1, τ = LinearAlgebra.LAPACK.gerqf!(C1)
         T <: Complex ? tran = 'C' : tran = 'T'
         LinearAlgebra.LAPACK.ormrq!('R',tran,C1,τ,A1)
         LinearAlgebra.LAPACK.ormrq!('L','N',C1,τ,view(A,ia,:))
         withQ && LinearAlgebra.LAPACK.ormrq!('R',tran,C1,τ,view(Q,:,ia)) 
         ismissing(B) || LinearAlgebra.LAPACK.ormrq!('L','N',C1,τ,view(B,ia,:)) 
         C1[:,:] = [ zeros(T,p,n-p) triu(C1[1:p,n-p+1:n]) ]
      end
      pn = min(n,p)
      ics = 1:p
      jcs = n-pn+1:n
      SVD = svd(C1[ics,jcs], full = true)
      ρ = count(x -> x > tol, SVD.S) 
      ρ == pn && (return ρ)
      C1[ics,jcs] = [ zeros(T,p,pn-ρ) reverse(SVD.U,dims=2)[:,p-ρ+1:end]*Diagonal(reverse(SVD.S[1:ρ])) ]
      ρ == 0 && (return ρ)
      Q1 = reverse(SVD.V,dims=2)
      withQ && (Q[:,jcs] = Q[:,jcs]*Q1)
      A[jcs,:] = Q1'*A[jcs,:]
      A[ia,jcs] = A[ia,jcs]*Q1
      ismissing(B) || (B[jcs,:] = Q1'*B[jcs,:])
   end
   return ρ 
end
function _sreduceAC!(n::Int,p::Int,A::AbstractMatrix{T},C::AbstractMatrix{T},B::Union{AbstractVecOrMat{T},Missing},
                     Q::Union{AbstractMatrix{T},Nothing}, tol::Real; 
                     fast::Bool = true, init::Bool = true, rtrail::Int = 0, ctrail::Int = 0, withQ::Bool = true) where {T}
   (p == 0 || n == 0) && (return 0)
   mA, nA = size(A) 
   ZERO = zero(T)
   ia = 1:n
   if init 
      # coff and roff must be zero
      (ctrail == 0 && rtrail == 0) || error("rtrail and ctrail must be zero at first call")
      ic = 1:p
      C1 = view(C,ic,ia)
      A1 = view(A,ia,ia)
   else
      (ctrail+n == nA && rtrail+n+p == mA) || error("rtrail and ctrail must have compatible values with the dimensions of A")
      ic = n+1:n+p
      C1 = view(A,ic,ia)
      A1 = view(A,ia,ia)
   end
   if p == 1 
      c = view(C1,1,ia)
      n == 1 && (abs(c[1]) > tol ? (return 1) : (c[1] = ZERO; return 0))
      #τ, β = larfgl!(c)
      τ = reflectorf!(c)
      β = c[end]
      if abs(β) <= tol
         c[:] = zeros(T,n)
         return 0
      else
         T <: Complex && (c[:] = conj(c))
         #larf!('L', c, τ, view(A, ia, :))  
         reflectorfApply!(c, τ, view(A, ia, :))  
         #ismissing(B) || larf!('L', c, τ, view(B, ia, :)) 
         ismissing(B) || reflectorfApply!(c, τ, view(B, ia, :))  
         τ = conj(τ)
         #larf!('R', c, τ, A1)  
         reflectorfApply!(A1,c, τ) 
         #withQ && larf!('R', c, τ, view(Q,:,ia)) 
         withQ && reflectorfApply!(view(Q,:,ia), c, τ) 
         c[:] = [zeros(T,n-1); fill(β,1)]; 
         return 1
      end
   end
   if fast
      # compute the RQ decomposition with row pivoting 
      temp = reverse(copy(transpose(C1)),dims=1)
      #temp, τ, jp = LinearAlgebra.LAPACK.geqp3!(temp) 
      F = qr!(temp,ColumnNorm())
      ρ = count(x -> x > tol, abs.(diag(temp))) 
      np = min(n,p)
      τ = F.τ
      jp = F.p
      for i = 1:np
         v = reverse(temp[i:n,i]); v[end] = 1;
         T <: Complex && (v = conj(v))
         it = 1:n-i+1
         #larf!('L',v, τ[i], view(A,it,:))
         #LinearAlgebra.reflectorApply!(v,τ[i],view(A,it,:))
         reflectorfApply!(v,τ[i],view(A,it,:))
         #larf!('R',v, conj(τ[i]), view(A,:,it))
         #GenericLinearAlgebra.reflectorApply!(view(A,:,it),v,conj(τ[i]))
         reflectorfApply!(view(A,:,it),v,conj(τ[i]))
         #ismissing(B) || larf!('L',v, τ[i], view(B,it,:))
         #ismissing(B) || LinearAlgebra.reflectorApply!(v,τ[i],view(B,it,:))
         ismissing(B) || reflectorfApply!(v,τ[i],view(B,it,:))
         #withQ && larf!('R',v, conj(τ[i]), view(Q,:,it))
         withQ && reflectorfApply!(view(Q,:,it),v,conj(τ[i]))
      end
      C1[:,:] = [ zeros(T,p,n-ρ) reverse(transpose(triu(temp[1:ρ,:])),dims=2)[invperm(jp),:] ]
   else
      if n > p
         # compute the RQ decomposition  
         # C1, τ = LinearAlgebra.LAPACK.gerqf!(C1)
         # T <: Complex ? tran = 'C' : tran = 'T'
         F = qr(reverse(C1,dims=1)')
         reverse!(rmul!(C1,F.Q),dims=2)
         #LinearAlgebra.LAPACK.ormrq!('R',tran,C1,τ,A1)
         reverse!(rmul!(A1,F.Q),dims=2)
         #LinearAlgebra.LAPACK.ormrq!('L','N',C1,τ,view(A,ia,:))
         reverse!(lmul!(F.Q',view(A,ia,:)),dims=1)
         #withQ && LinearAlgebra.LAPACK.ormrq!('R',tran,C1,τ,view(Q,:,ia)) 
         withQ && reverse!(rmul!(view(Q,:,ia),F.Q),dims=2) 
         #ismissing(B) || LinearAlgebra.LAPACK.ormrq!('L','N',C1,τ,view(B,ia,:)) 
         ismissing(B) || reverse!(lmul!(F.Q',view(B,ia,:)),dims=1) 
         C1[:,:] = [ zeros(T,p,n-p) triu(C1[1:p,n-p+1:n]) ]
         #C1[:,:] = [ zeros(T,p,n-p) reverse(reverse(F.R,dims=1),dims=2)' ]
      end
      pn = min(n,p)
      ics = 1:p
      jcs = n-pn+1:n
      SVD = svd(C1[ics,jcs], full = true)
      ρ = count(x -> x > tol, SVD.S) 
      ρ == pn && (return ρ)
      C1[ics,jcs] = [ zeros(T,p,pn-ρ) reverse(SVD.U,dims=2)[:,p-ρ+1:end]*Diagonal(reverse(SVD.S[1:ρ])) ]
      ρ == 0 && (return ρ)
      Q1 = reverse(SVD.V,dims=2)
      withQ && (Q[:,jcs] = Q[:,jcs]*Q1)
      A[jcs,:] = Q1'*A[jcs,:]
      A[ia,jcs] = A[ia,jcs]*Q1
      ismissing(B) || (B[jcs,:] = Q1'*B[jcs,:])
   end
   return ρ 
end
@inline function reflectorApply!(A::StridedMatrix, x::AbstractVector, τ::Number) # apply conjugate transpose reflector from right.
    m, n = size(A)
    if length(x) != n
        throw(
            DimensionMismatch(
                "reflector must have same length as second dimension of matrix",
            ),
        )
    end
    @inbounds begin
        for i = 1:m
            Aiv = A[i, 1]
            for j = 2:n
                Aiv += A[i, j] * x[j]
            end
            Aiv = Aiv * τ
            A[i, 1] -= Aiv
            for j = 2:n
                A[i, j] -= Aiv * x[j]'
            end
        end
    end
    return A
end
@inline function reflectorf!(x::AbstractVector{T}) where {T}
    #require_one_based_indexing(x)
    n = length(x)
    n == 0 && return zero(eltype(x))
    @inbounds begin
        ξ1 = x[n]
        normu = norm(x)
        if iszero(normu)
            return zero(ξ1/normu)
        end
        ν = T(copysign(normu, real(ξ1)))
        ξ1 += ν
        x[n] = -ν
        for i = 1:n-1
            x[i] /= ξ1
        end
    end
    ξ1/ν
end
@inline function reflectorfApply!(x::AbstractVector, τ::Number, A::AbstractVecOrMat)
    #require_one_based_indexing(x)
    m, n = size(A, 1), size(A, 2)
    if length(x) != m
        throw(DimensionMismatch(lazy"reflector has length $(length(x)), which must match the first dimension of matrix A, $m"))
    end
    m == 0 && return A
    @inbounds for j = 1:n
        Aj, xj = view(A, 1:m-1, j), view(x, 1:m-1)
        vAj = conj(τ)*(A[m, j] + dot(xj, Aj))
        A[m, j] -= vAj
        axpy!(-vAj, xj, Aj)
    end
    return A
end

@inline function reflectorfApply!(A::StridedMatrix, x::AbstractVector, τ::Number) # apply conjugate transpose reflector from right.
    m, n = size(A)
    if length(x) != n
        throw(
            DimensionMismatch(
                "reflector must have same length as second dimension of matrix",
            ),
        )
    end
    @inbounds begin
        for i = 1:m
            Aiv = A[i, n]
            for j = 1:n-1
                Aiv += A[i, j] * x[j]
            end
            Aiv = Aiv * τ
            A[i, n] -= Aiv
            for j = 1:n-1
                A[i, j] -= Aiv * x[j]'
            end
        end
    end
    return A
end


"""
    _sreduceBAE!(n::Int,m::Int,A::AbstractMatrix{T},E::AbstractMatrix{T},B::AbstractMatrix{T},C::Union{AbstractMatrix{T},Missing},
                 Q::Union{AbstractMatrix{T},Nothing}, Z::Union{AbstractMatrix{T},Nothing}, tol::Real; 
                 fast = true, init = true, roff = 0, coff = 0, withQ = true, withZ = true)

Reduce for `init = true`, the pair `(A-λE,B)`, with `E` upper-triangular, using an orthogonal or unitary 
similarity transformations on the matrices `A`, `E` and `B` of the form `At =  Q1'*A*Z1`, `Et =  Q1'*E*Z1`, 
`Bt = Q1'*B`, to the form

     Bt = [ B11 ]  At = [ A11 A12 ] ρ     Et = [ E11 E12 ] ρ
          [ 0   ]         B2  A22 ] n-ρ        [  0  E22 ] n-ρ
            m             ρ   n-ρ                 ρ   n-ρ

where `B11` has full row rank `ρ` and `Et` is upper-triangular. `Bt`, `At`, `Et` and `C*Z1` are returned in 
`B`, `A`, `E` and `C`, respectively. 
The performed orthogonal or unitary transformations are accumulated in `Q` as `Q <- Q*Q1` if `withQ = true`
and in `Z` as `Z <- Z*Z1` if `withZ = true`.
The rank decisions use the absolute tolerance `tol` for the nonzero elements of `B`.

Reduce for `init = false`, the matrices `A` and `E` of the form

    A = [ *   *  *  ] roff     E = [ *   *  *  ] roff
        [ 0   B1 A1 ] n            [ 0   0  E1 ] n
         coff m  n                  coff m  n 

with `E1` upper triangular, using an orthogonal or unitary similarity transformations on the submatrices 
`A1`, `E1` and `B1` of the form `At1 =  Q1'*A1*Z1`, `Et1 =  Q1'*E1*Z1`, `Bt1 = Q1'*B1`, to the form
    
                                      [ *   *    *   *   ] roff
     At = diag(I,Q1')*A*diag(I,Z1) =  [ 0   B11  A11 A12 ] ρ
                                      [ 0   0    B2  A22 ] n-ρ
                                       coff m    ρ   n-ρ    

                                      [ *   *    *   *   ] roff
     Et = diag(I,Q1')*A*diag(I,Z1) =  [ 0   0    E11 E12 ] ρ
                                      [ 0   0    0   E22 ] n-ρ
                                       coff m    ρ   n-ρ    


where `B11` has full row rank `ρ`, and `E11` and `E22` are upper triangular. `At`, `Et` and `C*diag(I,Z1)` 
are returned in `A`, `E` and `C`, respectively, and `B` is unchanged. 
The performed orthogonal or unitary transformations are accumulated in `Q` as `Q <- Q*diag(I,Q1)` if `withQ = true`
and in `Z` as `Z <- Z*diag(I,Z1)` if `withZ = true`.
The rank decisions use the absolute tolerance `tol` for the nonzero elements of `A`.
"""
function _sreduceBAE!(n::Int,m::Int,A::AbstractMatrix{T},E::AbstractMatrix{T},B::AbstractVecOrMat{T},C::Union{AbstractMatrix{T},Missing},
                      Q::Union{AbstractMatrix{T},Nothing}, Z::Union{AbstractMatrix{T},Nothing}, tol::Real; 
                      fast::Bool = true, init::Bool = true, roff::Int = 0, coff::Int = 0, 
                      withQ::Bool = true, withZ::Bool = true) where {T}
   (m == 0 || n == 0) && (return 0)
   mA, nA = size(A) 
   ZERO = zero(T)
   ib = roff+1:roff+n
   ia = 1:roff+n
   if init 
      # coff and roff must be zero
      (coff == 0 && roff == 0) || error("coff and roff must be zero at first call")
      ja = 1:n
      B1 = view(B,ib,1:m)
      A1 = view(A,ib,ja)
      E1 = view(E,ib,ja)
   else
      (coff+m+n == nA && roff+n == mA) || error("coff and roff must have compatible values with the dimensions of A")
      ja = coff+m+1:coff+m+n
      B1 = view(A,ib,coff+1:coff+m)
      A1 = view(A,ib,ja)
      E1 = view(E,ib,ja)
   end
   if fast
      ρ = 0
      nrm = similar(real(A),m)
      jp = Vector(1:m)
      nm = min(n,m)
      for j = 1:nm
         for l = j:m
            nrm[l] = norm(B1[j:n,l])
         end
         nrmax, ind = findmax(nrm[j:m]) 
         ind += j-1
         if nrmax <= tol
            break
         else
            ρ += 1
         end
         if ind != j
            (jp[j], jp[ind]) = (jp[ind], jp[j])
            (B1[:,j],B1[:,ind]) = (B1[:,ind],B1[:,j])
         end
         for ii = n:-1:j+1
             iim1 = ii-1
             if B1[ii,j] != ZERO
                G, B1[iim1,j] = givens(B1[iim1,j],B1[ii,j],iim1,ii)
                B1[ii,j] = ZERO
                lmul!(G,view(B1,:,j+1:m))
                lmul!(G,A1)
                lmul!(G,view(E1,:,iim1:n))
                withQ && rmul!(view(Q,:,ib),G') 
                G, r = givens(conj(E1[ii,ii]),conj(E1[ii,iim1]),ii,iim1)
                E1[ii,ii] = conj(r)
                E1[ii,iim1] = ZERO 
                rmul!(view(E,1:roff+iim1,ja),G')
                withZ && rmul!(view(Z,:,ja),G') 
                rmul!(view(A,:,ja),G')
                ismissing(C) || rmul!(view(C,:,ja),G') 
             end
         end
      end
      B1[:,:] = [ B1[1:ρ,invperm(jp)]; zeros(T,n-ρ,m) ]
      return ρ
   else
      if n > m
         for j = 1:m
            for ii = n:-1:j+1
               iim1 = ii-1
               if B1[ii,j] != ZERO
                  G, B1[iim1,j] = givens(B1[iim1,j],B1[ii,j],iim1,ii)
                  B1[ii,j] = ZERO
                  lmul!(G,view(B1,:,j+1:m))
                  lmul!(G,A1)
                  lmul!(G,view(E1,:,iim1:n))
                  withQ && rmul!(view(Q,:,ib),G') 
                  G, r = givens(conj(E1[ii,ii]),conj(E1[ii,iim1]),ii,iim1)
                  E1[ii,ii] = conj(r)
                  E1[ii,iim1] = ZERO 
                  rmul!(view(E,1:roff+iim1,ja),G')
                  withZ && rmul!(view(Z,:,ja),G') 
                  rmul!(view(A,:,ja),G')
                  ismissing(C) || rmul!(view(C,:,ja),G') 
               end
            end
         end
      end
      mn = min(n,m)
      mn == 0 && (return 0)
      ics = 1:mn
      jcs = 1:m
      SVD = svd(B1[ics,jcs], full = true)
      ρ = count(x -> x > tol, SVD.S) 
      ρ == mn && (return ρ)
      B1[ics,jcs] = [ Diagonal(SVD.S[1:ρ])*SVD.Vt[1:ρ,:]; zeros(T,mn-ρ,m) ]
      ρ == 0 && (return ρ)
      ibt = roff+1:roff+mn
      withQ && (Q[:,ibt] = Q[:,ibt]*SVD.U)
      E[ibt,ja] = SVD.U'*E[ibt,ja]
      A[ibt,ja] = SVD.U'*A[ibt,ja]
      init ? (jt1 = 1:mn) : (jt1 = coff+m+1:coff+m+mn)
      E11 = view(E,ibt,jt1)
      if T <: BlasFloat
         tau = similar(E,mn)
         LinearAlgebra.LAPACK.gerqf!(E11,tau)
         T <: Complex ? tran = 'C' : tran = 'T'
         LinearAlgebra.LAPACK.ormrq!('R',tran,E11,tau,view(A,:,jt1))
         withZ && LinearAlgebra.LAPACK.ormrq!('R',tran,E11,tau,view(Z,:,jt1)) 
         LinearAlgebra.LAPACK.ormrq!('R',tran,E11,tau,view(E,1:roff,jt1))
         ismissing(C) || LinearAlgebra.LAPACK.ormrq!('R',tran,E11,tau,view(C,:,jt1)) 
         triu!(E11)
      else
         F = qr(reverse(E11,dims=1)')
         reverse!(rmul!(view(A,:,jt1),F.Q),dims=2)
         withZ && reverse!(rmul!(view(Z,:,jt1),F.Q),dims=2)
         reverse!(rmul!(view(E,1:roff,jt1),F.Q),dims=2)
         ismissing(C) || reverse!(rmul!(view(C,:,jt1),F.Q),dims=2)
         E11[:,:] = reverse(reverse(F.R,dims=1),dims=2)'
      end
   end
   return ρ 
end
"""
    _sreduceBAE2!(n::Int,m::Int,A::AbstractMatrix{T},E::AbstractMatrix{T},B::AbstractMatrix{T},
                  C::Union{AbstractMatrix{T},Missing}, Q::Union{AbstractMatrix{T},Nothing}, 
                  Z::Union{AbstractMatrix{T},Nothing}, tol::Real; 
                  fast = true, coff = 0, withQ = true, withZ = true) -> ρ 

Reduce the matrices `B`, `A` and the upper-triangular `E` of the form 


    B = [ *  *  ]       A - λE = [ *    *        A12-λE12 ] 
        [ B1 B2 ] n              [ *   A21-λE21  A22-λE22 ] n
          m  k               coff       n

using orthogonal or unitary transformations `Q1` and `Z1` such that `B <- G = diag(I,Q1')*B`, 
`G` has the form

         [  *   *  ]    
     G = [ B11 B12 ] ρ      ,
         [  0  B22 ] n-ρ    
            m  k                

where `B11` has full row rank `ρ`, and the blocks `A12-λE12`, `A21-λE21` and `A22-λE22` 
of `A-λE` are updated such that `A12-λE12 <- (A12-λE12)*Z1`, `A21-λE21 <- Q1'*(A21-λE21)` 
and `A22-λE22 <- Q1'*(A22-λE22)`. 
Update  `C <- C*diag(I,Q1)`, `Q <- Q*diag(I,Q1)` if `withQ = true` and  
`Z <- Z*diag(I,Z1)` if `withZ = true`. 
The rank decisions use the absolute tolerance `tol` for the nonzero elements of `B`.
"""
function _sreduceBAE2!(n::Int,m::Int,A::AbstractMatrix{T},E::AbstractMatrix{T},B::AbstractMatrix{T},C::Union{AbstractMatrix{T},Missing},
                      Q::Union{AbstractMatrix{T},Nothing}, Z::Union{AbstractMatrix{T},Nothing}, tol::Real; 
                      fast::Bool = true, coff::Int = 0, withQ::Bool = true, withZ::Bool = true) where {T}
   (m == 0 || n == 0) && (return 0)
   nA, mB = size(B) 
   roff = nA - n 
   ZERO = zero(T)
   ib = roff+1:nA    # row range of [B1, B2] and [A21 A22]
   ia = 1:nA         # row range of [A12; A22]
   ja = ib           # column rabge of A2 = [A12; A22]
   ja1 = coff+1:nA   # column range of A1 = [A21 A22]
   B1 = view(B,ib,1:m)
   B2mat = (m < mB)  # B2 is present if B has more columns than m
   B2mat && (B2 = view(B,ib,m+1:mB))
   A1 = view(A,ib,ja1)  
   #E1 = view(E,ib,ja1)  
   E1 = view(E,ib,ib)
   A2 = view(A,ia,ja)
   E2 = view(E,ia,ja)
   if fast
      ρ = 0
      nrm = similar(real(A),m)
      jp = Vector(1:m)
      nm = min(n,m)
      for j = 1:nm
         for l = j:m
            nrm[l] = norm(B1[j:n,l])
         end
         nrmax, ind = findmax(nrm[j:m]) 
         ind += j-1
         if nrmax <= tol
            break
         else
            ρ += 1
         end
         if ind != j
            (jp[j], jp[ind]) = (jp[ind], jp[j])
            (B1[:,j],B1[:,ind]) = (B1[:,ind],B1[:,j])
         end
         for ii = n:-1:j+1
             iim1 = ii-1
             if B1[ii,j] != ZERO
                G, B1[iim1,j] = givens(B1[iim1,j],B1[ii,j],iim1,ii)
                B1[ii,j] = ZERO
                lmul!(G,view(B1,:,j+1:m))
                lmul!(G,A1)
                B2mat && lmul!(G,B2)
                lmul!(G,view(E,ib,roff+iim1:nA))
                withQ && rmul!(view(Q,:,ib),G') 
                G, r = givens(conj(E1[ii,ii]),conj(E1[ii,iim1]),ii,iim1)
                E1[ii,ii] = conj(r)
                E1[ii,iim1] = ZERO 
                rmul!(view(E,1:roff+iim1,ja),G')
                withZ && rmul!(view(Z,:,ja),G') 
                rmul!(view(A,:,ja),G')
                ismissing(C) || rmul!(view(C,:,ja),G') 
             end
         end
      end
      B1[:,:] = [ B1[1:ρ,invperm(jp)]; zeros(T,n-ρ,m) ]
      return ρ
   else
      if n > m
         for j = 1:m
            for ii = n:-1:j+1
               iim1 = ii-1
               if B1[ii,j] != ZERO
                  G, B1[iim1,j] = givens(B1[iim1,j],B1[ii,j],iim1,ii)
                  B1[ii,j] = ZERO
                  lmul!(G,view(B1,:,j+1:m))
                  lmul!(G,A1)
                  B2mat && lmul!(G,B2)
                  lmul!(G,view(E,ib,roff+iim1:nA))
                  withQ && rmul!(view(Q,:,ib),G') 
                  G, r = givens(conj(E1[ii,ii]),conj(E1[ii,iim1]),ii,iim1)
                  E1[ii,ii] = conj(r)
                  E1[ii,iim1] = ZERO 
                  rmul!(view(E,1:roff+iim1,ja),G')
                  withZ && rmul!(view(Z,:,ja),G') 
                  rmul!(view(A,:,ja),G')
                  ismissing(C) || rmul!(view(C,:,ja),G') 
               end
            end
         end
      end
      mn = min(n,m)
      mn == 0 && (return 0)
      ics = 1:mn
      jcs = 1:m
      SVD = svd(B1[ics,jcs], full = true)
      ρ = count(x -> x > tol, SVD.S) 
      ρ == mn && (return ρ)
      B1[ics,jcs] = [ Diagonal(SVD.S[1:ρ])*SVD.Vt[1:ρ,:]; zeros(T,mn-ρ,m) ]
      ρ == 0 && (return ρ)
      ibt = roff+1:roff+mn
      withQ && (Q[:,ibt] = Q[:,ibt]*SVD.U)
      E[ibt,ja1] = SVD.U'*view(E,ibt,ja1)
      A[ibt,ja1] = SVD.U'*view(A,ibt,ja1)
      B2mat &&  (B2[ibt,:] = SVD.U'*B2[ibt,:])  
      # tau = similar(E,mn)
      jt1 = coff+1:coff+mn
      E11 = view(E,ibt,jt1)
      # LinearAlgebra.LAPACK.gerqf!(E11,tau)
      # T <: Complex ? tran = 'C' : tran = 'T'
      # LinearAlgebra.LAPACK.ormrq!('R',tran,E11,tau,view(A,:,jt1))
      # withZ && LinearAlgebra.LAPACK.ormrq!('R',tran,E11,tau,view(Z,:,jt1)) 
      # LinearAlgebra.LAPACK.ormrq!('R',tran,E11,tau,view(E,1:roff,jt1))
      # ismissing(C) || LinearAlgebra.LAPACK.ormrq!('R',tran,E11,tau,view(C,:,jt1)) 
      # triu!(E11)
      if T <: BlasFloat
         tau = similar(E,mn)
         LinearAlgebra.LAPACK.gerqf!(E11,tau)
         T <: Complex ? tran = 'C' : tran = 'T'
         LinearAlgebra.LAPACK.ormrq!('R',tran,E11,tau,view(A,:,jt1))
         withZ && LinearAlgebra.LAPACK.ormrq!('R',tran,E11,tau,view(Z,:,jt1)) 
         LinearAlgebra.LAPACK.ormrq!('R',tran,E11,tau,view(E,1:roff,jt1))
         ismissing(C) || LinearAlgebra.LAPACK.ormrq!('R',tran,E11,tau,view(C,:,jt1)) 
         triu!(E11)
      else
         F = qr(reverse(E11,dims=1)')
         reverse!(rmul!(view(A,:,jt1),F.Q),dims=2)
         withZ && reverse!(rmul!(view(Z,:,jt1),F.Q),dims=2)
         reverse!(rmul!(view(E,1:roff,jt1),F.Q),dims=2)
         ismissing(C) || reverse!(rmul!(view(C,:,jt1),F.Q),dims=2)
         E11[:,:] = reverse(reverse(F.R,dims=1),dims=2)'
      end
   end
   return ρ 
end
"""
    _sreduceAEC!(n::Int,p::Int,A::AbstractMatrix{T},E::AbstractMatrix{T},C::AbstractMatrix{T},B::Union{AbstractMatrix{T},Missing},
                Q::Union{AbstractMatrix{T},Nothing}, Z::Union{AbstractMatrix{T},Nothing}, tol::Real; 
                fast = true, init = true, rtrail = 0, ctrail = 0, withQ = true, withZ = true) -> ρ

Reduce for `init = true`, the pair `(A-λE,C)`, with E upper-triangular, using an orthogonal or unitary 
similarity transformations on the matrices `A`, `E` and `C` of the form `At =  Q1'*A*Z1`, `Et =  Q1'*E*Z1`, 
`Ct = C*Z1`, to the form

     Ct = [ 0  C11 ] p   At = [ A11 A12 ] n-ρ   Et = [ E11 E12 ] n-ρ  
           n-ρ  ρ             [ C2  A22 ] ρ          [  0  E22 ] ρ   
                                n-ρ  ρ                 n-ρ  ρ 

where `C11` has full column rank `ρ` and `Et` is upper-triangular. `Ct`, `At`, `Et` and `Q1'*B` are returned in 
`C`, `A`, `E` and `B`, respectively. 
The performed orthogonal or unitary transformations are accumulated in `Q` as `Q <- Q*Q1` if `withQ = true`
and in `Z` as `Z <- Z*Z1` if `withZ = true`.
The rank decisions use the absolute tolerance `tol` for the nonzero elements of `C`.

Reduce for `init = false`, the matrices `A` and `E` of the form

        [ A1  *   ] n               [ E1  *   ] n
    A = [ C1  *   ] p          E =  [ 0   *   ] p
        [ 0   *   ] rtrail          [ 0   *   ] rtrail
          n ctrail 

with `E1` upper triangular, using an orthogonal or unitary similarity transformations on the submatrices 
`A1`, `E1` and `C1` of the form `At1 =  Q1'*A1*Z1`, `Et1 =  Q1'*E1*Z1`, `Ct1 = C1*Z1`, to the form


                                      [ A11 A12  *   ] n-ρ
                                      [ C2  A22  *   ] ρ
     At = diag(Q1',I)*A*diag(Q1,I) =  [ 0   C11  *   ] p
                                      [ 0   0    *   ] rtrail
                                       n-ρ  ρ  ctrail    

                                      [ E11 E12  *   ] n-ρ
                                      [ 0   E22  *   ] ρ
     Et = diag(Q1',I)*A*diag(Q1,I) =  [ 0   0    *   ] p
                                      [ 0   0    *   ] rtrail
                                       n-ρ  ρ  ctrail    
 
where `C11` has full column rank `ρ`, and `E11` and `E22` are upper triangular. `At`, `Et` and `diag(I,Q1')*B` 
are returned in `A`, `E` and `B`, respectively, and `C` is unchanged. 
The performed orthogonal or unitary transformations are accumulated in `Q` as `Q <- Q*diag(I,Q1)` if `withQ = true`
and in `Z` as `Z <- Z*diag(I,Z1)` if `withZ = true`.
The rank decisions use the absolute tolerance `tol` for the nonzero elements of `A`.
"""
function _sreduceAEC!(n::Int,p::Int,A::AbstractMatrix{T},E::AbstractMatrix{T},C::AbstractMatrix{T},B::Union{AbstractVecOrMat{T},Missing},
                      Q::Union{AbstractMatrix{T},Nothing}, Z::Union{AbstractMatrix{T},Nothing}, tol::Real; 
                      fast::Bool = true, init::Bool = true, rtrail::Int = 0, ctrail::Int = 0, 
                      withQ::Bool = true, withZ::Bool = true) where {T}
   (p == 0 || n == 0) && (return 0)
   mA, nA = size(A) 
   ZERO = zero(T)
   ia = 1:n
   ja = 1:nA
   A1 = view(A,ia,ia)
   E1 = view(E,ia,ia)
   if init 
      # coff and roff must be zero
      (ctrail == 0 && rtrail == 0) || error("rtrail and ctrail must be zero at first call")
      C1 = view(C,1:p,ia)
   else
      (ctrail+n == nA && rtrail+n+p == mA) || error("rtrail and ctrail must have compatible values with the dimensions of A")
      C1 = view(A,n+1:n+p,ia)
   end
   if fast
      ρ = 0
      nrm = similar(real(A),p)
      jp = Vector(1:p)
      np = min(n,p)
      for i = 1:np
         ii = p-i+1
         for l = 1:ii
            nrm[l] = norm(C1[l,1:n-i+1])
         end
         nrmax, ind = findmax(nrm[1:ii]) 
         if nrmax <= tol
            break
         else
            ρ += 1
         end
         if ind != ii
            (jp[ii], jp[ind]) = (jp[ind], jp[ii])
            (C1[ii,:],C1[ind,:]) = (C1[ind,:],C1[ii,:])
         end
         for jj = 1:n-i
             jjp1 = jj+1
             if C1[ii,jj] != ZERO
                G, r = givens(conj(C1[ii,jjp1]),conj(C1[ii,jj]),jjp1,jj)
                C1[ii,jjp1] = conj(r)
                C1[ii,jj] = ZERO
                rmul!(view(C1,1:ii-1,:),G')
                rmul!(A1,G')
                rmul!(view(E1,1:jjp1,ia),G')
                withZ && rmul!(view(Z,:,ia),G') 
                G, E[jj,jj] = givens(E[jj,jj],E[jjp1,jj],jj,jjp1)
                E[jjp1,jj] = ZERO
                lmul!(G,view(E,ia,jjp1:nA))
                withQ && rmul!(view(Q,:,ia),G') 
                lmul!(G,view(A,ia,ja))
                ismissing(B) || lmul!(G,view(B,ia,:))
             end
         end
      end
      C1[:,1:n] = [ zeros(T,p,n-ρ)  C1[invperm(jp),n-ρ+1:n]];
   else
      if n > p
         for i = 1:p
             ii = p-i+1
             for jj = 1:n-i
                jjp1 = jj+1
                if C1[ii,jj] != ZERO
                   G, r = givens(conj(C1[ii,jjp1]),conj(C1[ii,jj]),jjp1,jj)
                   C1[ii,jjp1] = conj(r)
                   C1[ii,jj] = ZERO
                   rmul!(view(C1,1:ii-1,:),G')
                   rmul!(A1,G')
                   rmul!(view(E,1:jjp1,ia),G')
                   withZ && rmul!(view(Z,:,ia),G') 
                   G, E[jj,jj] = givens(E[jj,jj],E[jjp1,jj],jj,jjp1)
                   E[jjp1,jj] = ZERO
                   lmul!(G,view(E,ia,jjp1:nA))
                   withQ && rmul!(view(Q,:,ia),G') 
                   lmul!(G,view(A,ia,ja))
                   ismissing(B) || lmul!(G,view(B,ia,:))
                end
             end
         end
      end
      pn = min(n,p)
      pn == 0 && (return 0)
      ics = 1:p
      jcs = n-pn+1:n
      SVD = svd(C1[ics,jcs], full = true)
      ρ = count(x -> x > tol, SVD.S) 
      ρ == pn && (return ρ)
      Q1 = reverse(SVD.U,dims=2)
      C1[ics,jcs] = [ zeros(T,p,pn-ρ) Q1[:,p-ρ+1:end]*Diagonal(reverse(SVD.S[1:ρ])) ] 
      ρ == 0 && (return ρ)
      Z1 = reverse(SVD.V,dims=2)
      jt = n-pn+1:n
      withZ && (Z[:,jt] = Z[:,jt]*Z1) 
      A[ia,jt] = A[ia,jt]*Z1
      E[ia,jt] = E[ia,jt]*Z1    # more efficient computation possible
      jt1 = n+1:nA
      E22 = view(E,jt,jt)
      if T <: BlasFloat
         tau = similar(E,pn)
         LinearAlgebra.LAPACK.geqrf!(E22,tau)
         T <: Complex ? tran = 'C' : tran = 'T'
         LinearAlgebra.LAPACK.ormqr!('L',tran,E22,tau,view(A,jt,ja))
         withQ && LinearAlgebra.LAPACK.ormqr!('R','N',E22,tau,view(Q,:,jt)) 
         LinearAlgebra.LAPACK.ormqr!('L',tran,E22,tau,view(E,jt,jt1))
         ismissing(B) || LinearAlgebra.LAPACK.ormqr!('L',tran,E22,tau,view(B,jt,:))
      else
         F = qr!(E22)
         lmul!(F.Q',view(A,jt,ja))
         withQ && lmul!(view(Q,:,jt),F.Q) 
         lmul!(F.Q',view(E,jt,jt1))
         ismissing(B) || lmul!(F.Q',view(B,jt,:))
      end
      triu!(E22)
   end
   return ρ 
end
