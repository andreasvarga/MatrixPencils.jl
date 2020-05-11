
"""
    _sreduceB!(A::AbstractMatrix{T1},E::AbstractMatrix{T1},B::AbstractMatrix{T1},Q::Union{AbstractMatrix{T1},Nothing}, tol::Real; 
                fast = true, withQ = true)

Reduce the `n x m` matrix `B` using an orthogonal or unitary similarity transformation `Q1` to the row 
compressed form 

     BT = Q1'*B = [ B11 ] ρ
                  [  0  ] n-ρ
                     m      

where `B11` has full row rank `ρ`. `Q1'*A`, `Q1'*E` and `BT` are returned in `A`, `E` and `B`, respectively. 
The performed orthogonal or unitary transformations are accumulated in `Q` if `withQ = true`. 
The rank decisions use the absolute tolerance `tol` for the nonzero elements of `B`.
"""
function _sreduceB!(A::AbstractMatrix{T1},E::AbstractMatrix{T1},B::AbstractMatrix{T1},
                    Q::Union{AbstractMatrix{T1},Nothing}, tol::Real; 
                    fast::Bool = true, withQ::Bool = true) where T1 <: BlasFloat
   n, m = size(B)                
   (m == 0 || n == 0) && (return 0)
   mA, nA = size(A) 
   T = eltype(A)
   ZERO = zero(T)
   if m == 1 
      b = view(B,:,1)
      if n == 1
         abs(b[1]) > tol ? (return 1) : (b[1] = ZERO; return 0)
      else
         τ, β = larfg!(b)
         if abs(β) <= tol
            b[:] = zeros(T,n)
            return 0
         else
            larf!('L', b, conj(τ), A)  
            larf!('L', b, conj(τ), E)  
            withQ && larf!('R', b, τ, Q) 
            b[:] = [ β; zeros(T,n-1)] 
            return 1
         end
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
      if ρ == mn
         return ρ
      else
         B[ics,jcs] = [ Diagonal(SVD.S[1:ρ])*SVD.Vt[1:ρ,:]; zeros(T,mn-ρ,m) ]
         if ρ == 0
            return ρ
         end
      end
      withQ && (Q[:,ics] = Q[:,ics]*SVD.U)
      A[ics,:] = SVD.U'*A[ics,:]
      E[ics,:] = SVD.U'*E[ics,:]
   end
   return ρ 
end
"""
    _sreduceC!(A::AbstractMatrix{T1},E::AbstractMatrix{T1},C::AbstractMatrix{T1},Z::Union{AbstractMatrix{T1},Nothing}, tol::Real; 
                fast = true, withZ = true)

Reduce the `p x n` matrix `C` using an orthogonal or unitary similarity transformation `Z1` to the column 
compressed form 

     CT = C*Z1 = [ 0  C11 ] p
                  n-ρ  ρ

where `C11` has full column rank `ρ`. `A*Z1`, `E*Z1` and `CT` are returned in `A`, `E` and `C`, respectively. 
The performed orthogonal or unitary transformations are accumulated in `Z` if `withZ = true`. 
The rank decisions use the absolute tolerance `tol` for the nonzero elements of `C`.
"""
function _sreduceC!(A::AbstractMatrix{T1},E::AbstractMatrix{T1},C::AbstractMatrix{T1},
                    Z::Union{AbstractMatrix{T1},Nothing}, tol::Real; 
                    fast::Bool = true, withZ::Bool = true) where T1 <: BlasFloat
   p, n = size(C)                
   (p == 0 || n == 0) && (return 0)
   mA, nA = size(A) 
   T = eltype(A)
   ZERO = zero(T)
   ia = 1:n
   if p == 1 
      c = view(C,1,:)
      if n == 1
         abs(c[1]) > tol ? (return 1) : (c[1] = ZERO; return 0)
      else
         τ, β = larfgl!(c)
         if abs(β) < tol
            c[:] = zeros(T,n)
            return 0
         else
            T <: Complex && (c[:] = conj(c))
            τ = conj(τ)
            larf!('R', c, τ, A)  
            larf!('R', c, τ, E)  
            withZ && larf!('R', c, τ, Z) 
            c[:] = [zeros(T,n-1); β]; 
            return 1
         end
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
      if ρ == pn
         return ρ
      else
         C[ics,jcs] = [ zeros(T,p,pn-ρ) reverse(SVD.U,dims=2)[:,p-ρ+1:end]*Diagonal(reverse(SVD.S[1:ρ])) ]
         if ρ == 0
            return ρ
         end
      end
      Z1 = reverse(SVD.V,dims=2)
      withZ && (Z[:,jcs] = Z[:,jcs]*Z1)
      A[:,jcs] = A[:,jcs]*Z1
      E[:,jcs] = E[:,jcs]*Z1
   end
   return ρ 
end
"""
    _sreduceBA!(n::Int,m::Int,A::AbstractMatrix{T1},B::AbstractMatrix{T1},C::Union{AbstractMatrix{T1},Missing},Q::Union{AbstractMatrix{T1},Nothing}, tol::Real; 
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
function _sreduceBA!(n::Int,m::Int,A::AbstractMatrix{T1},B::AbstractMatrix{T1},C::Union{AbstractMatrix{T1},Missing},
                     Q::Union{AbstractMatrix{T1},Nothing}, tol::Real; 
                     fast::Bool = true, init::Bool = true, roff::Int = 0, coff::Int = 0, withQ::Bool = true) where T1 <: BlasFloat
   (m == 0 || n == 0) && (return 0)
   mA, nA = size(A) 
   T = eltype(A)
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
      if n == 1
         abs(b[1]) > tol ? (return 1) : (b[1] = ZERO; return 0)
      else
         τ, β = larfg!(b)
         if abs(β) < tol
            b[:] = zeros(T,n)
            return 0
         else
            #T <: Complex && (b[:] = conj(b))
            larf!('L', b, conj(τ), A1)  
            larf!('R', b, τ, view(A, ia, ja))  
            ismissing(C) || larf!('R', b, τ, view(C,:,ja))  
            withQ && larf!('R', b, τ, view(Q,:,ib)) 
            b[:] = [ β; zeros(T,n-1)] 
            return 1
         end
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
      if ρ == mn
         return ρ
      else
         B1[ics,jcs] = [ Diagonal(SVD.S[1:ρ])*SVD.Vt[1:ρ,:]; zeros(T,mn-ρ,m) ]
         if ρ == 0
            return ρ
         end
      end
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
    _sreduceAC!(n::Int,p::Int,A::AbstractMatrix{T1},C::AbstractMatrix{T1},B::Union{AbstractMatrix{T1},Missing},
                Q::Union{AbstractMatrix{T1},Nothing}, tol::Real; 
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
function _sreduceAC!(n::Int,p::Int,A::AbstractMatrix{T1},C::AbstractMatrix{T1},B::Union{AbstractMatrix{T1},Missing},
                     Q::Union{AbstractMatrix{T1},Nothing}, tol::Real; 
                     fast::Bool = true, init::Bool = true, rtrail::Int = 0, ctrail::Int = 0, withQ::Bool = true) where T1 <: BlasFloat
   (p == 0 || n == 0) && (return 0)
   mA, nA = size(A) 
   T = eltype(A)
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
      if n == 1
         abs(c[1]) > tol ? (return 1) : (c[1] = ZERO; return 0)
      else
         τ, β = larfgl!(c)
         if abs(β) < tol
            c[:] = zeros(T,n)
            return 0
         else
            T <: Complex && (c[:] = conj(c))
            larf!('L', c, τ, view(A, ia, :))  
            ismissing(B) || larf!('L', c, τ, view(B, ia, :)) 
            τ = conj(τ)
            larf!('R', c, τ, A1)  
            withQ && larf!('R', c, τ, view(Q,:,ia)) 
            c[:] = [zeros(T,n-1); β]; 
            return 1
         end
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
         larf!('R',v, conj(τ[i]), view(A1,:,it))
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
      if ρ == pn
         return ρ
      else
         C1[ics,jcs] = [ zeros(T,p,pn-ρ) reverse(SVD.U,dims=2)[:,p-ρ+1:end]*Diagonal(reverse(SVD.S[1:ρ])) ]
         if ρ == 0
            return ρ
         end
      end
      Q1 = reverse(SVD.V,dims=2)
      withQ && (Q[:,jcs] = Q[:,jcs]*Q1)
      A1[jcs,1:n+ctrail] = Q1'*A1[jcs,1:n+ctrail]
      A1[ia,jcs] = A1[ia,jcs]*Q1
      ismissing(B) || (B[jcs,:] = Q1'*B[jcs,:])
   end
   return ρ 
end
"""
    _sreduceBAE!(n::Int,m::Int,A::AbstractMatrix{T1},E::AbstractMatrix{T1},B::AbstractMatrix{T1},C::Union{AbstractMatrix{T1},Missing},
                 Q::Union{AbstractMatrix{T1},Nothing}, Z::Union{AbstractMatrix{T1},Nothing}, tol::Real; 
                 fast = true, init = true, roff = 0, coff = 0, withQ = true, withZ = true)

Reduce for `init = true`, the pair `(A-λE,B)`, with E upper-triangular, using an orthogonal or unitary 
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
function _sreduceBAE!(n::Int,m::Int,A::AbstractMatrix{T1},E::AbstractMatrix{T1},B::AbstractMatrix{T1},C::Union{AbstractMatrix{T1},Missing},
                      Q::Union{AbstractMatrix{T1},Nothing}, Z::Union{AbstractMatrix{T1},Nothing}, tol::Real; 
                      fast::Bool = true, init::Bool = true, roff::Int = 0, coff::Int = 0, 
                      withQ::Bool = true, withZ::Bool = true) where T1 <: BlasFloat
   (m == 0 || n == 0) && (return 0)
   mA, nA = size(A) 
   T = eltype(A)
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
         if nrmax < tol
            break
         else
            ρ += 1
         end
         if ind !== j
            (jp[j], jp[ind]) = (jp[ind], jp[j])
            (B1[:,j],B1[:,ind]) = (B1[:,ind],B1[:,j])
         end
         for ii = n:-1:j+1
             iim1 = ii-1
             if B1[ii,j] !== ZERO
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
               if B1[ii,j] !== ZERO
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
      if mn > 0
         ics = 1:mn
         jcs = 1:m
         SVD = svd(B1[ics,jcs], full = true)
         ρ = count(x -> x > tol, SVD.S) 
         if ρ == mn
            return ρ
         else
            B1[ics,jcs] = [ Diagonal(SVD.S[1:ρ])*SVD.Vt[1:ρ,:]; zeros(T,mn-ρ,m) ]
            if ρ == 0
               return ρ
            end
         end
         ibt = roff+1:roff+mn
         jt = coff+m+1:nA
         withQ && (Q[:,ibt] = Q[:,ibt]*SVD.U)
         E[ibt,jt] = SVD.U'*E[ibt,jt]
         A[ibt,jt] = SVD.U'*A[ibt,jt]
         tau = similar(E,mn)
         jt1 = coff+m+1:coff+m+mn
         E11 = view(E,ibt,jt1)
         LinearAlgebra.LAPACK.gerqf!(E11,tau)
         eltype(A) <: Complex ? tran = 'C' : tran = 'T'
         LinearAlgebra.LAPACK.ormrq!('R',tran,E11,tau,view(A,:,jt1))
         withZ && LinearAlgebra.LAPACK.ormrq!('R',tran,E11,tau,view(Z,:,jt1)) 
         LinearAlgebra.LAPACK.ormrq!('R',tran,E11,tau,view(E,1:roff,jt1))
         ismissing(C) || LinearAlgebra.LAPACK.ormrq!('R',tran,E11,tau,view(C,:,jt1)) 
         triu!(E11)
      else
         ρ = 0
      end
   end
   return ρ 
end
"""
    _sreduceAEC!(n::Int,p::Int,A::AbstractMatrix{T1},E::AbstractMatrix{T1},C::AbstractMatrix{T1},B::Union{AbstractMatrix{T1},Missing},
                Q::Union{AbstractMatrix{T1},Nothing}, Z::Union{AbstractMatrix{T1},Nothing}, tol::Real; 
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
function _sreduceAEC!(n::Int,p::Int,A::AbstractMatrix{T1},E::AbstractMatrix{T1},C::AbstractMatrix{T1},B::Union{AbstractMatrix{T1},Missing},
                      Q::Union{AbstractMatrix{T1},Nothing}, Z::Union{AbstractMatrix{T1},Nothing}, tol::Real; 
                      fast::Bool = true, init::Bool = true, rtrail::Int = 0, ctrail::Int = 0, 
                      withQ::Bool = true, withZ::Bool = true) where T1 <: BlasFloat
   (p == 0 || n == 0) && (return 0)
   mA, nA = size(A) 
   T = eltype(A)
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
         if nrmax < tol
            break
         else
            ρ += 1
         end
         if ind !== ii
            (jp[ii], jp[ind]) = (jp[ind], jp[ii])
            (C1[ii,:],C1[ind,:]) = (C1[ind,:],C1[ii,:])
         end
         for jj = 1:n-i
             jjp1 = jj+1
             if C1[ii,jj] !== ZERO
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
                if C1[ii,jj] !== ZERO
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
      if pn > 0
         ics = 1:p
         jcs = n-pn+1:n
         SVD = svd(C1[ics,jcs], full = true)
         ρ = count(x -> x > tol, SVD.S) 
         if ρ == pn
            return ρ
         else
            Q1 = reverse(SVD.U,dims=2)
            C1[ics,jcs] = [ zeros(T,p,pn-ρ) Q1[:,p-ρ+1:end]*Diagonal(reverse(SVD.S[1:ρ])) ] 
            if ρ == 0
               return ρ
            end
         end
         Z1 = reverse(SVD.V,dims=2)
         jt = n-pn+1:n
         withZ && (Z[:,jt] = Z[:,jt]*Z1) 
         A[ia,jt] = A[ia,jt]*Z1
         E[ia,jt] = E[ia,jt]*Z1    # more efficient computation possible
         jt1 = n+1:nA
         tau = similar(E,pn)
         E22 = view(E,jt,jt)
         LinearAlgebra.LAPACK.geqrf!(E22,tau)
         eltype(A) <: Complex ? tran = 'C' : tran = 'T'
         LinearAlgebra.LAPACK.ormqr!('L',tran,E22,tau,view(A,jt,ja))
         withQ && LinearAlgebra.LAPACK.ormqr!('R','N',E22,tau,view(Q,:,jt)) 
         LinearAlgebra.LAPACK.ormqr!('L',tran,E22,tau,view(E,jt,jt1))
         ismissing(B) || LinearAlgebra.LAPACK.ormqr!('L',tran,E22,tau,view(B,jt,:))
         triu!(E22)
      else
         ρ = 0
      end
   end
   return ρ 
end
