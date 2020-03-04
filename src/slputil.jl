"""
    _sreduceC!(A::AbstractMatrix,E::AbstractMatrix,C::AbstractMatrix,Z::Union{AbstractMatrix,Nothing}, tol; 
                fast = true, withZ = true)

Reduce the `p x n` matrix `C` using an orthogonal or unitary similarity transformation `Z1` to the column 
compressed form 

     CT = C*Z1 = [ 0  C11 ] p
                  n-ρ  ρ

where `C11` has full column rank `ρ`. `A*Z1`, `E*Z1` and `CT` are returned in `A`, `E` and `C`, respectively. 
The performed orthogonal or unitary transformations are accumulated in `Z` if `withZ = true`. 
The rank decisions use the absolute tolerance `tol` for the nonzero elements of `C`.
"""
function _sreduceC!(A::AbstractMatrix,E::AbstractMatrix,C::AbstractMatrix,Z::Union{AbstractMatrix,Nothing}, tol; 
                    fast = true, withZ = true)
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
   end
   return ρ 
end

"""
    _sreduceB!(A::AbstractMatrix,E::AbstractMatrix,B::AbstractMatrix,Q::Union{AbstractMatrix,Nothing}, tol; 
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
function _sreduceB!(A::AbstractMatrix,E::AbstractMatrix,B::AbstractMatrix,Q::Union{AbstractMatrix,Nothing}, tol; 
                    fast = true, withQ = true)
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
    _sreduceBA!(n::Int,m::Int,A::AbstractMatrix,B::AbstractMatrix,C::Union{AbstractMatrix,Missing},Q::Union{AbstractMatrix,Nothing}, tol; 
                fast = true, init = true, roff = 0, coff = 0, withQ = true)

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
function _sreduceBA!(n::Int,m::Int,A::AbstractMatrix,B::AbstractMatrix,C::Union{AbstractMatrix,Missing},Q::Union{AbstractMatrix,Nothing}, tol; 
                    fast = true, init = true, roff = 0, coff = 0, withQ = true)
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
      jt = coff+m+1:coff+m+mn
      withQ && (Q[:,ibt] = Q[:,ibt]*SVD.U)
      A1[ibt,ja] = SVD.U'*A1[ibt,ja]
      A1[ia,jt] = A1[ia,jt]*SVD.U
   end
   return ρ 
end
"""
    _sreduceAC!(n::Int,p::Int,A::AbstractMatrix,C::AbstractMatrix,B::Union{AbstractMatrix,Missing},Q::Union{AbstractMatrix,Nothing}, tol; 
                fast = true, init = true, rtrail = 0, ctrail = 0, withQ = true)

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
function _sreduceAC!(n::Int,p::Int,A::AbstractMatrix,C::AbstractMatrix,B::Union{AbstractMatrix,Missing},Q::Union{AbstractMatrix,Nothing}, tol; 
                    fast = true, init = true, rtrail = 0, ctrail = 0, withQ = true)
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
   end
   return ρ 
end
