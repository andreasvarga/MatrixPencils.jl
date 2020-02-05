"""
    _preduceBF!(M, N; fast = true, atol = 0, rtol, roff = 0, coff = 0, rtrail = 0, ctrail = 0, 
                withQ = true, withZ = true) -> Q, Z, n, m, p

Reduce the partitioned linear pencil `M - λN` 

              [   *         *         *  ] roff
     M - λN = [   0      M22-λN22     *  ] npp
              [   0         0         *  ] rtrail
                coff       npm     ctrail
  
to an equivalent basic form `F - λG = Q'(M - λN)Z` using orthogonal transformation matrices `Q` and `Z` 
such that the subpencil `M22 - λN22` is transformed into the following standard form
 
                       | B  | A-λE | 
          F22 - λG22 = |----|------| ,        
                       | D  |  C   |
where `E` is an `nxn` non-singular matrix, and `A`, `B`, `C`, `D` are `nxn`-, `nxm`-, `pxn`- and `pxm`-dimensional matrices,
respectively. The order `n` of `E` is equal to the numerical rank of `N` determined using the absolute tolerance `atol` and 
relative tolerance `rtol`. `M` and `N` are overwritten by `F` and `G`, respectively. 

If `fast = true`, `E` is determined upper triangular using a rank revealing QR-decomposition with column pivoting of `N22` 
and `n` is evaluated as the number of nonzero diagonal elements of the R factor, whose magnitudes are greater than 
`tol = max(atol,abs(R[1,1])*rtol)`. 
If `fast = false`,  `E` is determined diagonal using a rank revealing SVD-decomposition of `N22` and 
`n` is evaluated as the number of singular values greater than `tol = max(atol,smax*rtol)`, where `smax` 
is the largest singular value. 
The rank decision based on the SVD-decomposition is generally more reliable, but the involved computational effort is higher.
"""
function _preduceBF!(M::AbstractMatrix{T1}, N::AbstractMatrix{T1}; atol::Real = zero(eltype(M)),  
                    rtol::Real = (min(size(M)...)*eps(real(float(one(eltype(M))))))*iszero(atol), 
                    fast = true, roff = 0, coff = 0, rtrail = 0, ctrail = 0, 
                    withQ = true, withZ = true) where T1 <: BlasFloat
   mM, nM = size(M)
   (mM,nM) == size(N) || throw(DimensionMismatch("M and N must have the same dimensions"))
   T = eltype(M)
   npp = mM-rtrail-roff
   npm = nM-ctrail-coff
   # assume the partitioned form
   #
   #              [M11-λN11  M12-λN12  M13-λN13] roff
   #     M - λN = [   0      M22-λN22  M23-λM23] npp
   #              [   0         0      M33-λN33] rtrail
   #                coff       npm      ctrail
   #
   # where M22 and N22 are npp x npm matrices


   # fast returns for null dimensions
   if npp == 0 && npm == 0
      return withQ ? zeros(T,0,0) : nothing, withZ ? zeros(T,0,0) : nothing, 0, 0, 0
   elseif npp == 0
      return withQ ? zeros(T,0,0) : nothing, withZ ? Matrix{T}(I(nM)) : nothing, 0, npm, 0
   elseif npm == 0
      return withQ ? Matrix{T}(I(mM)) : nothing, withZ ? zeros(T,0,0) : nothing,  0, 0, npp
   end
   """
   Step 0: Reduce M22 -λ N22 to the standard form
   """
   i11 = 1:roff
   i22 = roff+1:roff+npp
   j22 = coff+1:coff+npm
   i12 = 1:roff+npp
   j23 = coff+1:nM
   jN23 = coff+npm+1:nM
   eltype(M) <: Complex ? tran = 'C' : tran = 'T'
   if fast
      # compute the complete orthogonal decomposition of N22 using the rank revealing QR-factorization
      QR = qr!(view(N,i22,j22), Val(true))
      tol = max(atol, rtol*abs(QR.R[1,1]))
      n = count(x -> x > tol, abs.(diag(QR.R))) 
      m = npm-n
      p = npp-n
      i1 = 1:n
      R, tau = LinearAlgebra.LAPACK.tzrzf!(QR.R[i1,:])
      jp = [coff+n+1:coff+npm; coff+1:coff+n] 
      # [M12; M22] <- [M12; M22]*P*Z*Pc
      M[i12,j22] = M[i12,j22[QR.p]]
      M[i12,j22] = LinearAlgebra.LAPACK.ormrz!('R',tran,R[i1,:],tau,M[i12,j22])
      M[i12,j22] = M[i12,jp]
      # [M22 M23] <- QR.Q'*[M22 M23]
      M[i22,j23] = QR.Q'*M[i22,j23]
      # N12 <- N12*P*Z*Pc
      N[i11,j22] = N[i11,j22[QR.p]]
      N[i11,j22] = LinearAlgebra.LAPACK.ormrz!('R',tran,R[i1,:],tau,N[i11,j22])
      N[i11,j22] = N[i11,jp]
      # N23 <- QR.Q'*N23
      N[i22,jN23] = QR.Q'*N[i22,jN23]
      if withQ
         Q = Matrix{T}(I(mM))
         Q[i22,i22] = QR.Q*Q[i22,i22]
      else
         Q = nothing
      end
      if withZ
         Z = Matrix{T}(I(nM))
         Z[j22,j22] = LinearAlgebra.LAPACK.ormrz!('R',tran,R[i1,:],tau,Z[j22,j22])[invperm(QR.p),:]
         Z = Z[:,jp] 
      else
         Z = nothing
      end
      N[i22,j22] = [ zeros(n,m) triu(R[i1,i1]); zeros(p,npm) ]
   else
      # compute the complete orthogonal decomposition of N22 using the SVD-decomposition
      SVD = svd!(view(N,i22,j22), full = true)
      tol = max(atol, rtol*SVD.S[1])
      n = count(x -> x > tol, SVD.S) 
      m = npm-n
      p = npp-n
      jp = [coff+n+1:coff+npm; coff+1:coff+n] 
      # [M12; M22] <- [M12; M22]*V*Pc
      M[i12,j22] = M[i12,j22]*SVD.V
      M[i12,j22] = M[i12,jp]
      # [M22 M23] <- U'*[M22 M23]
      M[i22,j23] = SVD.U'*M[i22,j23]
      # N12 <- N12*V*Pc
      N[i11,j22] = N[i11,j22]*SVD.V
      N[i11,j22] = N[i11,jp]
      # N23 <- U'*N23
      N[i22,jN23] = SVD.U'*N[i22,jN23]
      #Q = SVD.U
      if withQ
         Q = Matrix{T}(I(mM))
         Q[i22,i22] = SVD.U
      else
         Q = nothing
      end
      if withZ
         Z = Matrix{T}(I(nM))
         Z[j22,j22] = SVD.V
         Z[j22,j22] = Z[j22,jp] 
      else
         Z = nothing
      end
      N[i22,j22] = [ zeros(n,m) diagm(SVD.S[1:n]) ; zeros(p,npm) ]
   end
   return Q, Z, n, m, p
end
"""
    _preduce1!(n::Int,m::Int,p::Int,M::AbstractMatrix,N::AbstractMatrix,Q::Union{AbstractMatrix,Nothing},
               Z::Union{AbstractMatrix,Nothing}, tol; fast = true, roff = 0, coff = 0, rtrail = 0, ctrail = 0, 
               withQ = true, withZ = true)

Reduce the structured pencil  

     M = [ *  * *  *  ] roff          N = [ *  * *  *  ] roff
         [ 0  B A  *  ] n                 [ 0  0 E  *  ] n
         [ 0  D C  *  ] p                 [ 0  0 0  *  ] p
         [ 0  * *  *  ] rtrail            [ 0  * *  *  ] rtrail
         coff m n ctrail                  coff m n ctrail

with E upper triangular and nonsingular to the following form

     M = [ *   *    *    *    *  ] roff        N = [ *   *  *   *    *   ] roff
         [ 0   B1   A11 A12   *  ] τ+ρ             [ 0   0  E11 E12  *   ] τ+ρ
         [ 0   0    B2  A22   *  ] n-ρ             [ 0   0  0   E22  *   ] n-ρ
         [ 0   0    D2  C2    *  ] p-τ             [ 0   0  0   0    *   ] p-τ
         [ 0   *    *   *     *  ] rtrail          [ 0   *  *   *    *   ] rtrail
          coff m    ρ  n-ρ ctrail                   coff m  ρ  n-ρ ctrail   

where τ = rank D, B1 is full row rank, and E11 and E22 are upper triangular and nonsingular.
The performed orthogonal or unitary transformations are accumulated in `Q`, if `withQ = true`, and 
`Z`, if `withZ = true`. The rank decisions use the absolute tolerance `tol` for the nonzero elements of `M`.
"""
function _preduce1!(n::Int,m::Int,p::Int,M::AbstractMatrix,N::AbstractMatrix,Q::Union{AbstractMatrix,Nothing},
         Z::Union{AbstractMatrix,Nothing}, tol; fast = true, roff = 0, coff = 0, rtrail = 0, ctrail = 0, 
         withQ = true, withZ = true)
   #  Steps 1 and 2: QR- or SVD-decomposition based full column compression of [B; D] with the UT-form preservation of E
   #  Reduce the structured pencil  
   #
   #  M = [ *  * *  *  ] roff          N = [ *  * *  *  ] roff
   #      [ 0  B A  *  ] n                 [ 0  0 E  *  ] n
   #      [ 0  D C  *  ] p                 [ 0  0 0  *  ] p
   #      [ 0  * *  *  ] rtrail            [ 0  * *  *  ] rtrail
   #      coff m n ctrail                  coff m n ctrail
   #
   #  with E upper triangular and nonsingular to the following form
   #
   #  M = [ *   *    *    *    *  ] roff        N = [ *   *  *   *    *   ] roff
   #      [ 0   B1   A11 A12   *  ] τ+ρ             [ 0   0  E11 E12  *   ] τ+ρ
   #      [ 0   0    B2  A22   *  ] n-ρ             [ 0   0  0   E22  *   ] n-ρ
   #      [ 0   0    D2  C2    *  ] p-τ             [ 0   0  0   0    *   ] p-τ
   #      [ 0   *    *   *     *  ] rtrail          [ 0   *  *   *    *   ] rtrail
   #       coff m    ρ  n-ρ ctrail                   coff m  ρ  n-ρ ctrail   
   #
   #  where τ = rank D, B1 is full row rank and E11 and E22 are upper triangular and nonsingular.

   npm = n+m
   npp = n+p
   T = eltype(M)
   ZERO = zero(T)
   mM = roff + npp + rtrail
   nM = coff + npm + ctrail
   """
   Step 1:
   """
   ia = roff+1:roff+n
   ja = coff+m+1:coff+npm
   jb = coff+1:coff+m
   B = view(M,ia,jb) 
   BD = view(M,roff+1:roff+npp,jb)
   if p > 0
      # compress D to [D1 D2;0 0] with D1 invertible 
      ic = roff+n+1:roff+npp
      D = view(M,ic,jb)
      C = view(M,ic,ja)
      if fast
         QR = qr!(D, Val(true))
         τ = count(x -> x > tol, abs.(diag(QR.R))) 
      else
         #τ = rank(D; atol = tol)
         τ = count(x -> x > tol, svdvals(D))
         QR = qr!(D, Val(true))
      end
      B[:,:] = B[:,QR.p]
      lmul!(QR.Q',C)
      withQ && rmul!(view(Q,:,ic),QR.Q) 
      D[:,:] = [ QR.R[1:τ,:]; zeros(p-τ,m) ]
      """
      Step 2:
      """
      k = 1
      for j = coff+1:coff+τ
         for ii = roff+n+k:-1:roff+1+k
            iim1 = ii-1
            G, M[iim1,j] = givens(M[iim1,j],M[ii,j],iim1,ii)
            M[ii,j] = ZERO
            lmul!(G,view(M,:,j+1:coff+m+n))
            lmul!(G,view(N,:,ja))  # more efficient computation possible by selecting only the relevant columns
            withQ && rmul!(Q,G')
         end
         k += 1
      end
   else
      τ = 0
   end
   ρ = _preduce3!(n, m-τ, M, N, Q, Z, tol, fast = fast, coff = coff+τ, roff = roff+τ, rtrail = rtrail, ctrail = ctrail, withQ = withQ, withZ = withZ)
   if p > 0
      irt = 1:(τ+ρ)
      BD[irt,:] = BD[irt,invperm(QR.p)]
   end
   return τ, ρ 
end
"""
    _preduce2!(n::Int,m::Int,p::Int,M::AbstractMatrix,N::AbstractMatrix,Q::Union{AbstractMatrix,Nothing},
               Z::Union{AbstractMatrix,Nothing}, tol; fast = true, roff = 0, coff = 0, rtrail = 0, ctrail = 0, 
               withQ = true, withZ = true)

Reduce the structured pencil  

     M = [ *  * *   *  ] roff          N = [ *  * *  *   ] roff
         [ *  B A   *  ] n                 [ 0  0 E  *   ] n
         [ *  D C   *  ] p                 [ 0  0 0  *   ] p
         [ 0  0 0   *  ] rtrail            [ 0  0 0  *   ] rtrail
         coff m n ctrail                   coff m n ctrail

with `E` upper triangular and nonsingular to the following form

     M = [ *    *   *    *    *  ] roff        N = [ *   *  *   *    *   ] roff
         [ *    B1  A11 A12   *  ] n-ρ             [ 0   0  E11 E12  *   ] τ+ρ
         [ *    D1  C1  A22   *  ] ρ               [ 0   0  0   E22  *   ] n-ρ
         [ 0    0   0   C2    *  ] p               [ 0   0  0   0    *   ] p-τ
         [ 0    0   0   0     *  ] rtrail          [ 0   0  0   0    *   ] rtrail
          coff m-τ n-ρ τ+ρ ctrail                   coff m  ρ  n-ρ ctrail   

where `τ = rank D`, `C2` is full column rank and `E11` and `E22` are upper triangular and nonsingular. 
The performed orthogonal or unitary transformations are accumulated in `Q`, if `withQ = true`, and 
`Z`, if `withZ = true`. The rank decisions use the absolute tolerance `tol` for the nonzero elements of `M`.
"""
function _preduce2!(n::Int,m::Int,p::Int,M::AbstractMatrix,N::AbstractMatrix,Q::Union{AbstractMatrix,Nothing},
          Z::Union{AbstractMatrix,Nothing}, tol; fast = true, roff = 0, coff = 0, rtrail = 0, ctrail = 0, 
          withQ = true, withZ = true)
   #  Steps 1 and 2: QR- or SVD-decomposition based full column compression of [D C] with the UT-form preservation of E
   #  Reduce the structured pencil  
   #
   #  M = [ *  * *  *  ] roff          N = [ *  * *  *  ] roff
   #      [ *  B A  *  ] n                 [ 0  0 E  *  ] n
   #      [ *  D C  *  ] p                 [ 0  0 0  *  ] p
   #      [ 0  0 0  *  ] rtrail            [ 0  0 0  *  ] rtrail
   #      coff m n ctrail                  coff m n ctrail
   #
   #  with E upper triangular and nonsingular to the following form
   #
   #  M = [ *    *   *    *    *  ] roff        N = [ *   *  *   *    *   ] roff
   #      [ *    B1  A11 A12   *  ] n-ρ             [ 0   0  E11 E12  *   ] τ+ρ
   #      [ *    D1  C1  A22   *  ] ρ               [ 0   0  0   E22  *   ] n-ρ
   #      [ 0    0   0   C2    *  ] p               [ 0   0  0   0    *   ] p-τ
   #      [ 0    0   0   0     *  ] rtrail          [ 0   0  0   0    *   ] rtrail
   #       coff m-τ n-ρ τ+ρ ctrail                   coff m  ρ  n-ρ ctrail   
   #
   #  where τ = rank D, C2 is full column rank and E11 and E22 are upper triangular and nonsingular.
       
   npp = n+p 
   npm = n+m
   T = eltype(M)
   ZERO = zero(T)
   """
   Step 1:
   """
   mM = roff + npp + rtrail 
   nM = coff + npm + ctrail 
   T = eltype(M)
   ZERO = zero(T)
   ia = 1:roff+n
   ic = roff+n+1:roff+npp
   jc = coff+m+1:coff+npm
   jb = coff+1:coff+m
   jdc = coff+1:nM
   BE = view(M,ia,jb) 
   DC = view(M,ic,coff+1:coff+npm)
   C = view(M,ic,jc)
   
   if m > 0
      # compress D' to [D1' D2'; 0 0] = Q'*D'*P with D1 invertible
      ic = roff+n+1:roff+npp
      D = view(M,ic,jb)
      if fast
         QR = qr!(copy(D'), Val(true))
         τ = count(x -> x > tol, abs.(diag(QR.R))) 
      else
         # τ = rank(D; atol = tol)
         τ = count(x -> x > tol, svdvals(D))
         QR = qr!(copy(D'), Val(true))
      end
      rmul!(BE,QR.Q)  # BE*Q
      jt = m:-1:1
      BE[:,:] = BE[:,jt]      # BE*Q*P2
      if withZ 
         Z1 = view(Z,:,jb)
         rmul!(Z1,QR.Q) 
         Z1[:,:] = Z1[:,jt]
      end
      D[:,:] = [ zeros(p,m-τ)  QR.R[τ:-1:1,p:-1:1]' ]
      C[:,:] = C[QR.p,:]
      C[:,:] = reverse(C,dims=1)
      """
      Step 2:
      """
      k = 1
      for i = roff+npp:-1:roff+npp+1-τ     
         for jj = coff+m+1-k:coff+m+n-k
             jjp1 = jj+1
             G, r = givens(conj(M[i,jjp1]),conj(M[i,jj]),jjp1,jj)
             M[i,jjp1] = conj(r)
             M[i,jj] = ZERO
             rmul!(view(M,1:i-1,jdc),G')  
             rmul!(view(N,1:i,jdc),G')  
             withZ && rmul!(Z,G')
          end
         k += 1
      end
   else
      τ = 0
   end
   ρ = _preduce4!(n, m-τ, p-τ, M, N, Q, Z, tol, fast = fast, coff = coff, roff = roff, rtrail = rtrail+τ, ctrail = ctrail+τ, withQ = withQ, withZ = withZ)
   if m > 0
      jrt = npm-(τ+ρ)+1:npm
      DC[:,jrt] = reverse(DC[:,jrt],dims=1)
      DC[:,jrt] = DC[invperm(QR.p),jrt]
   end
   return τ, ρ 
end
"""
    _preduce3!(n::Int,m::Int,M::AbstractMatrix,N::AbstractMatrix,Q::Union{AbstractMatrix,Nothing}, Z::Union{AbstractMatrix,Nothing}, tol; 
               fast = true, roff = 0, coff = 0, rtrail = 0, ctrail = 0,  withQ = true, withZ = true)

Reduce the structured pencil  

        [ *  *  *   *  ] roff        [ *  *  *   *  ] roff
    M = [ 0  B  A   *  ] n       N = [ 0  0  E   *  ] n
        [ 0  *  *   *  ] rtrail      [ 0  *  *   *  ] rtrail
        coff m  n ctrail             coff m  n ctrail

with `E` upper triangular and nonsingular to the following form

        [ *  *   *   *     *  ] roff        [ *  *  *   *     *  ] roff
    M = [ 0  B1  A11 A12   *  ] n-ρ     N = [ 0  0  E11 E12   *  ] n-ρ
        [ 0  0   A21 A22   *  ] ρ           [ 0  0  0   E22   *  ] ρ
        [ 0  *   *   *     *  ] rtrail      [ 0  *  *   *     *  ] rtrail
        coff m  n-ρ  ρ   ctrail             coff m  n-ρ ρ   ctrail

where `B1` has full row rank and `E11` and `E22` are upper triangular and nonsingular. 
The performed orthogonal or unitary transformations are accumulated in `Q`, if `withQ = true`, and 
`Z`, if `withZ = true`. The rank decisions use the absolute tolerance `tol` for the nonzero elements of `M`.
"""
function _preduce3!(n::Int,m::Int,M::AbstractMatrix,N::AbstractMatrix,Q::Union{AbstractMatrix,Nothing}, Z::Union{AbstractMatrix,Nothing}, tol; 
                    fast = true, roff = 0, coff = 0, rtrail = 0, ctrail = 0,  withQ = true, withZ = true)
   #  Step 3: QR- or SVD-decomposition based full row compression of B and UT-form preservation of E
   #  Reduce the structured pencil  
   #
   #       [ *  *  *   *  ] roff        [ *  *  *   *  ] roff
   #   M = [ 0  B  A   *  ] n       N = [ 0  0  E   *  ] n
   #       [ 0  *  *   *  ] rtrail      [ 0  *  *   *  ] rtrail
   #       coff m  n ctrail             coff m  n ctrail
   #
   #  with E upper triangular and nonsingular to the following form
   #
   #       [ *  *   *   *     *  ] roff        [ *  *  *   *     *  ] roff
   #   M = [ 0  B1  A11 A12   *  ] n-ρ     N = [ 0  0  E11 E12   *  ] n-ρ
   #       [ 0  0   A21 A22   *  ] ρ           [ 0  0  0   E22   *  ] ρ
   #       [ 0  *   *   *     *  ] rtrail      [ 0  *  *   *     *  ] rtrail
   #       coff m  n-ρ  ρ   ctrail             coff m  n-ρ ρ   ctrail
   #
   #  where B1 has full row rank and E11 and E22 are upper triangular and nonsingular. 
   #
   npm = n+m
   mM = roff + n + rtrail
   nM = coff + npm + ctrail
   T = eltype(M)
   ZERO = zero(T)
   ib = roff+1:roff+n
   je = coff+m+1:coff+npm
   B = view(M,ib,coff+1:coff+m)
   E = view(N,ib,je)
   if fast
      ρ = 0
      nrm = similar(real(M),m)
      jp = Vector(1:m)
      nm = min(n,m)
      for j = 1:nm
         for l = j:m
            nrm[l] = norm(B[j:n,l])
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
            (B[:,j],B[:,ind]) = (B[:,ind],B[:,j])
         end
         for ii = n:-1:j+1
             iim1 = ii-1
             G, B[iim1,j] = givens(B[iim1,j],B[ii,j],iim1,ii)
             B[ii,j] = ZERO
             lmul!(G,view(M,ib,coff+j+1:nM))
             lmul!(G,view(N,ib,coff+m+iim1:nM))
             withQ && rmul!(view(Q,:,ib),G') 
             G, r = givens(conj(E[ii,ii]),conj(E[ii,iim1]),ii,iim1)
             E[ii,ii] = conj(r)
             E[ii,iim1] = ZERO 
             rmul!(view(N,1:roff+iim1,je),G')
             withZ && rmul!(view(Z,:,je),G') 
             rmul!(view(M,:,je),G')
         end
      end
      B[:,:] = [ B[1:ρ,invperm(jp)]; zeros(T,n-ρ,m) ]
   else
      if n > m
        for j = 1:m
          for ii = n:-1:j+1
            iim1 = ii-1
             G, B[iim1,j] = givens(B[iim1,j],B[ii,j],iim1,ii)
            B[ii,j] = ZERO
            lmul!(G,view(M,ib,coff+j+1:nM))
            lmul!(G,view(N,ib,coff+m+iim1:nM))
            withQ && rmul!(view(Q,:,ib),G') 
            G, r = givens(conj(E[ii,ii]),conj(E[ii,iim1]),ii,iim1)
            E[ii,ii] = conj(r)
            E[ii,iim1] = ZERO 
            rmul!(view(N,1:roff+iim1,je),G')
            withZ && rmul!(view(Z,:,je),G') 
            rmul!(view(M,:,je),G')
          end
        end
      end
      mn = min(n,m)
      if mn > 0
         ics = 1:mn
         jcs = 1:m
         SVD = svd(B[ics,jcs], full = true)
         ρ = count(x -> x > tol, SVD.S) 
         if ρ == mn
            return ρ
         else
            B[ics,jcs] = [ diagm(SVD.S[1:ρ])*SVD.Vt[1:ρ,:]; zeros(T,mn-ρ,m) ]
            if ρ == 0
               return ρ
            end
         end
         ibt = roff+1:roff+mn
         jt = coff+m+1:nM
         withQ && (Q[:,ibt] = Q[:,ibt]*SVD.U)
         N[ibt,jt] = SVD.U'*N[ibt,jt]
         M[ibt,jt] = SVD.U'*M[ibt,jt]
         tau = similar(N,mn)
         jt1 = coff+m+1:coff+m+mn
         E11 = view(N,ibt,jt1)
         LinearAlgebra.LAPACK.gerqf!(E11,tau)
         eltype(M) <: Complex ? tran = 'C' : tran = 'T'
         LinearAlgebra.LAPACK.ormrq!('R',tran,E11,tau,view(M,:,jt1))
         withZ && LinearAlgebra.LAPACK.ormrq!('R',tran,E11,tau,view(Z,:,jt1)) 
         LinearAlgebra.LAPACK.ormrq!('R',tran,E11,tau,view(N,1:roff,jt1))
         triu!(E11)
      else
         ρ = 0
      end
   end
   return ρ 
end
"""
    _preduce4!(n::Int,m::Int,p::Int,M::AbstractMatrix,N::AbstractMatrix,Q::Union{AbstractMatrix,Nothing}, Z::Union{AbstractMatrix,Nothing}, tol; 
               fast = true, roff = 0, coff = 0, rtrail = 0, ctrail = 0, withQ = true, withZ = true)

Reduce the structured pencil  

    M = [ *  *  *   *  ] roff          N = [ *  *  *   *  ] roff
        [ 0  B  A   *  ]  n                [ 0  0  E   *  ]  n
        [ 0  0  C   *  ]  p                [ 0  0  0   *  ]  p
        [ 0  *  *   *  ] rtrail            [ 0  *  *   *  ] rtrail
        coff m  n ctrail                   coff m  n ctrail

with `E` upper triangular and nonsingular to the following form

    M = [ *  *   *   *    * ] roff          N = [ *  *  *   *    * ] roff
        [ 0  B1  A11 A12  * ]  n-ρ              [ 0  0  E11 E12  * ]  n-ρ
        [ 0  B2  A21 A22  * ]  ρ                [ 0  0  0   E22  * ]  ρ
        [ 0  0   0   C1   * ]  p                [ 0  0  0   0    * ]  p
        [ 0  *   *   *    * ] rtrail            [ 0  *  *   *    * ] rtrail
        coff m  n-ρ  ρ ctrail                   coff m n-ρ  ρ ctrail   

where `C1` has full column rank and `E11` and `E22` are upper triangular and nonsingular. 
The performed orthogonal or unitary transformations are accumulated in `Q`, if `withQ = true`, and 
`Z`, if `withZ = true`. The rank decisions use the absolute tolerance `tol` for the nonzero elements of `M`.
"""
function _preduce4!(n::Int,m::Int,p::Int,M::AbstractMatrix,N::AbstractMatrix,Q::Union{AbstractMatrix,Nothing}, Z::Union{AbstractMatrix,Nothing}, tol; 
                    fast = true, roff = 0, coff = 0, rtrail = 0, ctrail = 0, withQ = true, withZ = true)
   #  Step 4: QR- or SVD-decomposition based full column compression of C and UT-form preservation of E
   #  Reduce the structured pencil  
   #
   #  M = [ *  *  *   *  ] roff          N = [ *  *  *   *  ] roff
   #      [ 0  B  A   *  ]  n                [ 0  0  E   *  ]  n
   #      [ 0  0  C   *  ]  p                [ 0  0  0   *  ]  p
   #      [ 0  *  *   *  ] rtrail            [ 0  *  *   *  ] rtrail
   #      coff m  n ctrail                   coff m  n ctrail
   #
   #  with E upper triangular and nonsingular to the following form
   #
   #  M = [ *  *   *   *    * ] roff          N = [ *  *  *   *    * ] roff
   #      [ 0  B1  A11 A12  * ]  n-ρ              [ 0  0  E11 E12  * ]  n-ρ
   #      [ 0  B2  A21 A22  * ]  ρ                [ 0  0  0   E22  * ]  ρ
   #      [ 0  0   0   C1   * ]  p                [ 0  0  0   0    * ]  p
   #      [ 0  *   *   *    * ] rtrail            [ 0  *  *   *    * ] rtrail
   #      coff m  n-ρ  ρ ctrail                   coff m n-ρ  ρ ctrail   
   #
   #  where C1 has full column rank and E11 and E22 are upper triangular and nonsingular. 
   npp = n+p
   npm = n+m
   mM = roff + npp + rtrail
   nM = coff + npm + ctrail
   T = eltype(M)
   ZERO = zero(T)
   ie = roff+1:roff+n
   ic = roff+n+1:roff+npp
   jc = coff+m+1:coff+npm
   C = view(M,ic,jc)
   E = view(N,ie,jc)
   AE = view(M,1:roff+n,jc)
   if fast
      ρ = 0
      nrm = similar(real(M),p)
      jp = Vector(1:p)
      np = min(n,p)
      for i = 1:np
         ii = p-i+1
         for l = 1:ii
            nrm[l] = norm(C[l,1:n-i+1])
         end
         nrmax, ind = findmax(nrm[1:ii]) 
         if nrmax < tol
            break
         else
            ρ += 1
         end
         if ind !== ii
            (jp[ii], jp[ind]) = (jp[ind], jp[ii])
            (C[ii,:],C[ind,:]) = (C[ind,:],C[ii,:])
         end
         for jj = 1:n-i
             jjp1 = jj+1
             G, r = givens(conj(C[ii,jjp1]),conj(C[ii,jj]),jjp1,jj)
             C[ii,jjp1] = conj(r)
             C[ii,jj] = ZERO
             rmul!(view(C,1:ii-1,:),G')
             rmul!(AE,G')
             rmul!(view(N,1:roff+jjp1,jc),G')
             withZ && rmul!(view(Z,:,jc),G') 
             G, E[jj,jj] = givens(E[jj,jj],E[jjp1,jj],jj,jjp1)
             E[jjp1,jj] = ZERO
             lmul!(G,view(N,ie,coff+m+jjp1:nM))
             withQ && rmul!(view(Q,:,ie),G') 
             lmul!(G,view(M,ie,coff+1:nM))
         end
      end
      C[:,1:n] = [ zeros(T,p,n-ρ)  C[invperm(jp),n-ρ+1:n]];
   else
      if n > p
         for i = 1:p
             ii = p-i+1
             for jj = 1:n-i
                jjp1 = jj+1
                G, r = givens(conj(C[ii,jjp1]),conj(C[ii,jj]),jjp1,jj)
                C[ii,jjp1] = conj(r)
                C[ii,jj] = ZERO
                rmul!(view(C,1:ii-1,:),G')
                rmul!(AE,G')
                rmul!(view(N,1:roff+jjp1,jc),G')
                withZ && rmul!(view(Z,:,jc),G') 
                G, E[jj,jj] = givens(E[jj,jj],E[jjp1,jj],jj,jjp1)
                E[jjp1,jj] = ZERO
                lmul!(G,view(N,ie,coff+m+jjp1:nM))
                withQ && rmul!(view(Q,:,ie),G') 
                lmul!(G,view(M,ie,coff+1:nM))
            end
         end
      end
      pn = min(n,p)
      if pn > 0
         ics = 1:p
         jcs = n-pn+1:n
         SVD = svd(C[ics,jcs], full = true)
         ρ = count(x -> x > tol, SVD.S) 
         Z1 = reverse(SVD.V,dims=2)
         Q1 = reverse(SVD.U,dims=2)
         if ρ == pn
            return ρ
         else
            C[ics,jcs] = [ zeros(p,pn-ρ) Q1*[zeros(p-ρ,ρ); diagm(reverse(SVD.S[1:ρ])) ] ] 
            if ρ == 0
               return ρ
            end
         end
         jt = coff+npm-pn+1:coff+npm
         withZ && (Z[:,jt] = Z[:,jt]*Z1) 
         M[1:roff+n,jt] = M[1:roff+n,jt]*Z1
         N[1:roff+n,jt] = N[1:roff+n,jt]*Z1    # more efficient computation possible
         it = roff+n-pn+1:roff+n
         jt1 = coff+n+m+1:nM
         tau = similar(N,pn)
         E22 = view(N,it,jt)
         LinearAlgebra.LAPACK.geqrf!(E22,tau)
         eltype(M) <: Complex ? tran = 'C' : tran = 'T'
         LinearAlgebra.LAPACK.ormqr!('L',tran,E22,tau,view(M,it,coff+1:nM))
         withQ && LinearAlgebra.LAPACK.ormqr!('R','N',E22,tau,view(Q,:,it)) 
         LinearAlgebra.LAPACK.ormqr!('L',tran,E22,tau,view(N,it,jt1))
         triu!(E22)
      else
         ρ = 0
      end
   end
   return ρ 
end
# this function ensures compatibility with Julia 1.0
function rank(A::AbstractMatrix; atol::Real = 0.0, rtol::Real = (min(size(A)...)*eps(real(float(one(eltype(A))))))*iszero(atol))
   isempty(A) && return 0 # 0-dimensional case
   s = svdvals(A)
   tol = max(atol, rtol*s[1])
   count(x -> x > tol, s)
end

