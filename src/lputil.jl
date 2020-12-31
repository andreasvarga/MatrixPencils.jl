"""
    _preduceBF!(M, N, Q, Z, L::Union{AbstractMatrix,Missing}, R::Union{AbstractMatrix,Missing}; 
                fast = true, atol = 0, rtol, roff = 0, coff = 0, rtrail = 0, ctrail = 0, 
                withQ = true, withZ = true) -> n, m, p

Reduce the partitioned matrix pencil `M - λN` 

              [   *         *         *  ] roff
     M - λN = [   0      M22-λN22     *  ] npp
              [   0         0         *  ] rtrail
                coff       npm     ctrail
  
to an equivalent basic form `F - λG = Q1'*(M - λN)*Z1` using orthogonal transformation matrices `Q1` and `Z1` 
such that the subpencil `M22 - λN22` is transformed into the following standard form
 
                       | B  | A-λE | 
          F22 - λG22 = |----|------| ,        
                       | D  |  C   |

where `E` is an `nxn` non-singular matrix, and `A`, `B`, `C`, `D` are `nxn`-, `nxm`-, `pxn`- and `pxm`-dimensional matrices,
respectively. The order `n` of `E` is equal to the numerical rank of `N` determined using the absolute tolerance `atol` and 
relative tolerance `rtol`. `M` and `N` are overwritten by `F` and `G`, respectively. 

The performed orthogonal or unitary transformations are accumulated in `Q` (i.e., `Q <- Q*Q1`), if `withQ = true`, and 
`Z` (i.e., `Z <- Z*Z1`), if `withZ = true`. 

The  matrix `L` is overwritten by `Q1'*L` unless `L = missing` and the  matrix `R` is overwritten by `R*Z1` unless `R = missing`. 

If `fast = true`, `E` is determined upper triangular using a rank revealing QR-decomposition with column pivoting of `N22` 
and `n` is evaluated as the number of nonzero diagonal elements of the R factor, whose magnitudes are greater than 
`tol = max(atol,abs(R[1,1])*rtol)`. 
If `fast = false`,  `E` is determined diagonal using a rank revealing SVD-decomposition of `N22` and 
`n` is evaluated as the number of singular values greater than `tol = max(atol,smax*rtol)`, where `smax` 
is the largest singular value. 
The rank decision based on the SVD-decomposition is generally more reliable, but the involved computational effort is higher.
"""
function _preduceBF!(M::AbstractMatrix{T}, N::AbstractMatrix{T}, 
                     Q::Union{AbstractMatrix{T},Nothing}, Z::Union{AbstractMatrix{T},Nothing},
                     L::Union{AbstractVecOrMat{T},Missing} = missing, R::Union{AbstractMatrix{T},Missing} = missing; 
                     atol::Real = zero(real(T)), rtol::Real = (min(size(M)...)*eps(real(float(one(T)))))*iszero(atol), 
                     fast::Bool = true, roff::Int = 0, coff::Int = 0, rtrail::Int = 0, ctrail::Int = 0, 
                     withQ::Bool = true, withZ::Bool = true) where T <: BlasFloat
   # In interest of performance, no dimensional checks are performed                  
   mM, nM = size(M)
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

   # fast return for null dimensions
   (npp == 0 || npm == 0) && (return 0, npm, npp)  

   # Step 0: Reduce M22 -λ N22 to the standard form
   i11 = 1:roff
   i22 = roff+1:roff+npp
   j22 = coff+1:coff+npm
   i12 = 1:roff+npp
   j23 = coff+1:nM
   jN23 = coff+npm+1:nM
   T <: Complex ? tran = 'C' : tran = 'T'
   Q === nothing && (withQ = false)
   Z === nothing && (withZ = false)
   if fast
      # compute in-place the QR-decomposition N22*P2 = Q2*[R2;0] with column pivoting 
      N22 = view(N,i22,j22)
      _, τ, jpvt = LinearAlgebra.LAPACK.geqp3!(N22)
      tol = max(atol, rtol*abs(N22[1,1]))
      n = count(x -> x > tol, abs.(diag(N22))) 
      m = npm-n
      p = npp-n
      i1 = 1:n
      # [M22 M23] <- Q2'*[M22 M23]
      LinearAlgebra.LAPACK.ormqr!('L',tran,N22,τ,view(M,i22,j23))
      # [M12; M22] <- [M12; M22]*P2
      M[i12,j22] = M[i12,j22[jpvt]]      
      # N23 <- Q2'*N23
      LinearAlgebra.LAPACK.ormqr!('L',tran,N22,τ,view(N,i22,jN23))
      # N12 <- N12*P2
      N[i11,j22] = N[i11,j22[jpvt]]
      # Q <- Q*Q2
      withQ && LinearAlgebra.LAPACK.ormqr!('R','N',N22,τ,view(Q,:,i22))
      # Z <- Z*P2
      withZ && (Z[:,j22] = Z[:,j22[jpvt]])
      # L <- Q2'*L
      ismissing(L) || LinearAlgebra.LAPACK.ormqr!('L',tran,N22,τ,view(L,i22,:))
      # N22 = [R2;0] 
      N22[:,:] = [ triu(N22[i1,:]); zeros(T,p,npm) ]
       # R <- R*P2
      ismissing(R) || (R[:,j22] = R[:,j22[jpvt]])
      
      # compute in-place the complete orthogonal decomposition N22*Z2*Pc2 = [0 E; 0 0] with E nonsingular and UT
      _, tau = LinearAlgebra.LAPACK.tzrzf!(view(N22,i1,:))
      jp = [coff+n+1:coff+npm; coff+1:coff+n] 
      N22r = view(N22,i1,:)
      # [M12; M22] <- [M12; M22]*Z2*Pc2
      LinearAlgebra.LAPACK.ormrz!('R',tran,N22r,tau,view(M,i12,j22))
      M[i12,j22] = M[i12,jp]
      # N12 <- N12*Z2*Pc2
      LinearAlgebra.LAPACK.ormrz!('R',tran,N22r,tau,view(N,i11,j22))
      N[i11,j22] = N[i11,jp]
      if withZ
         LinearAlgebra.LAPACK.ormrz!('R',tran,N22r,tau,view(Z,:,j22))
         Z[:,j22] = Z[:,jp] 
      end
      # R <- R*Z2*Pc2
      ismissing(R) || (LinearAlgebra.LAPACK.ormrz!('R',tran,N22r,tau,view(R,:,j22)); R[:,j22] = R[:,jp]) 
      N22[:,:] = [ zeros(T,n,m) triu(N22[i1,i1]); zeros(T,p,npm) ]
   else
      # compute the complete orthogonal decomposition of N22 using the SVD-decomposition
      U, S, Vt = LinearAlgebra.LAPACK.gesdd!('A',view(N,i22,j22))
      tol = max(atol, rtol*S[1])
      n = count(x -> x > tol, S) 
      m = npm-n
      p = npp-n
      jp = [coff+n+1:coff+npm; coff+1:coff+n] 
      # [M12; M22] <- [M12; M22]*V*Pc
      M[i12,j22] = M[i12,j22]*Vt'
      M[i12,j22] = M[i12,jp]
      # [M22 M23] <- U'*[M22 M23]
      M[i22,j23] = U'*M[i22,j23]
      # N12 <- N12*V*Pc
      N[i11,j22] = N[i11,j22]*Vt'
      N[i11,j22] = N[i11,jp]
      # N23 <- U'*N23
      N[i22,jN23] = U'*N[i22,jN23]
      # Q <- Q*U
      withQ && (Q[:,i22] = Q[:,i22]*U)
      # Z <- Q*V
      if withZ
         Z[:,j22] = Z[:,j22]*Vt'
         Z[:,j22] = Z[:,jp] 
      end
      # L <- U'*L
      ismissing(L) || (L[i22,:] = U'*L[i22,:])
      # R <- R*V
      ismissing(R) || (R[:,j22] = R[:,j22]*Vt'; R[:,j22] = R[:,jp] )
      N[i22,j22] = [ zeros(T,n,m) Diagonal(S[1:n]) ; zeros(T,p,npm) ]
   end
   return n, m, p
end
"""
    _preduce1!(n::Int, m::Int, p::Int, M::AbstractMatrix, N::AbstractMatrix,
               Q::Union{AbstractMatrix,Nothing}, Z::Union{AbstractMatrix,Nothing}, tol,
               L::Union{AbstractMatrix{T},Missing} = missing, R::Union{AbstractMatrix{T},Missing} = missing; 
               fast = true, roff = 0, coff = 0, rtrail = 0, ctrail = 0, withQ = true, withZ = true)

Reduce the structured pencil  `M - λN`

     M = [ *  * *  *  ] roff          N = [ *  * *  *  ] roff
         [ 0  B A  *  ] n                 [ 0  0 E  *  ] n
         [ 0  D C  *  ] p                 [ 0  0 0  *  ] p
         [ 0  * *  *  ] rtrail            [ 0  * *  *  ] rtrail
         coff m n ctrail                  coff m n ctrail

with E upper triangular and nonsingular to the following form `M1 - λN1 = Q1'*(M - λN)*Z1` with

    M1 = [ *   *    *    *    *  ] roff       N1 = [ *   *  *   *    *   ] roff
         [ 0   B1   A11 A12   *  ] τ+ρ             [ 0   0  E11 E12  *   ] τ+ρ
         [ 0   0    B2  A22   *  ] n-ρ             [ 0   0  0   E22  *   ] n-ρ
         [ 0   0    D2  C2    *  ] p-τ             [ 0   0  0   0    *   ] p-τ
         [ 0   *    *   *     *  ] rtrail          [ 0   *  *   *    *   ] rtrail
          coff m    ρ  n-ρ ctrail                   coff m  ρ  n-ρ ctrail   

where τ = rank D, B1 is full row rank, and E11 and E22 are upper triangular and nonsingular.
The performed orthogonal or unitary transformations are accumulated in `Q` (i.e., `Q <- Q*Q1`), if `withQ = true`, and 
`Z` (i.e., `Z <- Z*Z1`), if `withZ = true`. The rank decisions use the absolute tolerance `tol` for the nonzero elements of `M`.

The  matrix `L` is overwritten by `Q1'*L` unless `L = missing` and the  matrix `R` is overwritten by `R*Z1` unless `R = missing`. 
"""
function _preduce1!(n::Int, m::Int, p::Int, M::AbstractMatrix{T}, N::AbstractMatrix{T},
                    Q::Union{AbstractMatrix{T},Nothing}, Z::Union{AbstractMatrix{T},Nothing}, tol::Real, 
                    L::Union{AbstractVecOrMat{T},Missing} = missing, R::Union{AbstractMatrix{T},Missing} = missing; 
                    fast::Bool = true, roff::Int = 0, coff::Int = 0, rtrail::Int = 0, ctrail::Int = 0, 
                    withQ::Bool = true, withZ::Bool = true) where T <: BlasFloat
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
   #  M1 = [ *   *    *    *    *  ] roff        N1 = [ *   *  *   *    *   ] roff
   #       [ 0   B1   A11 A12   *  ] τ+ρ              [ 0   0  E11 E12  *   ] τ+ρ
   #       [ 0   0    B2  A22   *  ] n-ρ              [ 0   0  0   E22  *   ] n-ρ
   #       [ 0   0    D2  C2    *  ] p-τ              [ 0   0  0   0    *   ] p-τ
   #       [ 0   *    *   *     *  ] rtrail           [ 0   *  *   *    *   ] rtrail
   #        coff m    ρ  n-ρ ctrail                    coff m  ρ  n-ρ ctrail   
   #
   #  where τ = rank D, B1 is full row rank and E11 and E22 are upper triangular and nonsingular.

   npm = n+m
   npp = n+p
   ZERO = zero(T)
   mM = roff + npp + rtrail
   nM = coff + npm + ctrail
   # Step 1:
   ia = roff+1:roff+n
   ja = coff+m+1:nM
   jb = coff+1:coff+m
   B = view(M,ia,jb) 
   BD = view(M,roff+1:roff+npp,jb)
   if p > 0
      # compress D to [D1 D2;0 0] with D1 invertible 
      ic = roff+n+1:roff+npp
      D = view(M,ic,jb)
      CE = view(M,ic,ja)
      EE = view(N,ic,coff+npm+1:nM)
      if fast
         QR = qr!(D, Val(true))
         τ = count(x -> x > tol, abs.(diag(QR.R))) 
      else
         τ = count(x -> x > tol, svdvals(D))
         QR = qr!(D, Val(true))
      end
      B[:,:] = B[:,QR.p]
      lmul!(QR.Q',CE)
      lmul!(QR.Q',EE)
      withQ && rmul!(view(Q,:,ic),QR.Q) 
      ismissing(L) || lmul!(QR.Q',view(L,ic,:))
      D[:,:] = [ QR.R[1:τ,:]; zeros(p-τ,m) ]
      # Step 2:
      k = 1
      for j = coff+1:coff+τ
         for ii = roff+n+k:-1:roff+1+k
            iim1 = ii-1
            G, M[iim1,j] = givens(M[iim1,j],M[ii,j],iim1,ii)
            M[ii,j] = ZERO
            lmul!(G,view(M,:,j+1:nM))
            lmul!(G,view(N,:,ja))  # more efficient computation possible by selecting only the relevant columns
            withQ && rmul!(Q,G')
            ismissing(L) || lmul!(G,L)
         end
         k += 1
      end
   else
      τ = 0
   end
   ρ = _preduce3!(n, m-τ, M, N, Q, Z, tol, L, R, fast = fast, coff = coff+τ, roff = roff+τ, rtrail = rtrail, ctrail = ctrail, withQ = withQ, withZ = withZ)
   if p > 0
      irt = 1:(τ+ρ)
      BD[irt,:] = BD[irt,invperm(QR.p)]
   end
   return τ, ρ 
end
"""
    _preduce2!(n::Int, m::Int, p::Int, M::AbstractMatrix, N::AbstractMatrix, 
               Q::Union{AbstractMatrix,Nothing}, Z::Union{AbstractMatrix,Nothing}, tol,
               L::Union{AbstractMatrix{T},Missing} = missing, R::Union{AbstractMatrix{T},Missing} = missing; 
               fast = true, roff = 0, coff = 0, rtrail = 0, ctrail = 0, withQ = true, withZ = true)

Reduce the structured pencil  

     M = [ *  * *   *  ] roff          N = [ *  * *  *   ] roff
         [ *  B A   *  ] n                 [ 0  0 E  *   ] n
         [ *  D C   *  ] p                 [ 0  0 0  *   ] p
         [ 0  0 0   *  ] rtrail            [ 0  0 0  *   ] rtrail
         coff m n ctrail                   coff m n ctrail

with `E` upper triangular and nonsingular to the following form `M1 - λN1 = Q1'*(M - λN)*Z1` with

    M1 = [ *    *   *    *    *  ] roff       N1 = [ *    *   *   *    *   ] roff
         [ *    B1  A11 A12   *  ] n-ρ             [ 0    0  E11 E12   *   ] n-ρ
         [ *    D1  C1  A22   *  ] ρ               [ 0    0   0  E22   *   ] ρ
         [ 0    0   0   C2    *  ] p               [ 0    0   0   0    *   ] p
         [ 0    0   0   0     *  ] rtrail          [ 0    0   0   0    *   ] rtrail
          coff m-τ n-ρ τ+ρ ctrail                   coff m-τ n-ρ τ+ρ ctrail   

where `τ = rank D`, `C2` is full column rank and `E11` and `E22` are upper triangular and nonsingular. 
The performed orthogonal or unitary transformations are accumulated in `Q` (i.e., `Q <- Q*Q1`), if `withQ = true`, and 
`Z` (i.e., `Z <- Z*Z1`), if `withZ = true`. The rank decisions use the absolute tolerance `tol` for the nonzero elements of `M`.

The  matrix `L` is overwritten by `Q1'*L` unless `L = missing` and the  matrix `R` is overwritten by `R*Z1` unless `R = missing`. 
"""
function _preduce2!(n::Int, m::Int, p::Int, M::AbstractMatrix{T}, N::AbstractMatrix{T}, 
                    Q::Union{AbstractMatrix{T},Nothing}, Z::Union{AbstractMatrix{T},Nothing}, tol::Real,  
                    L::Union{AbstractVecOrMat{T},Missing} = missing, R::Union{AbstractMatrix{T},Missing} = missing; 
                    fast::Bool = true, roff::Int = 0, coff::Int = 0, rtrail::Int = 0, ctrail::Int = 0, 
                    withQ::Bool = true, withZ::Bool = true) where T <: BlasFloat
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
   #  M1 = [ *   *   *    *    *  ] roff       N1 = [ *    *   *   *    *   ] roff
   #       [ *   B1  A11 A12   *  ] n-ρ             [ 0    0  E11 E12   *   ] n-ρ
   #       [ *   D1  C1  A22   *  ] ρ               [ 0    0   0  E22   *   ] ρ
   #       [ 0   0   0   C2    *  ] p               [ 0    0   0   0    *   ] p
   #       [ 0   0   0   0     *  ] rtrail          [ 0    0   0   0    *   ] rtrail
   #       coff m-τ n-ρ τ+ρ ctrail                   coff m-τ n-ρ τ+ρ ctrail   
   #
   #  where τ = rank D, C2 is full column rank and E11 and E22 are upper triangular and nonsingular.
       
   npp = n+p 
   npm = n+m
   ZERO = zero(T)
   # Step 1:
   mM = roff + npp + rtrail 
   nM = coff + npm + ctrail 
   ia = 1:roff+n
   ic = roff+n+1:roff+npp
   jc = coff+m+1:coff+npm
   jb = coff+1:coff+m
   jdc = 1:nM
   BE = view(M,ia,jb) 
   EE = view(N,1:roff,jb) 
   DC = view(M,ic,coff+1:coff+npm)
   C = view(M,ic,jc)
   D = view(M,ic,jb)
   
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
      ismissing(R) || ( rmul!(view(R,:,jb),QR.Q); R[:,jb] = R[:,reverse(jb)] )
      rmul!(EE,QR.Q)  
      EE[:,:] = EE[:,jt]      # BE*Q*P2
      D[:,:] = [ zeros(p,m-τ)  QR.R[τ:-1:1,p:-1:1]' ]
      C[:,:] = C[QR.p,:]
      C[:,:] = reverse(C,dims=1)
      # Step 2:
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
             ismissing(R) || rmul!(R,G')
         end
         k += 1
      end
   else
      τ = 0
   end
   ρ = _preduce4!(n, m-τ, p-τ, M, N, Q, Z, tol, L, R, fast = fast, coff = coff, roff = roff, rtrail = rtrail+τ, ctrail = ctrail+τ, withQ = withQ, withZ = withZ)
   if m > 0
      jrt = npm-(τ+ρ)+1:npm
      DC[:,jrt] = reverse(DC[:,jrt],dims=1)
      DC[:,jrt] = DC[invperm(QR.p),jrt]
   end
   return τ, ρ 
end
"""
    _preduce3!(n::Int, m::Int, M::AbstractMatrix, N::AbstractMatrix, 
               Q::Union{AbstractMatrix,Nothing}, Z::Union{AbstractMatrix,Nothing}, tol,
               L::Union{AbstractMatrix{T},Missing} = missing, R::Union{AbstractMatrix{T},Missing} = missing; 
               fast = true, roff = 0, coff = 0, rtrail = 0, ctrail = 0,  withQ = true, withZ = true)

Reduce the structured pencil  

        [ *  *  *   *  ] roff        [ *  *  *   *  ] roff
    M = [ 0  B  A   *  ] n       N = [ 0  0  E   *  ] n
        [ 0  *  *   *  ] rtrail      [ 0  *  *   *  ] rtrail
        coff m  n ctrail             coff m  n ctrail

with `E` upper triangular and nonsingular to the following form `M1 - λN1 = Q1'*(M - λN)*Z1` with

         [ *  *   *   *     *  ] roff        [ *  *  *   *     *  ] roff
    M1 = [ 0  B1  A11 A12   *  ] ρ      N1 = [ 0  0  E11 E12   *  ] ρ
         [ 0  0   A21 A22   *  ] n-ρ         [ 0  0  0   E22   *  ] n-ρ
         [ 0  *   *   *     *  ] rtrail      [ 0  *  *   *     *  ] rtrail
         coff m   ρ   n-ρ ctrail             coff m  ρ   n-ρ ctrail

where `B1` has full row rank `ρ` and `E11` and `E22` are upper triangular and nonsingular. 
The performed orthogonal or unitary transformations are accumulated in `Q` (i.e., `Q <- Q*Q1`), if `withQ = true`, and 
`Z` (i.e., `Z <- Z*Z1`), if `withZ = true`. The rank decisions use the absolute tolerance `tol` for the nonzero elements of `M`.

The  matrix `L` is overwritten by `Q1'*L` unless `L = missing` and the  matrix `R` is overwritten by `R*Z1` unless `R = missing`. 
"""
function _preduce3!(n::Int, m::Int, M::AbstractMatrix{T}, N::AbstractMatrix{T}, 
                    Q::Union{AbstractMatrix{T},Nothing}, Z::Union{AbstractMatrix{T},Nothing}, tol::Real, 
                    L::Union{AbstractVecOrMat{T},Missing} = missing, R::Union{AbstractMatrix{T},Missing} = missing; 
                    fast::Bool = true, roff::Int = 0, coff::Int = 0, rtrail::Int = 0, ctrail::Int = 0,  
                    withQ::Bool = true, withZ::Bool = true) where T <: BlasFloat
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
   #  M1 = [ 0  B1  A11 A12   *  ] ρ      N1 = [ 0  0  E11 E12   *  ] ρ
   #       [ 0  0   A21 A22   *  ] n-ρ         [ 0  0  0   E22   *  ] n-ρ
   #       [ 0  *   *   *     *  ] rtrail      [ 0  *  *   *     *  ] rtrail
   #       coff m  n-ρ  ρ   ctrail             coff m  n-ρ ρ   ctrail
   #
   #  where B1 has full row rank and E11 and E22 are upper triangular and nonsingular. 
   #
   npm = n+m
   mM = roff + n + rtrail
   nM = coff + npm + ctrail
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
         if ind != j
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
             ismissing(L) || lmul!(G,view(L,ib,:))
             G, r = givens(conj(E[ii,ii]),conj(E[ii,iim1]),ii,iim1)
             E[ii,ii] = conj(r)
             E[ii,iim1] = ZERO 
             rmul!(view(N,1:roff+iim1,je),G')
             withZ && rmul!(view(Z,:,je),G') 
             rmul!(view(M,:,je),G')
             ismissing(R) || rmul!(view(R,:,je),G') 
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
            ismissing(L) || lmul!(G,view(L,ib,:))
            G, r = givens(conj(E[ii,ii]),conj(E[ii,iim1]),ii,iim1)
            E[ii,ii] = conj(r)
            E[ii,iim1] = ZERO 
            rmul!(view(N,1:roff+iim1,je),G')
            withZ && rmul!(view(Z,:,je),G') 
            rmul!(view(M,:,je),G')
            ismissing(R) || rmul!(view(R,:,je),G') 
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
            B[ics,jcs] = [ Diagonal(SVD.S[1:ρ])*SVD.Vt[1:ρ,:]; zeros(T,mn-ρ,m) ]
            if ρ == 0
               return ρ
            end
         end
         ibt = roff+1:roff+mn
         jt = coff+m+1:nM
         withQ && (Q[:,ibt] = Q[:,ibt]*SVD.U)
         N[ibt,jt] = SVD.U'*N[ibt,jt]
         M[ibt,jt] = SVD.U'*M[ibt,jt]
         ismissing(L) || (L[ibt,:] = SVD.U'*L[ibt,:])
         tau = similar(N,mn)
         jt1 = coff+m+1:coff+m+mn
         E11 = view(N,ibt,jt1)
         LinearAlgebra.LAPACK.gerqf!(E11,tau)
         T <: Complex ? tran = 'C' : tran = 'T'
         LinearAlgebra.LAPACK.ormrq!('R',tran,E11,tau,view(M,:,jt1))
         withZ && LinearAlgebra.LAPACK.ormrq!('R',tran,E11,tau,view(Z,:,jt1)) 
         LinearAlgebra.LAPACK.ormrq!('R',tran,E11,tau,view(N,1:roff,jt1))
         ismissing(R) || LinearAlgebra.LAPACK.ormrq!('R',tran,E11,tau,view(R,:,jt1)) 
         triu!(E11)
      else
         ρ = 0
      end
   end
   return ρ 
end
"""
    _preduce4!(n::Int, m::Int, p::Int, M::AbstractMatrix, N::AbstractMatrix, 
               Q::Union{AbstractMatrix,Nothing}, Z::Union{AbstractMatrix,Nothing}, tol,
               L::Union{AbstractMatrix{T},Missing} = missing, R::Union{AbstractMatrix{T},Missing} = missing; 
               fast = true, roff = 0, coff = 0, rtrail = 0, ctrail = 0,  withQ = true, withZ = true)

Reduce the structured pencil  

    M = [ *  *  *   *  ] roff          N = [ *  *  *   *  ] roff
        [ 0  B  A   *  ]  n                [ 0  0  E   *  ]  n
        [ 0  0  C   *  ]  p                [ 0  0  0   *  ]  p
        [ 0  *  *   *  ] rtrail            [ 0  *  *   *  ] rtrail
        coff m  n ctrail                   coff m  n ctrail

with `E` upper triangular and nonsingular to the following form `M1 - λN1 = Q1'*(M - λN)*Z1` with

    M1 = [ *  *   *   *    * ] roff         N1 = [ *  *  *   *    * ] roff
         [ 0  B1  A11 A12  * ]  n-ρ              [ 0  0  E11 E12  * ]  n-ρ
         [ 0  B2  A21 A22  * ]  ρ                [ 0  0  0   E22  * ]  ρ
         [ 0  0   0   C1   * ]  p                [ 0  0  0   0    * ]  p
         [ 0  *   *   *    * ] rtrail            [ 0  *  *   *    * ] rtrail
         coff m  n-ρ  ρ ctrail                   coff m n-ρ  ρ ctrail   

where `C1` has full column rank and `E11` and `E22` are upper triangular and nonsingular. 
The performed orthogonal or unitary transformations are accumulated in `Q` (i.e., `Q <- Q*Q1`), if `withQ = true`, and 
`Z` (i.e., `Z <- Z*Z1`), if `withZ = true`. The rank decisions use the absolute tolerance `tol` for the nonzero elements of `M`.

The  matrix `L` is overwritten by `Q1'*L` unless `L = missing` and the  matrix `R` is overwritten by `R*Z1` unless `R = missing`. 
"""
function _preduce4!(n::Int, m::Int, p::Int, M::AbstractMatrix{T},N::AbstractMatrix{T},
                    Q::Union{AbstractMatrix{T},Nothing}, Z::Union{AbstractMatrix{T},Nothing}, tol::Real,  
                    L::Union{AbstractVecOrMat{T},Missing} = missing, R::Union{AbstractMatrix{T},Missing} = missing; 
                    fast::Bool = true, roff::Int = 0, coff::Int = 0, rtrail::Int = 0, ctrail::Int = 0, 
                    withQ::Bool = true, withZ::Bool = true) where T <: BlasFloat
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
   #  M1 = [ *  *   *   *    * ] roff         N1 = [ *  *  *   *    * ] roff
   #       [ 0  B1  A11 A12  * ]  n-ρ              [ 0  0  E11 E12  * ]  n-ρ
   #       [ 0  B2  A21 A22  * ]  ρ                [ 0  0  0   E22  * ]  ρ
   #       [ 0  0   0   C1   * ]  p                [ 0  0  0   0    * ]  p
   #       [ 0  *   *   *    * ] rtrail            [ 0  *  *   *    * ] rtrail
   #       coff m  n-ρ  ρ ctrail                   coff m n-ρ  ρ ctrail   
   #
   #  where C1 has full column rank and E11 and E22 are upper triangular and nonsingular. 
   npp = n+p
   npm = n+m
   mM = roff + npp + rtrail
   nM = coff + npm + ctrail
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
         if ind != ii
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
             ismissing(R) || rmul!(view(R,:,jc),G')
             G, E[jj,jj] = givens(E[jj,jj],E[jjp1,jj],jj,jjp1)
             E[jjp1,jj] = ZERO
             lmul!(G,view(N,ie,coff+m+jjp1:nM))
             withQ && rmul!(view(Q,:,ie),G') 
             lmul!(G,view(M,ie,coff+1:nM))
             ismissing(L) || lmul!(G,view(L,ie,:))
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
                ismissing(R) || rmul!(view(R,:,jc),G')
                G, E[jj,jj] = givens(E[jj,jj],E[jjp1,jj],jj,jjp1)
                E[jjp1,jj] = ZERO
                lmul!(G,view(N,ie,coff+m+jjp1:nM))
                withQ && rmul!(view(Q,:,ie),G') 
                lmul!(G,view(M,ie,coff+1:nM))
                ismissing(L) || lmul!(G,view(L,ie,:))
               end
         end
      end
      pn = min(n,p)
      if pn > 0
         ics = 1:p
         jcs = n-pn+1:n
         SVD = svd(C[ics,jcs], full = true)
         ρ = count(x -> x > tol, SVD.S) 
         if ρ == pn
            return ρ
         else
            Q1 = reverse(SVD.U,dims=2)
            C[ics,jcs] = [ zeros(T,p,pn-ρ) Q1[:,p-ρ+1:end]*Diagonal(reverse(SVD.S[1:ρ])) ] 
            if ρ == 0
               return ρ
            end
         end
         Z1 = reverse(SVD.V,dims=2)
         jt = coff+npm-pn+1:coff+npm
         withZ && (Z[:,jt] = Z[:,jt]*Z1) 
         M[1:roff+n,jt] = M[1:roff+n,jt]*Z1
         N[1:roff+n,jt] = N[1:roff+n,jt]*Z1    # more efficient computation possible
         ismissing(R) || (M[:,jt] = M[:,jt]*Z1)
         it = roff+n-pn+1:roff+n
         jt1 = coff+n+m+1:nM
         tau = similar(N,pn)
         E22 = view(N,it,jt)
         LinearAlgebra.LAPACK.geqrf!(E22,tau)
         T <: Complex ? tran = 'C' : tran = 'T'
         LinearAlgebra.LAPACK.ormqr!('L',tran,E22,tau,view(M,it,coff+1:nM))
         withQ && LinearAlgebra.LAPACK.ormqr!('R','N',E22,tau,view(Q,:,it)) 
         LinearAlgebra.LAPACK.ormqr!('L',tran,E22,tau,view(N,it,jt1))
         ismissing(L) || LinearAlgebra.LAPACK.ormqr!('L',tran,E22,tau,view(L,it,:))
         triu!(E22)
      else
         ρ = 0
      end
   end
   return ρ 
end

