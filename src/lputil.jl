"""
    pbalance!(M, N; r, c, regpar, shift, maxiter, tol, pow2) -> (Dl, Dr)

Balance the `m×n` matrix pencil `M - λN` by reducing the 1-norm of the matrix `T := abs(M)+abs(N)`
by row and column balancing. 
This involves similarity transformations with diagonal matrices `Dl` and `Dr` applied
to `T` to make the rows and columns of `Dl*T*Dr` as close in norm as possible.
The modified [Sinkhorn–Knopp algorithm](https://en.wikipedia.org/wiki/Sinkhorn%27s_theorem) described in [1] 
is employed to reduce `T` to an approximately doubly stochastic matrix. 
The targeted row and column sums can be specified using the keyword arguments `r = rs` and `c = cs`, 
where `rs` and `cs` are `m-` and `n-`dimensional positive vectors, 
representing the desired row and column sums, respectively (Default: `rs = ones(m)` and `cs = ones(n)`).

The resulting `Dl` and `Dr` are diagonal scaling matrices.  
If the keyword argument `pow2 = true` is specified, then the components of the resulting 
optimal `Dl` and `Dr` are replaced by their nearest integer powers of 2. 
If `pow2 = false`, the optimal values `Dl` and `Dr` are returned.
The resulting `Dl*M*Dr` and `Dl*N*Dr` overwrite `M` and `N`, respectively

A regularization-based scaling is performed if a nonzero regularization parameter `α` is specified
via the keyword argument `regpar = α`.  
If `diagreg = true`, then the balancing algorithm is performed on
the extended symmetric matrix `[ α^2*I  T; T' α^2*I ]`, while if `diagreg = false` (default),   
the balancing algorithm is performed on the matrix `[ (α/m)^2*em*em'  T; T' (α/n)^2*en*en' ]`, 
where `em` and `en` are `m-` and `n-`dimensional vectors with elements equal to one.  
If `α = 0` and `shift = γ > 0` is specified, then the algorithm is performed on
the rank-one perturbation `T+γ*em*en`. 

The keyword argument `tol = τ`, with `τ ≤ 1`,  specifies the tolerance used in the stopping criterion. 
The iterative process is stopped as soon as the incremental scalings are `tol`-close to the identity. 

The keyword argument `maxiter = k` specifies the maximum number of iterations `k` 
allowed in the balancing algorithm. 

_Method:_ This function employs the regularization approaches proposed in [1], modified 
to handle matrices with zero rows or zero columns. The alternative shift based regularization 
has been proposed in [2].  

[1] F.M.Dopico, M.C.Quintana and P. van Dooren, 
    "Diagonal scalings for the eigenstructure of arbitrary pencils", SIMAX, 43:1213-1237, 2022. 

[2] P.A.Knight, The Sinkhorn–Knopp algorithm: Convergence and applications, SIAM J. Matrix
    Anal. Appl., 30 (2008), pp. 261–275.   
"""
function pbalance!(M::AbstractMatrix{T}, N::AbstractMatrix{T}; regpar = 0, shift = 0, diagreg = false, 
                   r = fill(one(T),size(M,1)), c = fill(one(T),size(M,2)),                 
                   maxiter = 100, tol = 1, pow2 = true) where {T}
   m, n = size(M)
   (m,n) != size(N) && throw(DimensionMismatch("M and N must have the same dimensions"))
   if m == 0 && n == 0 
      return Diagonal(T[]), Diagonal(T[])
   elseif n == 0 
      return Diagonal(ones(T,m)), Diagonal(T[])
   elseif m == 0
      return Diagonal(T[]), Diagonal(ones(T,n))
   end
   α = regpar
   if α == 0
      Dl, Dr = rcsumsbal!(abs.(M)+abs.(N); shift, r, c, maxiter, tol)
   else
      W = abs.(M)+abs.(N)
      rc = [r;c]
      if diagreg
         dleft, dright = rcsumsbal!([α^2*I W; W' α^2*I]; r = rc, c = rc, maxiter, tol)
      else
         dleft, dright = rcsumsbal!([fill((α/m)^2,m,m) W; W' fill((α/n)^2,n,n)]; r = rc, c = rc, maxiter, tol)
      end
      Dl = Diagonal(dleft.diag[1:m]); Dr = Diagonal(dright.diag[m+1:m+n])
   end
   if pow2 
      radix = real(T)(2.)
      Dl.diag .= radix .^(round.(Int,log2.(Dl.diag)))
      Dr.diag .= radix .^(round.(Int,log2.(Dr.diag)))
   end
   lmul!(Dl,M); rmul!(M,Dr)
   lmul!(Dl,N); rmul!(N,Dr)
   return Dl, Dr
end
"""
    qs = pbalqual(M, N) 

Compute the 1-norm based scaling quality of a matrix pencil `M-λN`.

The resulting `qs` is computed as 

        qs = qS(abs(M)+abs(N)) ,

where `qS(⋅)` is the scaling quality measure defined in Definition 5.5 of [1] for 
nonnegative matrices. This definition has been extended to also cover matrices with
zero rows or columns. If `N = I`, `qs = qs(M)` is computed. 

A large value of `qs` indicates a possibly poorly scaled matrix pencil.   

[1] F.M.Dopico, M.C.Quintana and P. van Dooren, 
    "Diagonal scalings for the eigenstructure of arbitrary pencils", SIMAX, 43:1213-1237, 2022. 
"""
function pbalqual(A::AbstractMatrix{T}, E::Union{AbstractMatrix{T},UniformScaling{Bool}}) where {T}
   if (!(typeof(E) <: AbstractMatrix) || isequal(E,I)) 
      return qS1(A)
   else
      return qS1(abs.(A).+abs.(E))
   end
end 
"""
    rcsumsbal!(M; shift, r, c, maxiter, tol) -> (Dl,Dr)

Perform the Sinkhorn-Knopp-like algorithm to scale 
a non-negative matrix `M` such that `Dl*M*Dr`
has column sums equal to a positive row vector `cs` and row sums equal to a 
positive column vector `rs`, where `sum(c) = sum(r)`. 
If shift = γ > 0 is specified, the algorithm is performed on
the rank-one perturbation `M+γ*e1*e2`,  where `e1` and `e2` are vectors 
of ones of appropriate dimensions. 
The iterative process is stopped as soon as the incremental
scalings are `tol`-close to the identity.  
`maxiter` is the maximum number of allowed iterations and `tol` is the
tolerance for the transformation updates. 

The resulting `Dl*M*Dr` overwrites `M` and is a matrix with equal row sums and 
equal column sums. `Dl` and `Dr` are the diagonal scaling matrices. 

_Note:_ This function is based on the MATLAB function `rowcolsums.m` of [1], modified 
to handle matrices with zero rows or columns. The implemented shift based regularization 
has been proposed in [2].  

[1] F.M.Dopico, M.C.Quintana and P. van Dooren, 
    "Diagonal scalings for the eigenstructure of arbitrary pencils", SIMAX, 43:1213-1237, 2022. 

[2] P.A.Knight, The Sinkhorn–Knopp algorithm: Convergence and applications, SIAM J. Matrix
    Anal. Appl., 30 (2008), pp. 261–275.   
"""
function rcsumsbal!(M::AbstractMatrix{T}; r::AbstractVector{T} = fill(T(size(M,1)),size(M,1)), 
                                          c::AbstractVector{T} = fill(T(size(M,2)),size(M,2)), maxiter = 100, tol = 1., shift = 0, pow2 = true) where {T}
   m, n = size(M); 
   any(M .< 0) && throw(ArgumentError("matrix M must have nonnegative elements"))
   m == length(r) || throw(ArgumentError("dimension of row sums vector r must be equal to row size of M"))
   n == length(c) || throw(ArgumentError("dimension of column sums vector c must be equal to column size of M"))
   shift < 0 && throw(ArgumentError("the shift must be nonnegative"))

   radix = real(T)(2.)
   # handle zero rows and columns  
   sumr = reshape(sum(M,dims=2),m)
   indr = findall(!iszero,sumr); indrz = findall(iszero,sumr); m1 = length(indr); nrows = m1 == m
   sumc = reshape(sum(M,dims=1),n)
   indc = findall(!iszero,sumc); indcz = findall(iszero,sumc); n1 = length(indc); ncols = n1 == n
   # exclude zero rows and columns: no copying performed if all rows/columns are nonzero
   M1 = (nrows && ncols) ? M : view(M,indr,indc)
   r1 = nrows ? r : view(r,indr)
   c1 = ncols ? c : view(c,indc)
   # scale the matrix to have total sum(sum(M))=sum(c)=sum(r);
   sumcr = sum(c1); 
   sumM = sum(M1); 
   sc = sumcr/sumM
   lmul!(sc,M1)
   t = sqrt(sc); 
   dleft = Vector{T}(undef,m); 
   dleft1 = nrows ? dleft : view(dleft,indr)
   fill!(dleft1,t); fill!(view(dleft,indrz),one(T))
   dright = Vector{T}(undef,n); 
   dright1 = ncols ? dright : view(dright,indc)
   fill!(dright1,t); fill!(view(dright,indcz),one(T))
   m1 == 0 && n1 == 0 && (return Diagonal(dleft), Diagonal(dright))

   # Scale left and right to make row and column sums equal to r and c
   conv = false
   for i = 1:maxiter
       conv = true
       dr = reshape(sum(M1,dims=1) .+ m1*shift,n1) ./ c1; #@show dr
       rdiv!(M1,Diagonal(dr)); 
       er = minimum(dr)/maximum(dr); dright1 ./= dr
       dl = reshape(sum(M1,dims=2) .+ n1*shift,m1) ./r1; #@show dl
       ldiv!(Diagonal(dl),M1); 
       el = minimum(dl)/maximum(dl); dleft1 ./= dl
       #@show i, er, el
       max(1-er,1-el) < tol/2 && break
       conv = false
   end
   conv || (@warn "the iterative algorithm did not converge in $maxiter iterations")
   # Finally scale the two scalings to have equal maxima
   scaled = sqrt(maximum(dright1)/maximum(dleft1))
   dleft1 .= dleft1*scaled; dright1 .= dright1/scaled
   return Diagonal(dleft), Diagonal(dright)
end
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

where τ = rank D, B1 is full row rank, and E22 is upper triangular and nonsingular.
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
   #  where τ = rank D, B1 is full row rank and E22 is upper triangular and nonsingular.

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
   T <: Complex ? tran = 'C' : tran = 'T'
   if p > 0
      # compress D to [D1 D2;0 0] with D1 invertible 
      ic = roff+n+1:roff+npp
      D = view(M,ic,jb)
      CE = view(M,ic,ja)
      EE = view(N,ic,coff+npm+1:nM)
      if fast
         #QR = qr!(D, Val(true))
         #τ = count(x -> x > tol, abs.(diag(QR.R))) 
         _, τau, jpvt = LinearAlgebra.LAPACK.geqp3!(D)
         τ = count(x -> x > tol, abs.(diag(D))) 
      else
         τ = count(x -> x > tol, svdvals(D))
         #QR = qr!(D, Val(true))
         _, τau, jpvt = LinearAlgebra.LAPACK.geqp3!(D)
      end
      #B[:,:] = B[:,QR.p]
      B[:,:] = B[:,jpvt]
      #lmul!(QR.Q',CE)
      #lmul!(QR.Q',EE)
      # N23 <- Q2'*N23
      LinearAlgebra.LAPACK.ormqr!('L',tran,D,τau,CE)
      LinearAlgebra.LAPACK.ormqr!('L',tran,D,τau,EE)
      #withQ && rmul!(view(Q,:,ic),QR.Q) 
      withQ && LinearAlgebra.LAPACK.ormqr!('R','N',D,τau,view(Q,:,ic))
      #ismissing(L) || lmul!(QR.Q',view(L,ic,:))
      ismissing(L) || LinearAlgebra.LAPACK.ormqr!('L',tran,D,τau,view(L,ic,:))
      #D[:,:] = [ QR.R[1:τ,:]; zeros(T,p-τ,m) ]
      D[:,:] = [ triu(D[1:τ,:]); zeros(T,p-τ,m)  ]
      # if fast
      #    QR = qr!(D, Val(true))
      #    τ = count(x -> x > tol, abs.(diag(QR.R))) 
      # else
      #    τ = count(x -> x > tol, svdvals(D))
      #    QR = qr!(D, Val(true))
      # end
      # B[:,:] = B[:,QR.p]
      # lmul!(QR.Q',CE)
      # lmul!(QR.Q',EE)
      # withQ && rmul!(view(Q,:,ic),QR.Q) 
      # ismissing(L) || lmul!(QR.Q',view(L,ic,:))
      # D[:,:] = [ QR.R[1:τ,:]; zeros(T,p-τ,m) ]
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
      #BD[irt,:] = BD[irt,invperm(QR.p)]
      BD[irt,:] = BD[irt,invperm(jpvt)]
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
         # QR = qr!(copy(D'), Val(true))
         # τ = count(x -> x > tol, abs.(diag(QR.R))) 
         DT, τau, jpvt = LinearAlgebra.LAPACK.geqp3!(copy(D'))
         τ = count(x -> x > tol, abs.(diag(DT))) 
      else
         # τ = rank(D; atol = tol)
         τ = count(x -> x > tol, svdvals(D))
         # QR = qr!(copy(D'), Val(true))
         DT, τau, jpvt = LinearAlgebra.LAPACK.geqp3!(copy(D'))
      end
      #rmul!(BE,QR.Q)  # BE*Q
      LinearAlgebra.LAPACK.ormqr!('R','N',DT,τau,BE)
      jt = m:-1:1
      BE[:,:] = BE[:,jt]      # BE*Q*P2
      if withZ 
         Z1 = view(Z,:,jb)
         # rmul!(Z1,QR.Q) 
         LinearAlgebra.LAPACK.ormqr!('R','N',DT,τau,Z1)
         Z1[:,:] = Z1[:,jt]
      end
      #ismissing(R) || ( rmul!(view(R,:,jb),QR.Q); R[:,jb] = R[:,reverse(jb)] )
      ismissing(R) || ( LinearAlgebra.LAPACK.ormqr!('R','N',DT,τau,view(R,:,jb)); R[:,jb] = R[:,reverse(jb)] )
      #rmul!(EE,QR.Q)  
      LinearAlgebra.LAPACK.ormqr!('R','N',DT,τau,EE)
      EE[:,:] = EE[:,jt]      # BE*Q*P2
      #D[:,:] = [ zeros(p,m-τ)  QR.R[τ:-1:1,p:-1:1]' ]
      D[:,:] = [ zeros(p,m-τ)  triu(DT)[τ:-1:1,p:-1:1]' ]
      C[:,:] = C[jpvt,:]
      C[:,:] = reverse(C,dims=1)

      # if fast
      #    QR = qr!(copy(D'), Val(true))
      #    τ = count(x -> x > tol, abs.(diag(QR.R))) 
      # else
      #    # τ = rank(D; atol = tol)
      #    τ = count(x -> x > tol, svdvals(D))
      #    QR = qr!(copy(D'), Val(true))
      # end
      # rmul!(BE,QR.Q)  # BE*Q
      # jt = m:-1:1
      # BE[:,:] = BE[:,jt]      # BE*Q*P2
      # if withZ 
      #    Z1 = view(Z,:,jb)
      #    rmul!(Z1,QR.Q) 
      #    Z1[:,:] = Z1[:,jt]
      # end
      # ismissing(R) || ( rmul!(view(R,:,jb),QR.Q); R[:,jb] = R[:,reverse(jb)] )
      # rmul!(EE,QR.Q)  
      # EE[:,:] = EE[:,jt]      # BE*Q*P2
      # D[:,:] = [ zeros(p,m-τ)  QR.R[τ:-1:1,p:-1:1]' ]
      # C[:,:] = C[QR.p,:]
      # C[:,:] = reverse(C,dims=1)

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
      # DC[:,jrt] = DC[invperm(QR.p),jrt]
      DC[:,jrt] = DC[invperm(jpvt),jrt]
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

