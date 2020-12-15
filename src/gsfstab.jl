"""
    salocd(A, C; evals, sdeg, disc = false, atol1 = 0, atol2 = 0, rtol) -> (K, Scl, blkdims)

Compute for the pair `(A,C)`, a matrix `K` such that all eigenvalues of the matrix `A+K*C` lie in the stability domain `Cs` 
specified by the stability degree parameter `sdeg` and stability type parameter `disc`. 
If `disc = false`, `Cs` is the set of complex numbers with real parts at most `sdeg`, while if `disc = true`, 
`Cs` is the set of complex numbers with moduli at most `sdeg` (i.e., the interior of a disc of radius `sdeg` centered in the origin). 
`evals` is a real or complex vector, which contains the desired eigenvalues of the matrix `A+K*C` within `Cs`. 
For real data `A` and `C`, `evals` must be a self-conjugated complex set to ensure that the resulting `K` is also a real matrix. 

For a pair `(A,C)` with `A` of order `n`, the number of assignable eigenvalues is `nc := n-nu`,  
where `nu` is the number of fixed eigenvalues of `A`. The assignable eigenvalues are called the _observable eigenvalues_, 
while the fixed eigenvalues are called the _unobservable eigenvalues_ (these are the zeros of the pencil `[A-λI; C]`). 
The spectrum allocation is achieved by successively replacing the observable eigenvalues of `A` lying outside of the 
stability domain `Cs` with eigenvalues provided in `evals`. All eigenvalues of `A` lying in `Cs` are kept unalterred.  
If the number of specified eigenvalues in `evals` is less than the number of observable eigenvalues of `A` outside of `Cs`
(e.g., if `evals = missing`), then some of the observable eigenvalues of `A` are assigned to the nearest values 
on the boundary of `Cs`. If `sdeg = missing` and `evals = missing`, the default value used for `sdeg` is -0.05 
if  `disc = false` and 0.95 if `disc = true`. 

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of `A`, the absolute tolerance for the nonzero elements of `C`,  
and the relative tolerance for the nonzero elements of `A` and `C`.  
The default relative tolerance is `n*ϵ`, where `ϵ` is the machine epsilon of the element type of `A`. 

The resulting matrix `Acl := Z'*(A+K*C)*Z` in a Schur form, where `Z` is the orthogonal/unitary matrix used 
to obtain the matrix `A+K*C` in Schur form, has the form

           ( Au  *   *  )     
     Acl = ( 0   Aa  *  ) 
           ( 0   0   Ag )    

where:  `Au` contains `nu` unobservable eigenvalues of `A` lying outside `Cs`, 
`Aa` contains the `na` assigned eigenvalues in `Cs` and `Ag` contains the `ng` eigenvalues of `A` in `Cs`. 
The matrices `Acl` and `Z` and the vector `α` of eigenvalues of `Acl` (also of `A+K*C`)
are returned in the `Schur` object `Scl`. 
The values of `nu`, `na` and `ng` are returned in the 3-dimensional vector `blkdims = [nu, na, ng]`.

Method:  The Schur method of [1] is applied to the dual pair `(A',C')` (extended to possibly unobservable pairs).  

References:

[1] A. Varga. 
    A Schur method for pole assignment.
    IEEE Trans. on Automatic Control, vol. 26, pp. 517-519, 1981.
"""
function salocd(A::AbstractMatrix, C::AbstractMatrix; disc::Bool = false, 
                evals::Union{AbstractVector,Missing} = missing, sdeg::Union{Real,Missing} = missing, 
                atol1::Real = zero(real(eltype(A))), atol2::Real = zero(real(eltype(C))),  
                rtol::Real = ((size(A,1)+1)*eps(real(float(one(eltype(A))))))*iszero(max(atol1,atol2)))

   n = LinearAlgebra.checksquare(A)
   n == size(C,2) || throw(DimensionMismatch("A and C must have the same number of columns"))
   K, S, blkdims = saloc(copy(transpose(A)), copy(transpose(C)), evals = evals, sdeg = sdeg, atol1 = atol1, atol2 = atol2, rtol = rtol)
   return reverse(transpose(K), dims = 2), Schur(reverse(reverse(transpose(S.T),dims = 1),dims = 2), reverse(conj(S.Z),dims=2), reverse(S.values)), reverse(blkdims)
end
"""
    salocd(A, E, C; evals, sdeg, disc = false, atol1 = 0, atol2 = 0, atol3 = 0, rtol, sepinf = true, fast = true) -> (K, Scl, blkdims)

Compute for the pair `(A-λE,C)`, with `A-λE` a regular pencil, a matrix `K` such that all finite eigenvalues of the 
pencil `A+K*C-λE` lie in the stability domain `Cs` specified by the stability degree parameter `sdeg` and stability type parameter `disc`. 
If `disc = false`, `Cs` is the set of complex numbers with real parts at most `sdeg`, while if `disc = true`, 
`Cs` is the set of complex numbers with moduli at most `sdeg` (i.e., the interior of a disc of radius `sdeg` centered in the origin). 
`evals` is a real or complex vector, which contains a set of finite desired eigenvalues for the pencil `A+K*C-λE`. 
For real data `A`, `E`, and `B`, `evals` must be a self-conjugated complex set to ensure that the resulting `F` is also a real matrix. 

For a pair `(A-λE,C)` with `A` of order `n`, the number of assignable finite eigenvalues is `nfc := n-ninf-nfu`,  
where `ninf` is the number of infinite eigenvalues of `A-λE` and `nfu` is the number of fixed finite eigenvalues of `A-λE`. 
The assignable finite eigenvalues are called the _observable finite eigenvalues_, 
while the fixed finite eigenvalues are called the _unobservable finite eigenvalues_ (these are the finite zeros of the pencil `[A-λE; C]`). 
The spectrum allocation is achieved by successively replacing the observable finite eigenvalues of `A-λE` lying outside of the 
stability domain `Cs` with eigenvalues provided in `evals`. All finite eigenvalues of `A-λE` lying in `Cs` are kept unalterred.  
If the number of specified eigenvalues in `evals` is less than the number of observable finite eigenvalues of `A-λE` outside of `Cs`
(e.g., if `evals = missing`), then some of the observable finite eigenvalues of `A-λE` are assigned to the nearest values 
on the boundary of `Cs`. If `sdeg = missing` and `evals = missing`, the default value used for `sdeg` is -0.05 
if  `disc = false` and 0.95 if `disc = true`. 

The keyword arguments `atol1`, `atol2`, , `atol3`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of `A`, the absolute tolerance for the nonzero elements of `E`, the absolute tolerance for the nonzero elements of `C`,  
and the relative tolerance for the nonzero elements of `A`, `E` and `C`.  
The default relative tolerance is `n*ϵ`, where `ϵ` is the machine epsilon of the element type of `A`. 

The keyword argument `sepinf` specifies the option for a preliminary separation 
of the infinite eigenvalues of the pencil `A-λE` as follows: if `sepinf = false`, no separation of infinite eigenvalues is performed, 
while for `sepinf = true` (the default option), a preliminary separation of the infinite eigenvalues from the finite ones is performed.
If `E` is nonsingular, then `sepinf = false` is recommended to be used. If `E` is numerically singular, then the option `sepinf = false` is used. 
The separation of finite and infinite eigenvalues is performed using rank decisions based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.

The resulting pencil `Acl-λEcl := Q'*(A+K*C-λE)*Z`,  where `Q` and `Z` are the  
orthogonal/unitary matrices used to obtain the pair `(Acl,Ecl)` in a generalized Schur form (GSF), has the form

                ( Afu-λEfu    *        *       *    )     
     Acl-λEcl = (   0      Afa-λEfa    *       *    )  
                (   0         0     Afg-λEfg   *    )    
                (   0         0        0     Ai-λEi )    

where: `Afu-λEfu` contains `nfu` unobservable finite eigenvalues of `A-λE` lying outside `Cs`,
`Afa-λEfa` contains `nfa` assigned finite generalized eigenvalues in `Cs`, 
`Afg-λEfg` contains `nfg` finite eigenvalues of `A-λE` in `Cs`, and 
`Ai-λEi`, with `Ai` upper triangular and invertible and `Ei` upper triangular and nilpotent, 
contains the `ninf` infinite eigenvalues of `A-λE`.  
The matrices `Acl`, `Ecl`, `Q`, `Z` and the vectors `α` and `β` such that `α./β` are the generalized eigenvalues of 
the pair `(Acl,Ecl)` are returned in `GeneralizeSchur` object `Scl`. 
The values of `nfu`, `nfa`, `nfg` and `ninf` and are returned in the 4-dimensional vector `blkdims = [nfu, nfa, nfg, ninf]`.

Method:  For a pair `(A-λE,C)` with `E = I`, the dual Schur method of [1] is used, while for a general pair `(A-λE,C)` the dual generalized Schur method of [2] 
is used to solve the R-stabilzation problem of [2] for the dual pair `(A'-λE',C')`.

References:

[1] A. Varga. 
    A Schur method for pole assignment.
    IEEE Trans. on Automatic Control, vol. 26, pp. 517-519, 1981.

[2] A. Varga. 
    On stabilization methods of descriptor systems.
    Systems & Control Letters, vol. 24, pp.133-138, 1995.
"""
function salocd(A::AbstractMatrix, E::Union{AbstractMatrix,UniformScaling{Bool}}, C::AbstractMatrix; 
                evals::Union{AbstractVector,Missing} = missing, sdeg::Union{Real,Missing} = missing, 
                atol1::Real = zero(real(eltype(A))), atol2::Real = zero(real(eltype(A))), atol3::Real = zero(real(eltype(C))), 
                rtol::Real = ((size(A,1)+1)*eps(real(float(one(eltype(A))))))*iszero(max(atol1,atol2,atol3)), 
                fast::Bool = true, sepinf::Bool = true)

   n = LinearAlgebra.checksquare(A) 
   E == I || LinearAlgebra.checksquare(E) == n || throw(DimensionMismatch("E must be a $n x $n matrix"))

   n == size(C,2) || throw(DimensionMismatch("A and C must have the same number of columns"))
   K, S, blkdims = saloc(copy(transpose(A)), copy(transpose(E)), copy(transpose(C)), evals = evals, sdeg = sdeg, atol1 = atol1, atol2 = atol2, atol3 = atol3, sepinf = sepinf, fast = fast, rtol = rtol)
   return reverse(transpose(K), dims = 2), GeneralizedSchur(reverse(reverse(transpose(S.S),dims = 1),dims = 2), reverse(reverse(transpose(S.T),dims = 1),dims = 2), 
                                                  reverse(S.α), reverse(S.β), reverse(conj(S.Z),dims=2), reverse(conj(S.Q),dims=2)), reverse(blkdims)
end
"""
    saloc(A, B; evals, sdeg, disc = false, atol1 = 0, atol2 = 0, rtol) -> (F, Scl, blkdims)

Compute for the pair `(A,B)`, a matrix `F` such that all eigenvalues of the matrix `A+B*F` lie in the stability domain `Cs` 
specified by the stability degree parameter `sdeg` and stability type parameter `disc`. 
If `disc = false`, `Cs` is the set of complex numbers with real parts at most `sdeg`, while if `disc = true`, 
`Cs` is the set of complex numbers with moduli at most `sdeg` (i.e., the interior of a disc of radius `sdeg` centered in the origin). 
`evals` is a real or complex vector, which contains the desired eigenvalues of the matrix `A+B*F` within `Cs`. 
For real data `A` and `B`, `evals` must be a self-conjugated complex set to ensure that the resulting `F` is also a real matrix. 

For a pair `(A,B)` with `A` of order `n`, the number of assignable eigenvalues is `nc := n-nu`,  
where `nu` is the number of fixed eigenvalues of `A`. The assignable eigenvalues are called the controllable eigenvalues, 
while the fixed eigenvalues are called the _uncontrollable eigenvalues_ (these are the zeros of the pencil `[A-λI B]`). 
The spectrum allocation is achieved by successively replacing the _controllable eigenvalues_ of `A` lying outside of the 
stability domain `Cs` with eigenvalues provided in `evals`. All eigenvalues of `A` lying in `Cs` are kept unalterred.  
If the number of specified eigenvalues in `evals` is less than the number of controllable eigenvalues of `A` outside of `Cs`
(e.g., if `evals = missing`), then some of the controllable eigenvalues of `A` are assigned to the nearest values 
on the boundary of `Cs`. If `sdeg = missing` and `evals = missing`, the default value used for `sdeg` is -0.05 
if  `disc = false` and 0.95 if `disc = true`. 

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of `A`, the absolute tolerance for the nonzero elements of `B`,  
and the relative tolerance for the nonzero elements of `A` and `B`.  
The default relative tolerance is `n*ϵ`, where `ϵ` is the machine epsilon of the element type of `A`. 

The resulting matrix `Acl := Z'*(A+B*F)*Z` in a Schur form, where `Z` is the orthogonal/unitary matrix used 
to obtain the matrix `A+B*F` in Schur form, has the form

           ( Ag  *   *  )     
     Acl = ( 0   Aa  *  ) 
           ( 0   0   Au )    

where:  `Ag` contains the `ng` eigenvalues of `A` in `Cs`, `Aa` contains the `na` assigned eigenvalues in `Cs` and
`Au` contains `nu` uncontrollable eigenvalues of `A` lying outside `Cs`. 
The matrices `Acl` and `Z` and the vector `α` of eigenvalues of `Acl` (also of `A+B*F`)
are returned in the `Schur` object `Scl`. 
The values of `ng`, `na` and `nu` are returned in the 3-dimensional vector `blkdims = [ng, na, nu]`.

Method:  The Schur method of [1], extended to possibly uncontrollable pairs, is employed.  

References:

[1] A. Varga. 
    A Schur method for pole assignment.
    IEEE Trans. on Automatic Control, vol. 26, pp. 517-519, 1981.
"""
function saloc(A::AbstractMatrix, B::AbstractMatrix; disc::Bool = false, 
               evals::Union{AbstractVector,Missing} = missing, sdeg::Union{Real,Missing} = missing, 
               atol1::Real = zero(real(eltype(A))), atol2::Real = zero(real(eltype(B))),  
               rtol::Real = ((size(A,1)+1)*eps(real(float(one(eltype(A))))))*iszero(max(atol1,atol2)))

   n = LinearAlgebra.checksquare(A)
   n1, m = size(B)
   n == n1 || throw(DimensionMismatch("A and B must have the same number of rows"))
   
   T = promote_type( eltype(A), eltype(B) )
   T <: BlasFloat || (T = promote_type(Float64,T))
   
   A1 = copy_oftype(A,T)   
   B1 = copy_oftype(B,T)
   if ismissing(evals)
      evals1 = evals
   else 
      T1 = promote_type(T,eltype(evals)) 
      evals1 = copy_oftype(evals,T1)
   end
   
   # quick exit if possible 
   n == 0 && (return zeros(T,m,n), schur(zeros(T,n,n)), zeros(Int,4))

   ZERO = zero(T)

   # check for zero rows in the leading positions
   ilob = n+1
   for i = 1:n
       !iszero(view(B1,i,:)) && (ilob = i; break)
   end

   # return if B = 0
   ilob > n && (return zeros(T,m,n), schur(A1), [0,0,n] )

   # check for zero rows in the trailing positions
   ihib = ilob
   for i = n:-1:ilob+1
       !iszero(view(B1,i,:)) && (ihib = i; break)
   end
   
   # operate only on the nonzero rows of B
   ib = ilob:ihib
   nrmB = opnorm(view(B1,ib,:),1)
    
   complx = (T <: Complex)
    
   # sort desired eigenvalues
   if ismissing(evals1) 
      evalsr = missing
      evalsc = missing
   else
      if complx
         evalsr = copy(evals1)
         evalsc = missing
      else
         evalsr = evals1[imag.(evals1) .== 0]
         isempty(evalsr) && (evalsr = missing)
         tempc = evals1[imag.(evals1) .> 0]
         if isempty(tempc)
            evalsc = missing
         else
            tempc1 = conj(evals1[imag.(evals1) .< 0])
            isequal(tempc[sortperm(real(tempc))],tempc1[sortperm(real(tempc1))]) ||
                    error("evals must be a self-conjugated complex vector")
            evalsc = [transpose(tempc[:]); transpose(conj.(tempc[:]))][:]
         end
      end
      # check that all eigenvalues are inside of the stability region
      !ismissing(sdeg) && ((disc && any(abs.(evals1) .> sdeg) )  || (!disc && any(real.(evals1) .> sdeg)))  &&
            error("The elements of evals must lie in the stability region")
   end    
  
   # set default values of sdeg if evals = missing
   if ismissing(sdeg)
      if ismissing(evals1) 
        sdeg = disc ? real(T)(0.95) : real(T)(-0.05)
        smarg = sdeg;         
      else
         smarg = disc ? real(T)(0) : real(T)(-Inf) 
      end
   else
      sdeg = real(T)(sdeg)
      disc && sdeg < 0 && error("sdeg must be non-negative if disc = true")
      smarg = sdeg;  
   end
  
   nrmA = opnorm(A1,1)
   tola = max(atol1, rtol*nrmA)
   tolb = max(atol2, rtol*nrmB)
   
   #
   # separate stable and unstable parts with respect to sdeg
   # compute orthogonal Z  such that
   #
   #      Z^T*A*Z = [ Ag   * ]
   #                [  0  Ab ]
   #
   # where Ag has eigenvalues within the stability degree region
   # and Ab has eigenvalues outside the stability degree region.
    
   _, Z, α = LAPACK.gees!('V', A1)

   if disc
      select = Int.(abs.(α) .<= smarg)
   else
      select = Int.(real.(α) .<= smarg)
   end
   _, _, α = LAPACK.trsen!(select, A1, Z) 
   nb = length(select[select .== 0]) 
   
   ng = n-nb
   fnrmtol = 1000*nrmA/nrmB
   
   nu = 0; na = 0
   nc = n  
   F = zeros(T,m,n); 
   ia = n-nb+1
   ihf = 0
   while nb > 0
      noskip = true
      if nb == 1 || complx || A1[nc,nc-1] == 0
         k = 1
      else
         k = 2
      end 
      kk = nc-k+1:nc
      a2 = view(A1,kk,kk)
      evb = ordeigvals(a2)
      b2 = view(Z,ib,kk)'*view(B1,ib,:)
      if norm(b2,Inf) <= tolb
         # deflate uncontrollable block
         nb = nb-k; nc = nc-k; nu = nu+k; noskip = false
      elseif k == 1 && nb > 1 && ismissing(evalsr) && !ismissing(evalsc)
         # form a 2x2 block if there are no real eigenvalues to assign
         k = 2
         kk = nc-k+1:nc
         a2 = view(A1,kk,kk)
         if nb > 2 && A1[nc-1,nc-2] != ZERO
            # interchange last two blocks
            LAPACK.trexc!('V', nc, nc-2, A1, Z) 
            evb = ordeigvals(a2)
         else
            evb = disc ? maximum(abs.(ordeigvals(a2))) : maximum(real(ordeigvals(a2)))
         end
         b2 = view(Z,ib,kk)'*view(B1,ib,:)
         if norm(b2,Inf) <= tolb
            # deflate uncontrollable block
            nb = nb-k; nc = nc-k; nu = nu+k; noskip = false
         end
      end
      if noskip 
         if k == 1 
            # assign a single eigenvalue 
            γ, evalsr = eigselect1(evalsr, sdeg, complx ? evb[1] : real(evb[1]), disc; cflag = complx);
            if γ === nothing
                # no real eigenvalue available, adjoin a new 1x1 block if possible
                if nb == 1
                    # incompatible eigenvalues with the eigenvalue structure
                    # assign the last real pole to the default value of sdeg
                    γ = disc ? real(T)(0.95) : real(T)(-0.05)
                    f2 = -b2\(a2-I*γ)
                 else
                    # adjoin a real block or interchange the last two blocks
                    k = 2; kk = nc-1:nc; 
                    a2 = view(A1,kk,kk)
                    if nb > 2 && A1[nc-1,nc-2] != ZERO
                       # interchange last two blocks
                       LAPACK.trexc!('V', nc, nc-2, A1, Z) 
                       # update evb 
                       evb = ordeigvals(a2)
                    else
                       # update evb 
                       evb = disc ? maximum(abs.(ordeigvals(a2))) : maximum(real(ordeigvals(a2)))
                    end
                    b2 = view(Z,ib,kk)'*view(B1,ib,:)
                end
            else
               f2 = -b2\(a2-I*γ);
            end
         end
         if k == 2
            # assign a pair of eigenvalues 
            γ, evalsr, evalsc = eigselect2(evalsr,evalsc,sdeg,evb[end],disc)
            f2, u = saloc2(a2,b2,γ,tola,tolb)   
            if f2 === nothing  # the case b2 = 0 can not occur
               irow = 1:nc; jcol = nc-1:n;
               A1[kk,jcol] = u'*view(A1,kk,jcol); A1[irow,kk] = view(A1,irow,kk)*u;
               A1[nc,nc-1] = ZERO
               Z[:,kk] = view(Z,:,kk)*u
               nb -= 1; nc -= 1; nu += 1; 
               # recover the failed selection 
               imag(γ[1]) == 0 ? (ismissing(evalsr) ? evalsr = γ : evalsr = [γ; evalsr]) : (ismissing(evalsc) ? evalsc = γ : evalsc = [γ; evalsc])
            end
         end
         if f2 !== nothing
            # check for numerical stability
            norm(f2,Inf) > fnrmtol && (ihf += 1)
            X = view(B1,ib,:)*f2
            A1[1:nc,kk] += view(Z,ib,1:nc)'*X
            F += f2*view(Z,:,kk)'
            if k == 2
               # standardization step is necessary to use trexc 
               i1 = 1:nc-2; lcol = nc+1:n;
               # alternative computation
               # k1 = kk[1]; k2 = kk[2]
               # RT1R, RT1I, RT2R, RT2I, CS, SN = lanv2(A1[k1,k1], A1[k1,k2], A1[k2,k1], A1[k2,k2]) 
               _, Z2, _ = LAPACK.gees!('V', a2)
               A1[i1,kk] = view(A1,i1,kk)*Z2
               A1[kk,lcol] = Z2'*view(A1,kk,lcol) 
               Z[:,kk] = view(Z,:,kk)*Z2; 
               tworeals = (A1[nc,nc-1] == ZERO)
            else
               tworeals = false
            end
            # reorder eigenvalues 
            if nb > k
               try
                  LAPACK.trexc!('V', nc-k+1, ia, A1, Z) 
                  tworeals && LAPACK.trexc!('V', nc, ia+1, A1, Z)
               catch
               end
            end
            nb -= k
            ia += k 
            na += k
         end
      end
   end
   ihf > 0 && @warn("Possible loss of numerical reliability due to high feedback gain")
   blkdims = [ng, na, nu]
   return F, Schur(A1, Z, ordeigvals(A1)), blkdims

   # end saloc
end
"""
    saloc(A, E, B; evals, sdeg, disc = false, atol1 = 0, atol2 = 0, atol3 = 0, rtol, sepinf = true, fast = true) -> (F, Scl, blkdims)

Compute for the pair `(A-λE,B)`, with `A-λE` a regular pencil, a matrix `F` such that all finite eigenvalues of the 
pencil `A+B*F-λE` lie in the stability domain `Cs` specified by the stability degree parameter `sdeg` and stability type parameter `disc`. 
If `disc = false`, `Cs` is the set of complex numbers with real parts at most `sdeg`, while if `disc = true`, 
`Cs` is the set of complex numbers with moduli at most `sdeg` (i.e., the interior of a disc of radius `sdeg` centered in the origin). 
`evals` is a real or complex vector, which contains a set of finite desired eigenvalues for the pencil `A+B*F-λE`. 
For real data `A`, `E`, and `B`, `evals` must be a self-conjugated complex set to ensure that the resulting `F` is also a real matrix. 

For a pair `(A-λE,B)` with `A` of order `n`, the number of assignable finite eigenvalues is `nfc := n-ninf-nfu`,  
where `ninf` is the number of infinite eigenvalues of `A-λE` and `nfu` is the number of fixed finite eigenvalues of `A-λE`. 
The assignable finite eigenvalues are called the _controllable finite eigenvalues_, 
while the fixed finite eigenvalues are called the _uncontrollable finite eigenvalues_ (these are the finite zeros of the pencil `[A-λE B]`). 
The spectrum allocation is achieved by successively replacing the controllable finite eigenvalues of `A-λE` lying outside of the 
stability domain `Cs` with eigenvalues provided in `evals`. All finite eigenvalues of `A-λE` lying in `Cs` are kept unalterred.  
If the number of specified eigenvalues in `evals` is less than the number of controllable finite eigenvalues of `A-λE` outside of `Cs`
(e.g., if `evals = missing`), then some of the controllable finite eigenvalues of `A-λE` are assigned to the nearest values 
on the boundary of `Cs`. If `sdeg = missing` and `evals = missing`, the default value used for `sdeg` is -0.05 
if  `disc = false` and 0.95 if `disc = true`. 

The keyword arguments `atol1`, `atol2`, `atol3`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of `A`, the absolute tolerance for the nonzero elements of `E`, the absolute tolerance for the nonzero elements of `B`,  
and the relative tolerance for the nonzero elements of `A`, `E` and `B`.  
The default relative tolerance is `n*ϵ`, where `ϵ` is the machine epsilon of the element type of `A`. 

The keyword argument `sepinf` specifies the option for a preliminary separation 
of the infinite eigenvalues of the pencil `A-λE` as follows: if `sepinf = false`, no separation of infinite eigenvalues is performed, 
while for `sepinf = true` (the default option), a preliminary separation of the infinite eigenvalues from the finite ones is performed.
If `E` is nonsingular, then `sepinf = false` is recommended to be used. If `E` is numerically singular, then the option `sepinf = false` is used. 
The separation of finite and infinite eigenvalues is performed using rank decisions based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.

The resulting pencil `Acl-λEcl := Q'*(A+B*F-λE)*Z`,  where `Q` and `Z` are the  
orthogonal/unitary matrices used to obtain the pair `(Acl,Ecl)` in a generalized Schur form (GSF), has the form

                ( Ai-λEi    *        *       *      )     
     Acl-λEcl = (   0    Afg-λEfg    *       *      )  
                (   0       0     Afa-λEfa   *      )    
                (   0       0        0     Afu-λEfu )    

where: `Ai-λEi` with `Ai` upper triangular and invertible and `Ei` upper triangular and nilpotent, contains the `ninf` infinite eigenvalues of `A-λE`; 
`Afg-λEfg` contains `nfg` finite eigenvalues of `A-λE` in `Cs`; `Afa-λEfa` contains `nfa` assigned finite generalized eigenvalues in `Cs`; 
and `Afu-λEfu` contains `nfu` uncontrollable finite eigenvalues of `A-λE` lying outside `Cs`. 
The matrices `Acl`, `Ecl`, `Q`, `Z` and the vectors `α` and `β` such that `α./β` are the generalized eigenvalues of 
the pair `(Acl,Ecl)` are returned in the `GeneralizeSchur` object `Scl`. 
The values of `ninf`, `nfg`, `nfa` and `nfu` are returned in the 4-dimensional vector `blkdims = [ninf, nfg, nfa, nfu]`.

Method:  For a pair `(A-λE,B)` with `E = I`, the Schur method of [1] is used, while for a general pair `(A-λE,B)` the generalized Schur method of [2] 
is used to solve the R-stabilzation problem.

References:

[1] A. Varga. 
    A Schur method for pole assignment.
    IEEE Trans. on Automatic Control, vol. 26, pp. 517-519, 1981.

[2] A. Varga. 
    On stabilization methods of descriptor systems.
    Systems & Control Letters, vol. 24, pp.133-138, 1995.

"""
function saloc(A::AbstractMatrix, E::Union{AbstractMatrix,UniformScaling{Bool}}, B::AbstractMatrix; 
                  disc::Bool = false, evals::Union{AbstractVector,Missing} = missing, sdeg::Union{Real,Missing} = missing, 
                  atol1::Real = zero(real(eltype(A))), atol2::Real = zero(real(eltype(A))), atol3::Real = zero(real(eltype(B))), 
                  rtol::Real = ((size(A,1)+1)*eps(real(float(one(eltype(A))))))*iszero(max(atol1,atol2,atol3)), 
                  fast::Bool = true, sepinf::Bool = true)

   n = LinearAlgebra.checksquare(A)
   if E == I
      F, S, blkdims = saloc(A, B, evals = evals, sdeg = sdeg, atol1 = atol1, atol2 = atol3, rtol = rtol)
      TS = eltype(F)
      return F, GeneralizedSchur(S.T, Matrix{TS}(I,n,n), S.values, ones(TS,n), S.Z, S.Z), [0; blkdims]
   end
   
   LinearAlgebra.checksquare(E) == n || throw(DimensionMismatch("E must be a $n x $n matrix"))

   n1, m = size(B)
   n == n1 || throw(DimensionMismatch("A and B must have the same number of rows"))
   
   T = promote_type( eltype(A), eltype(B), eltype(E) )
   T <: BlasFloat || (T = promote_type(Float64,T))
   
   A1 = copy_oftype(A,T)   
   E1 = copy_oftype(E,T)
   B1 = copy_oftype(B,T)
   if ismissing(evals)
      evals1 = evals
   else 
      T1 = promote_type(T,eltype(evals)) 
      evals1 = copy_oftype(evals,T1)
   end
   
   # quick exit for n = 0 or B = 0

   # the following is not working for 0 dimensions
   # n == 0 || (return zeros(T,m,n), schur(zeros(T,n,n), ones(T,n,n)), zeros(T,n,m), zeros(Int,4)) 
   n == 0 && (return zeros(T,m,n), GeneralizedSchur(zeros(T,n,n),zeros(T,n,n),zeros(T,n),zeros(T,n),zeros(T,n,n),zeros(T,n,n)), zeros(Int,4)) 

   ZERO = zero(T)

   # check for zero rows in the leading positions
   ilob = n+1
   for i = 1:n
       !iszero(view(B1,i,:)) && (ilob = i; break)
   end

   # return if B = 0
   ilob > n && (return zeros(T,m,n), schur(A1,E1), [0, 0, 0, n] )

   # check for zero rows in the trailing positions
   ihib = ilob
   for i = n:-1:ilob+1
       !iszero(view(B1,i,:)) && (ihib = i; break)
   end
   
   # operate only on the nonzero rows of B
   ib = ilob:ihib
   nrmB = opnorm(view(B1,ib,:),1)
    
   complx = (T <: Complex)
   
   # sort desired eigenvalues
   if ismissing(evals1) 
      evalsr = missing
      evalsc = missing
   else
      if complx
         evalsr = copy(evals1)
         evalsc = missing
      else
         evalsr = evals1[imag.(evals1) .== 0]
         isempty(evalsr) && (evalsr = missing)
         tempc = evals1[imag.(evals1) .> 0]
         if isempty(tempc)
            evalsc = missing
         else
            tempc1 = conj(evals1[imag.(evals1) .< 0])
            isequal(tempc[sortperm(real(tempc))],tempc1[sortperm(real(tempc1))]) ||
                    error("evals must be a self-conjugated complex vector")
            evalsc = [transpose(tempc[:]); transpose(conj.(tempc[:]))][:]
         end
      end
      # check that all eigenvalues are inside of the stability region
      !ismissing(sdeg) && ((disc && any(abs.(evals1) .> sdeg) )  || (!disc && any(real.(evals1) .> sdeg)))  &&
            error("The elements of evals must lie in the stability region")
   end    
   
   # set default values of sdeg if evals = missing
   if ismissing(sdeg)
      if ismissing(evals1) 
         sdeg = disc ? real(T)(0.95) : real(T)(-0.05)
         smarg = sdeg;   
      else
         smarg = disc ? real(T)(0) : real(T)(-Inf) 
      end
   else
      sdeg = real(T)(sdeg)
      disc && sdeg < 0 && error("sdeg must be non-negative if disc = true")
      smarg = sdeg;  
   end
 
    
   nrmA = opnorm(A1,1)
   tola = max(atol1, rtol*nrmA)
   tolb = max(atol3, rtol*nrmB)

   Q = Matrix{T}(I,n,n)
   Z = Matrix{T}(I,n,n) 
   
   if !sepinf 
      # reduce (A,E) to generalized Schur form
      istriu(E1) ||  _qrE!(A1, E1, Q, missing; withQ = true) 
      rcond = LAPACK.trcon!('1','U','N',E1)
      if rcond < eps(real(T))
         @warn("E is numerically singular: computations resumed with sepinf = true")
         sepinf = true
         A1 = copy_oftype(A,T)   
         E1 = copy_oftype(E,T) 
         Q = Matrix{T}(I,n,n)
      else
         gghrd!('V','V',1, n, A1, E1, Q, Z)
         _, _, α, β, _, _ = hgeqz!('V','V',1, n, A1, E1, Q, Z)
         ninf = 0
         if disc
            select = Int.((abs.(α) .<= smarg*abs.(β))) 
         else
            select = Int.((real.(α ./ β) .< smarg)) 
         end
      end
   end

   if sepinf 
      _, blkdims = fisplit!(A1, E1, Q, Z, missing, missing; fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol, withQ = true, withZ = true) 
      ninf = blkdims[1]
      ilo = ninf+1; 
      gghrd!('V','V',ilo, n, A1, E1, Q, Z)
      _, _, α, β, _, _ = hgeqz!('V','V',ilo, n, A1, E1, Q, Z)
      i2 = ilo:n
      if disc
         select2 = Int.(abs.(α[i2]) .<= smarg*abs.(β[i2]))
      else
         select2 = Int.(real.(α[i2] ./ β[i2]) .< smarg)
      end
      select = [ones(Int,ninf);select2]
   end

   # separate stable and unstable eigenvalues 
  
   # compute orthogonal Q and Z such that
   #                            
   #                   [ Ai-λEi   *      *    ]
   #      Q*(A-λE)*Z = [   0    Ag-λEg   *    ]
   #                   [   0      0    Ab-λEb ]
   #
   # where Ag-λEg has eigenvalues within the stability degree region
   # and Ab-λEb has eigenvalues outside the stability degree region.

   _, _, α, β, _, _ = LAPACK.tgsen!(select, A1, E1, Q, Z) 
   nb = length(select[select .== 0]) 

   nfg = n-nb-ninf
   fnrmtol = 1000*max(nrmA,1)/nrmB
   
   nfu = 0; nfa = 0
   nc = n  
   F = zeros(T,m,n); 
   ia = n-nb+1
   ihf = 0
   while nb > 0
      noskip = true
      if nb == 1 || complx || A1[nc,nc-1] == 0
         k = 1
      else
         k = 2
      end
      kk = nc-k+1:nc
      a2 = view(A1,kk,kk)
      e2 = view(E1,kk,kk)
      evb, = ordeigvals(a2,e2)
      b2 = view(Q,ib,kk)'*view(B1,ib,:)
      if norm(b2,Inf) <= tolb
         # deflate uncontrollable stable block
         nb = nb-k; nc = nc-k; nfu = nfu+k; noskip = false
      elseif k == 1 && nb > 1 && ismissing(evalsr) && !ismissing(evalsc)
         # form a 2x2 block if there are no real no real eigenvalues to assign
         k = 2
         kk = nc-k+1:nc
         a2 = view(A1,kk,kk)
         e2 = view(E1,kk,kk)
         if nb > 2 && A1[nc-1,nc-2] != ZERO
            # interchange last two blocks
            tgexc!(true, true, nc, nc-2, A1, E1, Q, Z) 
            evb, = ordeigvals(a2,e2)
         else
            evb = disc ? maximum(abs.(ordeigvals(a2,e2)[1])) : maximum(real(ordeigvals(a2,e2)[1]))
         end
         b2 = view(Q,ib,kk)'*view(B1,ib,:)
         if norm(b2,Inf) <= tolb
            # deflate uncontrollable block
            nb = nb-k; nc = nc-k; nfu = nfu+k; noskip = false
         end
      end
      if noskip 
         if k == 1 
            # assign a single eigenvalue 
            γ, evalsr = eigselect1(evalsr, sdeg, complx ? evb[1] : real(evb[1]), disc; cflag = complx);
            if γ === nothing
                # no real pole available, adjoin a new 1x1 block if possible
                if nb == 1
                    # incompatible eigenvalues with the eigenvalue structure
                    # assign the last real pole to the default value of sdeg
                    γ = disc ? real(T)(0.95) : real(T)(-0.05)
                    f2 = -b2\(a2-e2*γ)
                else
                    # adjoin a real block or interchange the last two blocks
                    k = 2; kk = nc-k+1:nc; 
                    a2 = view(A1,kk,kk)
                    e2 = view(E1,kk,kk)
                    if nb > 2 && A1[nc-1,nc-2] != ZERO
                       # interchange last two blocks
                       tgexc!(true, true, nc, nc-2, A1, E1, Q, Z) 
                    end
                    # update evb 
                    evb = maximum(real(ordeigvals(a2,e2)[1]))
                    b2 = view(Q,ib,kk)'*view(B1,ib,:)
                end
            else
               f2 = -b2\(a2-e2*γ);
            end
         end
         if k == 2
            # assign a pair of eigenvalues 
            γ, evalsr, evalsc = eigselect2(evalsr,evalsc,sdeg,evb[end],disc)
            f2, u, v = saloc2(a2,e2,b2,γ,tola,tolb)
            if f2 === nothing  # the case b2 = 0 can not occur
               irow = 1:nc; jcol = nc-1:n;
               A1[kk,jcol] = u'*view(A1,kk,jcol); A1[irow,kk] = view(A1,irow,kk)*v;
               A1[nc,nc-1] = ZERO
               E1[kk,jcol] = u'*view(E1,kk,jcol); E1[irow,kk] = view(E1,irow,kk)*v; 
               E1[nc,nc-1] = ZERO
               Q[:,kk] = view(Q,:,kk)*u;
               Z[:,kk] = view(Z,:,kk)*v;
               nb -= 1; nc -= 1; nfu += 1
               # recover the failed selection 
               imag(γ[1]) == 0 ? (ismissing(evalsr) ? evalsr = γ : evalsr = [γ; evalsr]) : (ismissing(evalsc) ? evalsc = γ : evalsc = [γ; evalsc])
            end
         end
         if f2 !== nothing
            norm(f2,Inf) > fnrmtol && (ihf += 1)
            # update matrices Acl and F
            X = view(B1,ib,:)*f2
            A1[1:nc,kk] += view(Q,ib,1:nc)'*X
            F += f2*view(Z,:,kk)'
            if k == 2
               # standardization step is necessary to use tgsen
               i1 = 1:nc-2; lcol = nc+1:n;
               _, _, _, _, Q2, Z2 = LAPACK.gges!('V','V',a2,e2)
               A1[i1,kk] = view(A1,i1,kk)*Z2
               E1[i1,kk] = view(E1,i1,kk)*Z2
               A1[kk,lcol] = Q2'*view(A1,kk,lcol) 
               E1[kk,lcol] = Q2'*view(E1,kk,lcol) 
               Q[:,kk] = view(Q,:,kk)*Q2; 
               Z[:,kk] = view(Z,:,kk)*Z2; 
               tworeals = (A1[nc,nc-1] == 0)
            else
               tworeals = false
            end
            # reorder eigenvalues 
            if nb > k
               tgexc!(true, true, nc-k+1, ia, A1, E1, Q, Z) 
               tworeals && tgexc!(true, true, nc, ia+1, A1, E1, Q, Z) 
            end
            nb -= k
            ia += k 
            nfa += k
         end
      end
   end
   ihf > 0 && @warn("Possible loss of numerical reliability due to high feedback gain")
   blkdims = [ninf, nfg, nfa, nfu]
   _, α, β = ordeigvals(A1,E1)
   return F, GeneralizedSchur(A1, E1, α, β, Q, Z), blkdims
   
   # end saloc
end

"""
    salocinf(A, E, B; atol1 = 0, atol2 = 0, atol3 = 0, rtol, fast = true) -> (F, G, Scl, blkdims)

Compute for the controllable pair `(A-λE,B)`, with `A-λE` a regular pencil, two matrices `F` and `G` such that all eigenvalues of the 
pencil `A+B*F-λ(E+B*G)` are infinite. For a pair `(A-λE,B)` with fixed (uncontrollable) finite eigenvalues, only the assignable (controllable)
 finite eigenvalues are moved to infinity. 

For a pair `(A-λE,B)` with `A` of order `n`, the number of assignable infinite eigenvalues is `nia := n-ninf-nfu`,  
where `ninf` is the number of infinite eigenvalues of `A-λE` and `nfu` is the number of fixed finite eigenvalues of `A-λE`. 
The assignable finite eigenvalues are called the _controllable finite eigenvalues_, 
while the fixed finite eigenvalues are called the _uncontrollable finite eigenvalues_ (these are the finite zeros of the pencil `[A-λE B]`). 
The spectrum allocation is achieved by successively replacing the controllable finite eigenvalues of `A-λE` with infinite eigenvalues.
 All infinite eigenvalues of `A-λE` are kept unalterred.

The keyword arguments `atol1`, `atol2`, `atol3`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of `A`, the absolute tolerance for the nonzero elements of `E`, the absolute tolerance for the nonzero elements of `B`,  
and the relative tolerance for the nonzero elements of `A`, `E` and `B`.  
The default relative tolerance is `n*ϵ`, where `ϵ` is the machine epsilon of the element type of `A`. 

The preliminary separation of finite and infinite eigenvalues is performed using rank decisions based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.

The resulting pencil `Acl-λEcl := Q'*(A+B*F-λ(E+λB*G))*Z`,  where `Q` and `Z` are the  
orthogonal/unitary matrices used to obtain the pair `(Acl,Ecl)` in a generalized Schur form (GSF), has the form

                ( Aig-λEig    *        *     )     
     Acl-λEcl = (   0      Aia-λEia    *     )  
                (   0         0     Afu-λEfu )    

where: `Aig-λEig` with `Aig` upper triangular and invertible and `Eig` upper triangular and nilpotent, contains the `ninf` infinite eigenvalues of `A-λE`; 
`Aia-λEia` with `Aia` upper triangular and invertible and `Eia` upper triangular and nilpotent, contains `nia` assigned infinite generalized eigenvalues; 
and `Afu-λEfu`, with the pair `(Afu,Efu)` in a generalized Schur form,  contains `nfu` fixed (uncontrollable) finite eigenvalues of `A-λE`. 
The matrices `Acl`, `Ecl`, `Q`, `Z` and the vectors `α` and `β` such that `α./β` are the generalized eigenvalues of 
the pair `(Acl,Ecl)` are returned in the `GeneralizeSchur` object `Scl`. 
The values of `ninf`, `nia` and `nfu` are returned in the 3-dimensional vector `blkdims = [ninf, nia, nfu]`.

Method:  For a general pair `(A-λE,B)` the modified generalized Schur method of [1] is used to determine `F` and `G` such that the pair
`(A+B*F,E+B*G)` have only infinite eigenvalues.

References:

[1] A. Varga. 
    On stabilization methods of descriptor systems.
    Systems & Control Letters, vol. 24, pp.133-138, 1995.

"""
function salocinf(A::AbstractMatrix, E::AbstractMatrix, B::AbstractMatrix; 
                  atol1::Real = zero(real(eltype(A))), atol2::Real = zero(real(eltype(A))), atol3::Real = zero(real(eltype(B))), 
                  rtol::Real = ((size(A,1)+1)*eps(real(float(one(eltype(A))))))*iszero(max(atol1,atol2,atol3)), 
                  fast::Bool = true)

   n = LinearAlgebra.checksquare(A)
   
   LinearAlgebra.checksquare(E) == n || throw(DimensionMismatch("E must be a $n x $n matrix"))

   n1, m = size(B)
   n == n1 || throw(DimensionMismatch("A and B must have the same number of rows"))
   
   T = promote_type( eltype(A), eltype(B), eltype(E) )
   T <: BlasFloat || (T = promote_type(Float64,T))

   A1 = copy_oftype(A,T)   
   E1 = copy_oftype(E,T)
   B1 = copy_oftype(B,T)
   
   # quick exit for n = 0 or B = 0

   # the following is not working for 0 dimensions
   # n == 0 || (return zeros(T,m,n), schur(zeros(T,n,n), ones(T,n,n)), zeros(T,n,m), zeros(Int,4)) 
   n == 0 && (return zeros(T,m,n), zeros(T,m,n), GeneralizedSchur(zeros(T,n,n),zeros(T,n,n),zeros(T,n),zeros(T,n),zeros(T,n,n),zeros(T,n,n)), zeros(Int,3)) 

   ZERO = zero(T)

   # check for zero rows in the leading positions
   ilob = n+1
   for i = 1:n
       !iszero(view(B1,i,:)) && (ilob = i; break)
   end

   # return if B = 0
   ilob > n && (return zeros(T,m,n), zeros(T,m,n), schur(A1,E1), [0, 0, n] )

   # check for zero rows in the trailing positions
   ihib = ilob
   for i = n:-1:ilob+1
       !iszero(view(B1,i,:)) && (ihib = i; break)
   end
   
   # operate only on the nonzero rows of B
   ib = ilob:ihib
   nrmB = opnorm(view(B1,ib,:),1)
    
   complx = (T <: Complex)
   
       
   nrmA = opnorm(A1,1)
   nrmE = opnorm(E1,1)
   tola = max(atol1, rtol*nrmA)
   tole = max(atol2, rtol*nrmE)
   tolb = max(atol3, rtol*nrmB)

   Q = Matrix{T}(I,n,n)
   Z = Matrix{T}(I,n,n) 
   
   _, blkdims = fisplit!(A1, E1, Q, Z, missing, missing; fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol, withQ = true, withZ = true) 
   ninf = blkdims[1]
   ilo = ninf+1; 
   gghrd!('V','V',ilo, n, A1, E1, Q, Z)
   _, _, α, β, _, _ = hgeqz!('V','V',ilo, n, A1, E1, Q, Z)
   i2 = ilo:n

   nb = n-ninf
   fnrmtol = 1000*max(nrmA,nrmE,1)/nrmB
   scale = max(nrmA,1)/10

   nfu = 0; nia = 0
   nc = n  
   F = zeros(T,m,n); 
   G = zeros(T,m,n); 
   ia = ninf+1
   ihf = 0

   while nb > 0
      noskip = true
      if nb == 1 || complx || A1[nc,nc-1] == 0
         k = 1
      else
         k = 2
      end
      kk = nc-k+1:nc
      a2 = view(A1,kk,kk)
      e2 = view(E1,kk,kk)
      b2 = view(Q,ib,kk)'*view(B1,ib,:)
      if norm(b2,Inf) <= tolb
         # deflate uncontrollable block
         nb = nb-k; nc = nc-k; nfu = nfu+k; noskip = false
      else 
         # perturb a2 such that a2+b2*f2 is sufficiently nonzero 
         if maximum(abs.(a2)) < maximum(abs.(e2)) /10000 || rank(a2,atol=tola) < k
            f2 = lmul!(scale/nrmB,rand(T,m,k).+1)
            # update A and F
            X = view(B1,ib,:)*f2
            A1[1:nc,kk] += view(Q,ib,1:nc)'*X
            F += f2*view(Z,:,kk)'
         end
         if k == 1
            # assign a single infinite eigenvalue 
            g2 = -b2\e2
            X = view(B1,ib,:)*g2
            E1[1:nc,kk] += view(Q,ib,1:nc)'*X
            G += g2*view(Z,:,kk)'
            # reorder eigenvalues 
            if nb > k
               tgexc!(true, true, nc-k+1, ia, A1, E1, Q, Z) 
            end
            E1[ia,ia] = ZERO
         else
            # assign two infinite eigenvalues 
            g2,  = saloc2(e2,a2,b2,zeros(T,2),tole,tolb)  
            # update E and G
            X = view(B1,ib,:)*g2
            E1[1:nc,kk] += view(Q,ib,1:nc)'*X
            G += g2*view(Z,:,kk)'
            # perform standardization step to form two 1x1 blocks
            Q2, e2[1,1] = givens(e2[1,1],e2[2,1],1,2)
            e2[2,1] = ZERO; 
            lmul!(Q2,view(E1,kk,nc:n))
            e2[2,2] = ZERO
            lmul!(Q2,view(A1,kk,nc-1:n))            
            rmul!(view(Q,:,kk),Q2') 
            Z2, r = givens(conj(a2[2,2]),conj(a2[2,1]),2,1)
            a2[2,2] = conj(r)
            a2[2,1] = ZERO
            rmul!(view(E1,1:nc-1,kk),Z2')
            rmul!(view(A1,1:nc-1,kk),Z2')
            rmul!(view(Z,:,kk),Z2') 
            # reorder eigenvalues 
            if nb > k
               tgexc!(true, true, nc-k+1, ia, A1, E1, Q, Z) 
               tgexc!(true, true, nc, ia+1, A1, E1, Q, Z) 
            end
            E1[ia,ia] = ZERO
            E1[ia+1,ia+1] = ZERO
         end
         nb -= k
         ia += k 
         nia += k
      end
   end
   ihf > 0 && @warn("Possible loss of numerical reliability due to high feedback gain")
   blkdims = [ninf, nia, nfu]
   _, α, β = ordeigvals(A1,E1)
   return F, G, GeneralizedSchur(A1, E1, α, β, Q, Z), blkdims
   
   # end salocinf
end
"""
     ev = ordeigvals(A) 

Compute the vector `ev` of eigenvalues of a Schur matrix `A` in their order of appearance down the diagonal of `A`.
`ordeigvals` is an order-preserving version of `eigvals` for triangular/quasitriangular matrices.
"""
function ordeigvals(A::AbstractMatrix{T}) where T
   isqtriu(A) || error("A must be in Schur form")
   n = size(A,1)
   ZERO = zero(T)
   if T <: Complex
      α = Vector{T}(undef, n)
      for i = 1:n
         α[i] = A[i,i]
      end
      return α
   else
      αr = Vector{T}(undef, n)
      αi = zeros(T,n)
      for i = 1:n
         αr[i] = A[i,i]
      end
      for i = 1:n-1
         if A[i+1,i] != ZERO
            if A[i,i] == A[i+1,i+1] 
               # exploit LAPACK 2x2 block structure
               ei = sqrt(abs(A[i,i+1]))*sqrt(abs(A[i+1,i]))
               αi[i] = ei
               αi[i+1] = -ei
            else
               # an arbitrary 2x2 block
               αr[i], αi[i], αr[i+1], αi[i+1] = lanv2(A[i,i], A[i,i+1], A[i+1,i], A[i+1,i+1])
            end
            i +=1
         end
      end
      return iszero(αi) ? αr : Complex.(αr,αi)
   end
end
"""
    ordeigvals(A, B) -> (ev, α, β)

Compute the vector `ev` of generalized eigenvalues of the pair `(A,B)`, with `A` a Schur matrix and `B` an upper triangular matrix, 
in their order of appearance down the diagonals of `A` and `B`.
`ordeigvals` is an order-preserving version of `eigvals` for (triangular/quasi-upper-triangular, triangular) pairs of matrices.
`α` is a complex vector and `β` is a real vector such that the generalized eigenvalues `ev` can be alternatively obtained with `α./β`.
"""
function ordeigvals(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where T
   na, nb = LinearAlgebra.checksquare(A,B)
   na == nb ||  throw(DimensionMismatch("A and B must have the same size"))
   (isqtriu(A) && istriu(B)) || error("The pair (A,B) must be in a generalized Schur form")
   n = size(A,1)
   complx = T <: Complex
   if complx
      α = Vector{T}(undef, n)
      β = Vector{T}(undef, n)
      for i = 1:n
         α[i] = A[i,i]
         β[i] = B[i,i]
      end
      ev = α./β
      ev[iszero.(β)] .= Inf
      ev[iszero.(α) .& iszero.(β)] .= NaN + NaN*im
      return ev, α, β
   else
      αr = Vector{T}(undef, n)
      αi = Vector{T}(undef, n)
      β = Vector{T}(undef, n)
      ZERO = zero(T)
      small = safemin(T)
      i = 1
      while i <= n
         pair = (i < n && A[i+1,i] != ZERO) 
         if pair
            ii = i:i+1
            β[i], β[i+1], αr[i], αr[i+1], wi  = lag2(A[ii,ii],B[ii,ii],small)
            iszero(wi) ? (αi[i] = wi; αi[i+1] = wi) : (αi[i] = wi; αi[i+1] = -wi; β[i+1] = β[i])
            i += 2
         else
            αr[i] = A[i,i]
            αi[i] = ZERO
            β[i] = B[i,i]
            i += 1
         end
      end 
      α = iszero(αi) ? αr : Complex.(αr,αi)
      ev = α./β
      complx || (ev[imag.(ev) .> 0] = conj(ev[imag.(ev) .< 0]))
      ev[iszero.(β)] .= Inf
      ev[iszero.(α) .& iszero.(β)] .= NaN
      return ev, α, β
   end
end
"""
    saloc2(A, B, evc, tola, tolb) -> (F, U)

Compute for the real pair `(A,B)` with `A` of order two, a matrix `F` such that the eigenvalues 
of the matrix `A+B*F` are equal to the complex conjugate pair contained in `evc`. 
The absolute thresholds `tola` and `tolb` for the nonzero elements in `A` and `B`, respectively, 
are used for controllability checks.

If the pair `(A,B)` is uncontrollable, then `F = nothing` and `U` contains an orthogonal transformation 
matrix such that the transformed pair `(U'*A*U, U'*B)` is in the form

                     [ A1  X    B1 ]
    [ U'*A*U U'*B] = [             ] ,
                     [  0  A2   0  ]
 
where the pair `(A1,B1)` is controllable. If `norm(B) < tolb`, then `U` is the `2x2` identity matrix.
"""
function saloc2(A::AbstractMatrix{T},B::AbstractMatrix{T},evc::AbstractVector{T1},tola,tolb) where {T, T1}
           
   # check controllability and determine rank of B
   m = size(B,2);
   SF = svd(B; full=true); s1 = SF.S[1]
   U = SF.U
   v1 = SF.V
   if m == 1
      s2 = zero(T);
   else
      s2 = SF.S[2]
   end
   rankB = 0
   s1 > tolb && (rankB += 1)
   s2 > tolb && (rankB += 1)
   # return if norm(B) < tolb
   rankB == 0 && (return nothing, Matrix{T}(I,2,2))

   at = U'*A*U;  
   if s2 <= tolb && abs(at[2,1]) <= tola
         # return if rank(B) == 1 and (A-lambda*I,B) uncontrollable
      return nothing, U
   end
   if rankB == 2
      # try with a direct solution, without inversion
      γ = [real(evc[1]) imag(evc[1]); imag(evc[2]) real(evc[2])]; 
      ftry = -B\(A-γ);
   end
   sc = at[1,1]+at[2,2];
   sp = real(evc[1]+evc[2]); pp = real(evc[1]*evc[2]);
   k11 = (sc-sp)/s1; 
   k12 = at[2,2]/at[2,1]*k11+(pp-at[1,1]*at[2,2]+at[1,2]*at[2,1])/at[2,1]/s1;
   F = v1*[-k11 -k12; zeros(m-1,2)]*U'; 
   if rankB == 2 && norm(F) > norm(ftry)
     # choose the lower norm feedback 
      F = ftry;
   end
   return F, U
   # end saloc2
end
"""
    saloc2(A, B, E, evc, tola, tolb) -> (F, U, V)

Compute for the real pair `(A-λE,B)` with `A-λE` of order two, a matrix `F` such that the 
generalized eigenvalues of the matrix pair `(A+B*F,E)` are equal to the complex conjugate pair contained in `evc`. 
The absolute thresholds `tola` and `tolb` for the nonzero elements in `A` and `B`, respectively, 
are used for controllability checks.

If the pair `(A-λE,B)` is uncontrollable, then `F = nothing` and `U` and `V` contains orthogonal transformation 
matrices such that the transformed pair `(U'*A*V-λU'*E*V, U'*B)` is in the form

                             [ A1-λE1   X    B1 ]
    [ U'*A*V-λU'*E*V U'*B] = [                  ] ,
                             [  0    A2-λE2  0  ]
 
where the pair `(A1-λE1,B1)` is controllable. If `norm(B) < tolb`, then `U` and `V` are `2x2` identity matrices.
"""
   

function saloc2(A::AbstractMatrix{T},E::AbstractMatrix{T},B::AbstractMatrix{T},evc::AbstractVector{T1},tola,tolb) where {T, T1}
    
    # check controllability and determine rank of B
    m = size(B,2);
    SF = svd(B; full=true); s1 = SF.S[1]
    U = SF.U
    v1 = SF.V
    if m == 1
      s2 = zero(T);
    else
      s2 = SF.S[2]
    end
    rankB = 0
    s1 > tolb && (rankB += 1)
    s2 > tolb && (rankB += 1)
    # return if norm(B) < tolb
    rankB == 0 && (return nothing, Matrix{T}(I,2,2), Matrix{T}(I,2,2))

    at = U'*A; et = U'*E;
    # determine V such that et = U'*E*V is upper triangular
    # FQ = qr([et[2,2]; et[2,1]]); V = FQ.Q*Matrix{T}(I,2,2); V = V[2:-1:1,2:-1:1]';  
    G,  = LinearAlgebra.givens(et[2,2], et[2,1],1,2)
    # at = at*G; 
    rmul!(at,G) 
    if s2 <= tolb && abs(at[2,1]) <= tola
       # return if rank(B) == 1 and (A-lambda*E,B) uncontrollable
       # return nothing, U, V 
       return nothing, U, [G.c G.s; -G.s G.c]
    end
    # reduce to standard case
    SF = svd(E\B; full=true);
    s1 = SF.S[1]; U = SF.U; v1 = SF.V
    at = U'*(E\A)*U;
    if rankB == 2
       # try with a direct solution, without inversion
       γ = [real(evc[1]) imag(evc[1]); imag(evc[2]) real(evc[2])]; 
       ftry = -B\(A-E*γ);
    end
    sc = at[1,1]+at[2,2];
    sp = real(evc[1]+evc[2]); pp = real(evc[1]*evc[2]);
    k11 = (sc-sp)/s1; 
    k12 = at[2,2]/at[2,1]*k11+(pp-at[1,1]*at[2,2]+at[1,2]*at[2,1])/at[2,1]/s1;
    F = v1*[-k11 -k12; zeros(m-1,2)]*U'; V = Matrix{eltype(B)}(I,m,m) 
    if rankB == 2 && norm(F) > norm(ftry)
      # choose the lower norm feedback 
       F = ftry;
    end
    return F, U, V
    # end saloc2
end
"""
    eigselect1(ev, sdeg, evref, disc; cflag) -> (γ, evupd)

Select a real or complex eigenvalue `γ` to be assigned from a given set of eigenvalues 
in the vector `ev`. The resulting `evupd` contains the remaining eigenvalues from `ev`.
The selected eigenvalue `γ` is the nearest one to the reference value `evref`. 
If `ev = missing` and `disc = false`, then, if `cflag = false` (in the real case) `γ` is set equal to the 
desired stability degree `sdeg`, while if `cflag = true` (in the complex case) `γ` is set 
`γ = sdeg + im*imag(evref)`. 
If `ev = missing` and `disc = true`, then, if `cflag = false` (in the real case) `γ` is set equal to the 
desired stability degree `sdeg`, while if `cflag = true` (in the complex case) `γ` is set 
`γ = sdeg/abs(evref))*evref`. 
If `ev = missing` and `sdeg = missing`, then `γ = nothing`. 
"""
function eigselect1(ev::Union{AbstractVector,Missing},sdeg::Union{Real,Missing},evref::Union{Real,Complex},disc::Bool = false; cflag::Bool = false )
   T = typeof(evref)
   if ismissing(ev)
      if ismissing(sdeg)
         γ = nothing
      else
         γ = disc ? (cflag ? (sdeg/abs(evref))*evref : sdeg) : (cflag ? sdeg+im*imag(evref) : sdeg)
      end
      evupd = missing
   else
      i = argmin(abs.(ev.-evref))
      γ = cflag ? ev[i] : real(ev[i]);
      evupd = [ev[1:i-1,1]; ev[i+1:end,1]]
      isempty(evupd) && (evupd = missing)
   end
   return γ, evupd

# end eigselect1
end
"""
    eigselect2(evr, evc, sdeg, evref, disc) -> (γ, evrupd, evcupd)

Select a pair of eigenvalues `γ[1]` and `γ[2]` to be assigned from given sets of real eigenvalues 
in the vector `evr` and complex eigenvalues in the vector `evc`. 
The selected eigenvalues are the nearest ones to the reference value `evref`. 
A pair of complex eigenvalues is always selected, unless `evc = missing`, in which case a pair 
of real eigenvalues is selected. The resulting `evrupd` contains the remaining real eigenvalues from `evr` and 
the resulting `evcupd` contains the remaining complex eigenvalues from `evc`.  
If the desired stability degree `sdeg` is missing, then a default value `sdeg = sdegdef` is used, where
`sdegdef = -0.05`, if `disc = false`, and `sdegdef = .95`, if `disc = true`. 
If `evc = missing` and `evr` contains at least two eigenvalues, then 
`γ[1]` and `γ[2]` are chosen the two nearest real eigenvalues to `evref`. 
If `evc = missing` and `evr` contains only one value, then `γ[1] = evr` and `γ[2] = sdeg`. 
If `evr = missing`, `evc = missing` and `disc = false`, then 
`γ[1] = sdeg + im*imag(evref)` and `γ[2] = sdeg - im*imag(evref)`. 
If `evr = missing`, `evc = missing` and `disc = true`, then 
`γ[1] = sdeg/abs(evref))*evref[1]` and `γ[2] = sdeg/abs(evref))*evref[2]`. 
"""
function eigselect2(evr::Union{AbstractVector,Missing},evc::Union{AbstractVector,Missing},sdeg::Union{Real,Missing},evref::Union{Real,Complex},disc::Bool)
   
   evref = real(evref) + im*abs(imag(evref))
   if ismissing(evr) && ismissing(evc)
      if ismissing(sdeg)
         T = typeof(evref)
         sdegdef = disc ? real(T)(0.95) : real(T)(-0.05)
         evi = imag(evref)
         γ = [complex(sdegdef,evi); complex(sdegdef,-evi)]
      else
        if disc
           γ = [evref; conj(evref)]
           γ = (sdeg/abs(evref))*γ 
        else
           evi = imag(evref)
           γ = [complex(sdeg,evi); complex(sdeg,-evi)]
        end
      end
      evrupd = missing
      evcupd = missing;
   elseif ismissing(evc)
      # select two real eigenvalues
      if length(evr) < 2
         if ismissing(sdeg)
            T = typeof(evref)
            sdegdef = disc ? real(T)(0.95) : real(T)(-0.05)
            γ = [evr;sdegdef]; 
         else
            γ = [evr;sdeg]; 
         end
         evrupd = missing;
         evcupd = missing;
      else
         evr = evr[sortperm(abs.(evr .- evref))]
         γ = [ evr[1]; evr[2]];
         evrupd = evr[3:end]
         isempty(evrupd) && (evrupd = missing)
         evcupd = missing;
      end
   else
      i = argmin(abs.(evc .- evref))
      γ = [evc[i];evc[i+1]];       
      evcupd = [evc[1:i-1]; evc[i+2:end]];
      isempty(evcupd) && (evcupd = missing)
      evrupd = evr;
   end
   return γ, evrupd, evcupd
   
# end eigselect2
end
"""
    isqtriu(A::AbstractMatrix) -> Bool

Test whether `A` is a square matrix in a quasi upper triangular form 
(e.g., in real or complex Schur form). In the real case, `A` may have 2x2 
diagonal blocks, which however must not correspond to complex conjugate eigenvalues. 
In the complex case, it is tested if `A` is upper triangular.
"""
function isqtriu(A)
   @assert !Base.has_offset_axes(A)
   m, n = size(A)
   m == n || (return false)
   m == 1 && (return true)
   eltype(A)<:Complex && (return istriu(A))
   m == 2 && (return true)
   istriu(A,-1) || (return false)
   for i = 2:m-1
       !iszero(A[i,i-1]) && !iszero(A[i+1,i]) && (return false)
   end
   return true
end
@static if VERSION < v"1.2"
   function eigvalsnosort(M; kwargs...)
      return eigvals(M; kwargs...)
   end
   function eigvalsnosort(M, N; kwargs...)
      ev = eigvals(M, N; kwargs...)
      eltype(M) <: Complex || (ev[imag.(ev) .> 0] = conj(ev[imag.(ev) .< 0]))
      return ev
   end
   function eigvalsnosort!(M; kwargs...)
      return eigvals!(M; kwargs...)
   end
   function eigvalsnosort!(M, N; kwargs...)
      ev = eigvals!(M, N; kwargs...)
      eltype(M) <: Complex || (ev[imag.(ev) .> 0] = conj(ev[imag.(ev) .< 0]))
      return ev
   end
else
   function eigvalsnosort(M; kwargs...)
      return eigvals(M; sortby=nothing, kwargs...)
   end
   function eigvalsnosort(M, N; kwargs...)
      ev = eigvals(M, N; sortby=nothing, kwargs...)
      eltype(M) <: Complex || (ev[imag.(ev) .> 0] = conj(ev[imag.(ev) .< 0]))
      return ev
   end
   function eigvalsnosort!(M; kwargs...)
      return eigvals!(M; sortby=nothing, kwargs...)
   end
   function eigvalsnosort!(M, N; kwargs...)
      ev = eigvals!(M, N; sortby=nothing, kwargs...)
      eltype(M) <: Complex || (ev[imag.(ev) .> 0] = conj(ev[imag.(ev) .< 0]))
      return ev
   end
end
@static if VERSION < v"1.1"
   isnothing(::Any) = false
   isnothing(::Nothing) = true
end


        