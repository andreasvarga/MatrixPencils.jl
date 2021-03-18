"""
    rm2lspm(NUM, DEN; contr = false, obs = false, atol = 0, rtol) -> (A, B, C, D, blkdims)

Construct a representation of the rational matrix `R(λ) := NUM(λ)./DEN(λ)` as the sum of its strictly proper part,  
realized as a structured linearization 

              | A-λI | B | 
              |------|---|
              |  C   | 0 |  

and its polynomial part `D(λ)`. The resulting representation satisfies 

     R(λ) = C*inv(λI-A)*B + D(λ).

The linearization `(A-λI,B,C,0)` of the strictly proper part is in general non-minimal. 
A controllable realization is constructed if `contr = true`, while an observable realization results if `obs = true`. 
The matrix `A` results block diagonal and the vector `blkdims` contains the sizes of the diagonal blocks of `A`,
such that the size of the `i`-th diagonal block is provided in `blkdims[i]`. See [1] for details. 

`NUM(λ)` is a polynomial matrix of the form `NUM(λ) = N_1 + λ N_2 + ... + λ**k N_(k+1)`, for which  
the coefficient matrices `N_i`, `i = 1, ..., k+1` are stored in the 3-dimensional matrix `NUM`, 
where `NUM[:,:,i]` contains the `i`-th coefficient matrix `N_i` (multiplying `λ**(i-1)`). 

`DEN(λ)` is a polynomial matrix of the form `DEN(λ) = D_1 + λ D_2 + ... + λ**l D_(l+1)`, for which 
the coefficient matrices `D_i`, `i = 1, ..., l+1`, are stored in the 3-dimensional matrix `DEN`, 
where `DEN[:,:,i]` contain the `i`-th coefficient matrix `D_i` (multiplying `λ**(i-1)`). 

Alternatively, `NUM(λ)` and `DEN(λ)` can be specified as matrices of elements of the `Polynomial` type 
provided by the [Polynomials](https://github.com/JuliaMath/Polynomials.jl) package. 
In this case, no check is performed that `N(λ)` and `D(λ)` have the same indeterminates.

The polynomial part `D(λ)` is contained in `D`, which is a 2-dimensional array if `D(λ)` has degree 0
or a 3-dimensional array if `D(λ)` has degree  greater than 0. In the latter case `D[:,:,i]` contains the `i`-th 
coefficient matrix multiplying `λ**(i-1)`. 

The keyword arguments `atol` and `rtol`, specify the absolute and relative tolerances for the 
nonzero coefficients of `DEN(λ)`, respectively.  

[1] A. Varga, On computing the Kronecker structure of polynomial and rational matrices using Julia, 2020, 
[arXiv:2006.06825](https://arxiv.org/pdf/2006.06825).
"""
function rm2lspm(NUM::AbstractArray{T1,3},DEN::AbstractArray{T2,3}; contr::Bool = false, obs::Bool = false, 
                 atol::Real = 0, rtol::Real = Base.rtoldefault(float(real(T1)))) where {T1,T2}
    p, m, knum = size(NUM)
    p1, m1, kden = size(DEN)
    (p,m) == (p1, m1) || error("Numerator and denominator polynomial matrices must have the same size")

    degQ1 = 0
    degD1 = 0
    n = 0
    for j = 1:m
        for i = 1:p
            n1 = poldeg1(NUM[i,j,1:knum])
            d1 = poldeg1(DEN[i,j,1:kden])
            d1 == 0 && error("DivideError: zero denominator polynomial")
            n1 < d1 || (degQ1 = max(degQ1,n1-d1+1))
            degD1 = max(degD1,d1)
            n += d1-1
        end
    end
    T = eltype(one(T1)/one(T2))
    degQ1 ≤ 1 ? D = zeros(T,p,m) : D = zeros(T,p,m,max(degQ1,1))
    A = zeros(T,n,n)
    B = zeros(T,n,m)
    C = zeros(T,p,n)
    ONE = one(T)
    contr && obs && (m >= p ? obs = false : contr = false)
    if contr
        k = 1
        nb = 0
        blkdims = zeros(Int,m)
        for j = 1:m
            d1 = poldeg1(DEN[1,j,1:kden])
            d1 == 1 ? mc = [one(T)] : (mc = DEN[1,j,1:d1]) 
            for i = 2:p
                d1 = poldeg1(DEN[i,j,1:kden])
                d1 > 1 && ( mc = pollcm(mc, DEN[i,j,1:d1], atol = atol, rtol = rtol) )
            end
            d1 = poldeg1(mc)
            n1 = d1-1
            i2 = k+n1-1
            if n1 > 0 
                nb += 1
                blkdims[nb] = n1
                mc = mc/mc[d1]
                A[k, k:i2] = -mc[n1:-1:1]
                for ii = k:i2-1
                    A[ii+1, ii] = ONE
                end
                B[k,j] = ONE
                for i = 1:p
                    q, r = poldivrem(NUM[i,j,1:knum],DEN[i,j,1:kden])
                    nq1 = poldeg1(q)
                    nr1 = poldeg1(r)
                    degQ1 ≤ 1 ? D[i,j] = q[1] : D[i,j,1:nq1] = copy_oftype(q[1:nq1],T)
                    if nr1 > 0
                        pc = poldiv(conv(mc,r),DEN[i,j,1:kden])
                        npc = length(pc)
                        C[i,i2-npc+1:i2] = pc[npc:-1:1]
                    end
                end
                k += n1
            else
                for i = 1:p
                    nq1 = poldeg1(NUM[i,j,1:knum])
                    if nq1 > 0
                       q = NUM[i,j,1:nq1]/DEN[i,j,1]
                       degQ1 ≤ 1 ? D[i,j] = q[1] : D[i,j,1:nq1] = copy_oftype(q[1:nq1],T)
                    end
                end 
            end
        end
        n = k-1
        return A[1:n,1:n], B[1:n,:], C[:,1:n], D, blkdims[1:nb]   
    elseif obs
        k = 1
        nb = 0
        blkdims = zeros(Int,p)
        for i = 1:p
            d1 = poldeg1(DEN[i,1,1:kden])
            d1 == 1 ? mc = [one(T)] : (mc = DEN[i,1,1:d1]) 
            for j = 2:m
                d1 = poldeg1(DEN[i,j,1:kden])
                d1 > 1 && ( mc = pollcm(mc, DEN[i,j,1:d1], atol = atol, rtol = rtol) )
            end
            d1 = poldeg1(mc)
            n1 = d1-1
            i2 = k+n1-1
            if n1 > 0 
                nb += 1
                blkdims[nb] = n1
                mc = mc/mc[d1]
                A[k:i2,i2] = -mc[1:n1]
                for ii = k:i2-1
                    A[ii+1,ii] = ONE
                end
                C[i,i2] = ONE               
                for j = 1:m
                    q, r = poldivrem(NUM[i,j,1:knum],DEN[i,j,1:kden])
                    nq1 = poldeg1(q)
                    nr1 = poldeg1(r)
                    degQ1 ≤ 1 ? D[i,j] = q[1] : D[i,j,1:nq1] = copy_oftype(q[1:nq1],T)
                    if nr1 > 0
                        pc = poldiv(conv(mc,r),DEN[i,j,1:kden])
                        npc = length(pc)
                        B[k:k+npc-1,j] = pc[1:npc]
                    end
                end
                k += n1
            else
                for j = 1:m
                    nq1 = poldeg1(NUM[i,j,1:knum])
                    if nq1 > 0
                       q = NUM[i,j,1:nq1]/DEN[i,j,1]
                       degQ1 ≤ 1 ? D[i,j] = q[1] : D[i,j,1:nq1] = copy_oftype(q[1:nq1],T)
                    end
                end
            end
        end
        n = k-1
        return A[1:n,1:n], B[1:n,:], C[:,1:n], D, blkdims[1:nb]   
    else
        k = 1
        blkdims = zeros(Int,m*p)
        nb = 0
        for j = 1:m
            for i = 1:p
                q, r = poldivrem(NUM[i,j,1:knum],DEN[i,j,1:kden])
                nq1 = poldeg1(q)
                nr1 = poldeg1(r)
                degQ1 ≤ 1 ? D[i,j] = q[1] : D[i,j,1:nq1] = copy_oftype(q[1:nq1],T)
                d1 = poldeg1(DEN[i,j,1:kden])
                n1 = d1-1
                i2 = k+n1-1
                if n1 > 0 
                    nb += 1
                    blkdims[nb] = n1
                    s = -DEN[i,j,d1]
                    A[k, k:i2] = DEN[i,j,n1:-1:1]/s
                    for ii = k:i2-1
                        A[ii+1, ii] = ONE
                    end
                    B[k,j] = ONE
                    C[i,i2-nr1+1:i2] = r[nr1:-1:1]/(-s)
                    k += n1
                end
            end
        end
        return A, B, C, D, blkdims[1:nb]   
    end   
end
function rm2lspm(NUM::Union{AbstractVecOrMat{<:Polynomial},Polynomial,Number,AbstractVecOrMat{<:Number}},
                 DEN::Union{AbstractVecOrMat{<:Polynomial},Polynomial,Number,AbstractVecOrMat{<:Number}};kwargs...) 
    return rm2lspm(poly2pm(NUM),poly2pm(DEN);kwargs...)
end
function rm2lspm(NUM::Union{AbstractVecOrMat{<:Polynomial},Polynomial,Number,AbstractVecOrMat{<:Number}};kwargs...) 
   return rm2lspm(poly2pm(NUM);kwargs...)
end
function rm2lspm(NUM::AbstractArray{T,3}; kwargs...) where T
    sys = rm2lspm(NUM,ones(T,size(NUM)[1:2]...,1); kwargs...)
    return sys[1:4]..., [0]
end

"""
    rmeval(NUM,DEN,val) -> Rval

Evaluate the rational matrix `R(λ) := NUM(λ)./DEN(λ)` for `λ = val`. 

`NUM(λ)` is a polynomial matrix of the form `NUM(λ) = N_1 + λ N_2 + ... + λ**k N_(k+1)`, for which  
the coefficient matrices `N_i`, `i = 1, ..., k+1` are stored in the 3-dimensional matrix `N`, 
where `N[:,:,i]` contains the `i`-th coefficient matrix `N_i` (multiplying `λ**(i-1)`). 

`DEN(λ)` is a polynomial matrix of the form `DEN(λ) = D_1 + λ D_2 + ... + λ**l D_(l+1)`, for which 
the coefficient matrices `D_i`, `i = 1, ..., l+1`, are stored in the 3-dimensional matrix `D`, 
where `D[:,:,i]` contain the `i`-th coefficient matrix `D_i` (multiplying `λ**(i-1)`). 

Alternatively, `NUM(λ)` and `DEN(λ)` can be specified as matrices of elements of the `Polynomial` type 
provided by the [Polynomials](https://github.com/JuliaMath/Polynomials.jl) package. 
In this case, no check is performed that `N(λ)` and `D(λ)` have the same indeterminates.
"""
function rmeval(NUM::AbstractArray{T1,3},DEN::AbstractArray{T2,3},val::Number) where {T1,T2}
    return pmeval(NUM,val) ./ pmeval(DEN,val)
end
function rmeval(NUM::AbstractArray{T,3},val::Number) where {T}
    return pmeval(NUM,val) 
end
rmeval(NUM::Union{AbstractVecOrMat{<:Polynomial},Polynomial,Number,AbstractVecOrMat{<:Number}},
       DEN::Union{AbstractVecOrMat{<:Polynomial},Polynomial,Number,AbstractVecOrMat{<:Number}},val::Number) =
       rmeval(poly2pm(NUM),poly2pm(DEN),val::Number)
rmeval(NUM::Union{AbstractVecOrMat{<:Polynomial},Polynomial,Number,AbstractVecOrMat{<:Number}},val::Number) =
       pmeval(poly2pm(NUM),val::Number)
"""
      rm2ls(NUM, DEN; contr = false, obs = false, noseig = false, minimal = false,
            fast = true, atol = 0, rtol) -> (A, E, B, C, D, blkdims)
 
Build a structured linearization as a system matrix `S(λ)` of the form
 
               | A-λE | B | 
        S(λ) = |------|---|
               |  C   | D |  
       
for the rational matrix `R(λ) = NUM(λ) ./ DEN(λ)`, such that `S(λ)` preserves a part of the Kronecker structure of `R(λ)`. 
The regular pencil `A-λE` has the block diagonal form
 
             | Af-λI |   0   | 
      A-λE = |-------|-------|
             |  0    | I-λEi | 

and the dimensions of the diagonal blocks `Af` and `Ei` are provided in `blkdims[1]` and `blkdims[2]`, respectively.

`NUM(λ)` is a polynomial matrix of the form `NUM(λ) = N_1 + λ N_2 + ... + λ**k N_(k+1)`, for which  
the coefficient matrices `N_i`, `i = 1, ..., k+1` are stored in the 3-dimensional matrix `NUM`, 
where `NUM[:,:,i]` contains the `i`-th coefficient matrix `N_i` (multiplying `λ**(i-1)`). 

`DEN(λ)` is a polynomial matrix of the form `DEN(λ) = D_1 + λ D_2 + ... + λ**l D_(l+1)`, for which 
the coefficient matrices `D_i`, `i = 1, ..., l+1`, are stored in the 3-dimensional matrix `DEN`, 
where `DEN[:,:,i]` contain the `i`-th coefficient matrix `D_i` (multiplying `λ**(i-1)`). 

Alternatively, `NUM(λ)` and `DEN(λ)` can be specified as matrices of elements of the `Polynomial` type 
provided by the [Polynomials](https://github.com/JuliaMath/Polynomials.jl) package. 
In this case, no check is performed that `N(λ)` and `D(λ)` have the same indeterminates.
 
If `n` is the order of `A-λE`, then the computed linearization satisfies:
 
(1) `A-λE` is regular and `R(λ) = C*inv(λE-A)*B+D`;

(2) `rank[B A-λE] = n` (controllability) if `minimal = true` or `contr = true`, in which case 
the right Kronecker structure is preserved;

(3) `rank[A-λE; C] = n` (observability)  if `minimal = true` or `obs = true`, in which case 
the left Kronecker structure is preserved;

(4) `A-λE` has no non-dynamic modes if `minimal = true` or `noseig = true`. 

If conditions (1)-(4) are satisfied, the linearization is called `minimal` and the resulting order `n`
is the least achievable order. If conditions (1)-(3) are satisfied, the linearization is called `irreducible` 
and the resulting order `n` is the least achievable order using orthogonal similarity transformations.
For an irreducible linearization `S(λ)` preserves the pole-zero structure (finite and infinite) and the 
left and right Kronecker structures of `R(λ)`. 

The descriptor system based linearization is built using the methods described in [1] in conjunction with
pencil manipulation algorithms [2] and [3] to compute reduced order linearizations. These algorithms 
employ rank determinations based on either the use of 
rank revealing QR-decomposition with column pivoting, if `fast = true`, or the SVD-decomposition, if `fast = false`.
The rank decision based on the SVD-decomposition is generally more reliable, but the involved computational effort is higher.

The keyword arguments `atol` and `rtol`, specify the absolute and relative tolerances, respectively, for the 
nonzero coefficients of `NUM(λ)` and `DEN(λ)`.

[1] A. Varga, On computing the Kronecker structure of polynomial and rational matrices using Julia, 2020, 
[arXiv:2006.06825](https://arxiv.org/pdf/2006.06825).

[2] P. Van Dooreen, The generalized eigenstructure problem in linear system theory, 
IEEE Transactions on Automatic Control, vol. AC-26, pp. 111-129, 1981.

[3] A. Varga, Solving Fault Diagnosis Problems - Linear Synthesis Techniques, Springer Verlag, 2017. 
"""
function rm2ls(NUM::AbstractArray{T1,3},DEN::AbstractArray{T2,3}; minimal::Bool = false, contr::Bool = false, obs::Bool = false, noseig::Bool = false, 
                fast::Bool = true, atol::Real = zero(real(T1)), 
                rtol::Real = (min(size(NUM)...)*eps(real(float(one(T1)))))*iszero(atol)) where {T1,T2}
    Ap, Bp, Cp, D = rm2lspm(NUM, DEN, contr = contr, obs = obs, atol = atol, rtol = rtol)
    minimal && (contr = true; obs = true; noseig = true;)
    if ndims(D) == 2
        A, B, C  = lsminreal(Ap, Bp, Cp, contr = contr, obs = obs, fast = fast, atol = atol, rtol = rtol)
        E = I
        nf = size(A,1)
        ni = 0
    else
        T = typeof(one(T1)/one(T2))
        Ai, Ei, Bi, Ci, Di  = pm2ls(D, minimal = minimal, contr = contr, obs = obs, fast = fast, atol = atol, rtol = rtol) 
        Af, Ef, Bf, Cf, D  = lsminreal2(Ap,I,Bp,Cp,Di,contr = contr, obs = obs, fast = fast, atol1 = atol, atol2 = atol, rtol = rtol)
        nf = size(Af,1)
        ni = size(Ai,1)
        A = [Af zeros(T,nf,ni); zeros(T,ni,nf) Ai]
        E = [Ef zeros(T,nf,ni); zeros(T,ni,nf) Ei]
        B = [Bf; Bi]
        C = [Cf Ci]
    end    
    return A, E, B, C, D, [nf,ni]
end
rm2ls(NUM::Union{AbstractVecOrMat{<:Polynomial},Polynomial,Number,AbstractVecOrMat{<:Number}},
      DEN::Union{AbstractVecOrMat{<:Polynomial},Polynomial,Number,AbstractVecOrMat{<:Number}}; kwargs...) =
       rm2ls(poly2pm(NUM),poly2pm(DEN); kwargs...)
function rm2ls(NUM::AbstractArray{T,3}; kwargs...) where T
    sys = pm2ls(NUM; kwargs...)
    return sys..., [0;size(sys[1],1)]
end
function rm2ls(NUM::Union{AbstractVecOrMat{<:Polynomial},Polynomial,Number,AbstractVecOrMat{<:Number}}; kwargs...)
    sys = pm2ls(poly2pm(NUM); kwargs...)
    return sys..., [0;size(sys[1],1)]
end
"""
    ls2rm(A, E, B, C, D; fast = true, atol1 = 0, atol2 = 0, gaintol = 0, rtol = min(atol1,atol2) > 0 ? 0 : n*ϵ, val) -> (NUM, DEN)

Build the rational matrix `R(λ) = C*inv(λE-A)*B+D := NUM(λ) ./ DEN(λ)` corresponding to its structured linearization 

            | A-λE | B | 
     S(λ) = |------|---|
            |  C   | D |  

by explicitly determining for each rational entry of `R(λ)`, its numerator coefficients from its finite zeros and gain, and 
its denominator coefficients from its finite poles (see [1]). 

The keyword arguments `atol1` and `atol2` specify the absolute tolerances for the elements of `A`, `B`, `C`, `D`, and,  
respectively, of `E`, and `rtol` specifies the relative tolerances for the nonzero elements of `A`, `B`, `C`, `D` and `E`.
The default relative tolerance is `(n+1)*ϵ`, where `n` is the dimension of `A`, and `ϵ` is the 
machine epsilon of the element type of coefficients of `A`. 

The keyword argument `gaintol` specifies the threshold for the magnitude of the nonzero elements of the gain matrix 
`C*inv(γE-A)*B+D`, where `γ = val` if `val` is a number or `γ` is a randomly chosen complex value of unit magnitude, 
if `val = missing`. Generally, `val` should not be a root of any of entries of `P`.

`NUM(λ)` is a polynomial matrix of the form `NUM(λ) = N_1 + λ N_2 + ... + λ**k N_(k+1)`, for which  
the coefficient matrices `N_i`, `i = 1, ..., k+1` are stored in the 3-dimensional matrix `NUM`, 
where `NUM[:,:,i]` contains the `i`-th coefficient matrix `N_i` (multiplying `λ**(i-1)`). 

`DEN(λ)` is a polynomial matrix of the form `DEN(λ) = D_1 + λ D_2 + ... + λ**l D_(l+1)`, for which 
the coefficient matrices `D_i`, `i = 1, ..., l+1`, are stored in the 3-dimensional matrix `DEN`, 
where `DEN[:,:,i]` contain the `i`-th coefficient matrix `D_i` (multiplying `λ**(i-1)`). 

[1] A. Varga Computation of transfer function matrices of generalized state-space models. 
    Int. J. Control, 50:2543–2561, 1989.
"""
function ls2rm(A::AbstractMatrix, E::Union{AbstractMatrix,UniformScaling{Bool}}, B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix; 
                fast::Bool = true, atol1::Real = zero(real(eltype(A))), atol2::Real = zero(real(eltype(E))), 
                gaintol::Real = zero(real(eltype(A))), val::Union{Number,Missing} = missing,
                rtol::Real = ((size(A,1)+1)*eps(real(float(one(eltype(A))))))*iszero(min(atol1,atol2))) 
   emat = (typeof(E) <: AbstractMatrix)
   eident = !emat || isequal(E,I) 
   n = LinearAlgebra.checksquare(A)
   emat && (n,n) != size(E) && throw(DimensionMismatch("A and E must have the same dimensions"))
   p, m = size(D)
   (n,m) == size(B) || throw(DimensionMismatch("A, B and D must have compatible dimensions"))
   (p,n) == size(C) || throw(DimensionMismatch("A, C and D must have compatible dimensions"))

   T = promote_type(eltype(A),eltype(E), eltype(B),eltype(C),eltype(D))
   T <: BlasFloat || (T = promote_type(Float64,T))
   compl = T <: Complex

   ismissing(val) && (val = exp(rand()*im))
   Rval = lseval(A, E, B, C, D, val)
   unimodular = emat && isunimodular(A, E, atol1 = atol1, atol2 = atol2, rtol = rtol)  
   NUM = zeros(T,p,m,n+1) 
   if unimodular 
      DEN = ones(T,p,m,1) 
   else
      DEN = zeros(T,p,m,n+1) 
      DEN[:,:,1] = ones(T,p,m)
   end
   for i = 1:p
       Ao, Eo, Bo, Co, Do = lsminreal2(A, E, B, C[i:i,:], D[i:i,:], contr = false, infinite = false, noseig = false, 
                                       atol1 = atol1, atol2 = atol2, rtol = rtol)
       for j = 1:m
           if abs(Rval[i,j]) > gaintol
              A1, E1, B1, C1, D1 = lsminreal2(Ao, Eo, Bo[:,j:j], Co, Do[:,j:j], obs = false, infinite = false, noseig = false, 
                                              atol1 = atol1, atol2 = atol2, rtol = rtol)
              zer, iz, _ = spzeros(A1, E1, B1, C1, D1; fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol) 
              cz, zval = polcoeffval(zer[1:(length(zer)-sum(iz))],val)
              if unimodular
                 NUM[i,j,1:length(cz)] = compl ? cz*(Rval[i,j]/zval) : real(cz*(Rval[i,j]/zval)) 
              else
                 pol, ip, _ = pzeros(A1, E1; fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol) 
                 cp, pval = polcoeffval(pol[1:(length(pol)-sum(ip))],val)
                 NUM[i,j,1:length(cz)] = compl ? cz*(Rval[i,j]*pval/zval) : real(cz*(Rval[i,j]*pval/zval)) 
                 DEN[i,j,1:length(cp)] = compl ? cp : real(cp) 
              end
            end
       end
   end
   return NUM[:,:,1:pmdeg(NUM)+1], DEN[:,:,1:pmdeg(DEN)+1]
end
ls2rm(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix; kwargs...) = 
      ls2rm(A::AbstractMatrix, I, B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix; kwargs...)
"""
     rm2lps(NUM, DEN; contr = false, obs = false) -> (A, E, B, F, C, G, D, H, blkdims)

Build a structured pencil based linearization as a system matrix `S(λ)` of the form

            | A-λE | B-λF | 
     S(λ) = |------|------|
            | C-λG | D-λH |  
      
for the rational matrix `R(λ) = NUM(λ) ./ DEN(λ)` such that `S(λ)` preserves a part of the Kronecker structure of `R(λ)`. 
The regular pencil `A-λE` has the block diagonal form
 
             | Af-λI |   0    | 
      A-λE = |-------|--------|
             |  0    | Ai-λEi | 

and the dimensions of the diagonal blocks `Af` and `Ai` are provided in `blkdims[1]` and `blkdims[2]`, respectively.

`NUM(λ)` is a polynomial matrix of the form `NUM(λ) = N_1 + λ N_2 + ... + λ**k N_(k+1)`, for which  
the coefficient matrices `N_i`, `i = 1, ..., k+1` are stored in the 3-dimensional matrix `NUM`, 
where `NUM[:,:,i]` contains the `i`-th coefficient matrix `N_i` (multiplying `λ**(i-1)`). 

`DEN(λ)` is a polynomial matrix of the form `DEN(λ) = D_1 + λ D_2 + ... + λ**l D_(l+1)`, for which 
the coefficient matrices `D_i`, `i = 1, ..., l+1`, are stored in the 3-dimensional matrix `DEN`, 
where `DEN[:,:,i]` contain the `i`-th coefficient matrix `D_i` (multiplying `λ**(i-1)`). 

Alternatively, `NUM(λ)` and `DEN(λ)` can be specified as matrices of elements of the `Polynomial` type 
provided by the [Polynomials](https://github.com/JuliaMath/Polynomials.jl) package. 
In this case, no check is performed that `N(λ)` and `D(λ)` have the same indeterminates.

If `n` is the order of the matrix `A`, then the computed linearization satisfies:

(1) `A-λE` is a `n x n` regular pencil;

(2) `R(λ) = (C-λG)*inv(λE-A)*(B-λF)+D-λH`;

(3) `rank[B-λF A-λE] = n` for any finite and infinite `λ` (strong controllability) if `contr = true`, in which case 
the right Kronecker structure is preserved;

(4) `rank[A-λE; C-λG] = n` for any finite and infinite `λ` (strong observability)  if `obs = true`, in which case 
the left Kronecker structure is preserved. 

If conditions (1)-(4) are satisfied, the linearization is called `strongly minimal`, the resulting order `n`
is the least achievable order and `S(λ)` preserves the pole-zero structure (finite and infinite) and the 
left and right Kronecker structures of `R(λ)`. 

The pencil based linearization is built using the methods described in [1] in conjunction with
pencil manipulation algorithms [2] and [3] to compute reduced order linearizations. These algorithms 
employ rank determinations based on either the use of 
rank revealing QR-decomposition with column pivoting, if `fast = true`, or the SVD-decomposition, if `fast = false`.
The rank decision based on the SVD-decomposition is generally more reliable, but the involved computational effort is higher.

The keyword arguments `atol` and `rtol`, specify the absolute and relative tolerances for the 
nonzero coefficients of `NUM(λ)` and `DEN(λ)`, respectively.

[1] A. Varga, On computing the Kronecker structure of polynomial and rational matrices using Julia, 2020, 
[arXiv:2006.06825](https://arxiv.org/pdf/2006.06825).

[2] P. Van Dooreen, The generalized eigenstructure problem in linear system theory, 
IEEE Transactions on Automatic Control, vol. AC-26, pp. 111-129, 1981.

[3] F.M. Dopico, M.C. Quintana and P. Van Dooren, Linear system matrices of rational transfer functions, 
in "Realization and Model Reduction of Dynamical Systems, A Festschrift to honor the 70th birthday of Thanos Antoulas", 
Eds. C. Beattie, P. Benner, M. Embree, S. Gugercin and S. Lefteriu, Springer-Verlag, 2020. 
[arXiv:1903.05016](https://arxiv.org/pdf/1903.05016.pdf)
"""
function rm2lps(NUM::AbstractArray{T1,3},DEN::AbstractArray{T2,3}; minimal::Bool = false, contr::Bool = false, obs::Bool = false, noseig::Bool = false, 
                fast::Bool = true, atol::Real = zero(real(T1)), 
                rtol::Real = (min(size(NUM)...)*eps(real(float(one(T1)))))*iszero(atol)) where {T1,T2}
    T = typeof(one(T1)/one(T2))
    Ap, Bp, Cp, D = rm2lspm(NUM,DEN)
    minimal && (contr = true; obs = true; noseig = true;)
    if ndims(D) == 2
        A, B, C, _  = lsminreal(Ap, Bp, Cp, contr = contr, obs = obs, fast = fast, atol = atol, rtol = rtol)
        F = zeros(T,size(B)...)
        G = zeros(T,size(C)...)
        H = zeros(T,size(D)...)
        E = Matrix{T}(I,size(A)...)
        nf = size(A,1)
        ni = 0
    else
        Ai, Ei, Bi, Fi, Ci, Gi, D, H  = lpsminreal(pm2lps(D)...; contr = contr, obs = obs,
                                                    fast = fast, atol1 = atol, atol2 = atol, rtol = rtol)   
        Ap, Bp, Cp, _  = lsminreal(Ap, Bp, Cp, contr = contr, obs = obs, fast = fast, atol = atol,  rtol = rtol)
        nf = size(Ap,1)
        ni = size(Ai,1)
        A = [Ap zeros(T,nf,ni); zeros(T,ni,nf) Ai]
        E = [I zeros(T,nf,ni); zeros(T,ni,nf) Ei]
        B = [Bp; Bi]
        F = [zeros(T,size(Bp)...); Fi]
        C = [Cp Ci]
        G = [zeros(T,size(Cp)...) Gi]
    end    
    return A, E, B, F, C, G, D, H, [nf, ni]
end
rm2lps(NUM::Union{AbstractVecOrMat{<:Polynomial},Polynomial,Number,AbstractVecOrMat{<:Number}},
       DEN::Union{AbstractVecOrMat{<:Polynomial},Polynomial,Number,AbstractVecOrMat{<:Number}}; kwargs...) =
       rm2lps(poly2pm(NUM),poly2pm(DEN); kwargs...)
function rm2lps(NUM::AbstractArray{T,3}; kwargs...) where T 
    sys = pm2lps(NUM; kwargs...)
    return sys..., [0;size(sys[1],1)]
end
function rm2lps(NUM::Union{AbstractVecOrMat{<:Polynomial},Polynomial,Number,AbstractVecOrMat{<:Number}}; kwargs...) 
    sys = pm2lps(poly2pm(NUM); kwargs...)
    return sys..., [0;size(sys[1],1)]
end
"""
    lps2rm(A, E, B, F, C, G, D, H; fast = true, atol1 = 0, atol2 = 0, gaintol = 0, rtol = min(atol1,atol2) > 0 ? 0 : n*ϵ, val) -> (NUM, DEN)

Build the rational matrix `R(λ) = (C-λG)*inv(λE-A)*(B-λF)+D-λH := NUM(λ) ./ DEN(λ)` corresponding to its structured linearization 

            | A-λE | B-λF | 
     S(λ) = |------|------|
            | C-λG | D-λH |  

by explicitly determining for each rational entry of `R(λ)`, its numerator coefficients from its finite zeros and gain, and 
its denominator coefficients from its finite poles. An extension of the approach of [1] is employed, relying on the procedures
of [2] to compute strongly irreducible linearizations.

The keyword arguments `atol1` and `atol2` specify the absolute tolerances for the elements of `A`, `B`, `C`, `D`, and of  
`E`, `F`, `G`, `H`, respectively,  and `rtol` specifies the relative tolerances for the nonzero elements of 
`A`, `B`, `C`, `D`, `E`, F`, `G`, `H`. 
The default relative tolerance is `(n+2)*ϵ`, where `n` is the size of the size dimension of `A`, and `ϵ` is the 
machine epsilon of the element type of coefficients of `A`. 

The keyword argument `gaintol` specifies the threshold for the magnitude of the nonzero elements of the gain matrix 
`C*inv(γE-A)*B+D`, where `γ = val` if `val` is a number or `γ` is a randomly chosen complex value of unit magnitude, 
if `val = missing`. Generally, `val` should not be a root of any of entries of `P`. 

[1] A. Varga Computation of transfer function matrices of generalized state-space models. 
    Int. J. Control, 50:2543–2561, 1989.

[2] F.M. Dopico, M.C. Quintana and P. Van Dooren, Linear system matrices of rational transfer functions, 
in "Realization and Model Reduction of Dynamical Systems, A Festschrift to honor the 70th birthday of Thanos Antoulas", 
Eds. C. Beattie, P. Benner, M. Embree, S. Gugercin and S. Lefteriu, Springer-Verlag, 2020. 
[arXiv:1903.05016](https://arxiv.org/pdf/1903.05016.pdf)
"""
function lps2rm(A::AbstractMatrix, E::AbstractMatrix, B::AbstractMatrix, F::AbstractMatrix, 
                C::AbstractMatrix, G::AbstractMatrix, D::AbstractMatrix, H::AbstractMatrix; 
                fast::Bool = true, atol1::Real = zero(real(eltype(A))), atol2::Real = zero(real(eltype(E))), 
                gaintol::Real = zero(real(eltype(A))), val::Union{Number,Missing} = missing,
                rtol::Real = ((size(A,1)+2)*eps(real(float(one(eltype(A))))))*iszero(min(atol1,atol2))) 
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

   compl = T <: Complex
   ismissing(val) && (val = exp(rand()*im))
   Rval = lpseval(A, E, B, F, C, G, D, H, val)
   unimodular = isunimodular(A, E, atol1 = atol1, atol2 = atol2, rtol = rtol)  
   NUM = zeros(T,p,m,n+2) 
   if unimodular 
      DEN = ones(T,p,m,1) 
   else
      DEN = zeros(T,p,m,n+2) 
      DEN[:,:,1] = ones(T,p,m)
   end
   for i = 1:p
       Ao, Eo, Bo, Fo, Co, Go, Do, Ho, Vo, Wo = lpsminreal(A, E, B, F, C[i:i,:], G[i:i,:], D[i:i,:], H[i:i,:], 
                                                           contr = false, atol1 = atol1, atol2 = atol2, rtol = rtol)
       for j = 1:m
           A1, E1, B1, F1, C1, G1, D1, H1, _ = lpsminreal(Ao, Eo, (Bo/Wo)[:,j:j], (Fo/Wo)[:,j:j], Co, Go, (Do/Wo)[:,j:j], (Ho/Wo)[:,j:j], 
                                           obs = false, atol1 = atol1, atol2 = atol2, rtol = rtol)
           if abs(Rval[i,j]) > gaintol
              zer, iz, _ = pzeros([A1 B1; C1 D1],[E1 F1; G1 H1]; fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol) 
              cz, zval = polcoeffval(zer[1:(length(zer)-sum(iz))],val)
              if unimodular
                 NUM[i,j,1:length(cz)] = compl ? cz*(Rval[i,j]/zval) : real(cz*(Rval[i,j]/zval)) 
              else
                 n1 = size(A1,1)       
                 M1 = [A1 zeros(T,n1,2); zeros(T,1,n1+1) one(T); zeros(T,1,n1) one(T)  zero(T)]
                 N1 = [E1 F1 zeros(T,n1,1); G1 H1 zero(T); zeros(T,1,n1+2)]
                 pol, ip, _ = pzeros(M1,N1; fast = fast, atol1 = atol1, atol2 = atol2, rtol = rtol) 
                 cp, pval = polcoeffval(pol[1:(length(pol)-sum(ip))],val)
                 NUM[i,j,1:length(cz)] = compl ? cz*(Rval[i,j]*pval/zval) : real(cz*(Rval[i,j]*pval/zval)) 
                 DEN[i,j,1:length(cp)] = compl ? cp : real(cp) 
              end
           end
       end
   end
   return NUM[:,:,1:pmdeg(NUM)+1], DEN[:,:,1:pmdeg(DEN)+1]
end
"""
     lpmfd2ls(DEN, NUM; fast = true, contr = false, obs = false, minimal = false, atol = 0, rtol) -> (A, E, B, C, D)
            
Build a structured linearization as a system matrix `S(λ)` of the form

            | A-λE | B | 
     S(λ) = |------|---|
            |  C   | D |  

of the left polynomial matrix fractional description `R(λ) = inv(DEN(λ))*NUM(λ)`, 
such that `R(λ) = C*inv(λE-A)*B+D`. The resulting linearization `S(λ)` preserves a part, 
if `minimal = false`, or the complete Kronecker structure, if `minimal = true`, of `R(λ)`. 
In the latter case, the order `n` of `A-λE` is the least possible one.

`DEN(λ)` and `NUM(λ)` can be specified as polynomial matrices of the form `X(λ) = X_1 + λ X_2 + ... + λ**k X_(k+1)`, 
for `X = DEN` or `X = NUM`, for which the coefficient matrices `X_i`, `i = 1, ..., k+1`, are stored in 
the 3-dimensional matrices `X`, where `X[:,:,i]` contains the `i`-th coefficient matrix `X_i` (multiplying `λ**(i-1)`). 

`DEN(λ)` and `NUM(λ)` can also be specified as matrices, vectors or scalars of elements of the `Polynomial` type 
provided by the [Polynomials](https://github.com/JuliaMath/Polynomials.jl) package.   
In this case, no check is performed that `N(λ)` and `D(λ)` have the same indeterminates.

The computed structured linearization satisfies:

(1) `A-λE` is regular;

(2) `rank[B A-λE] = n` (controllability) if `minimal = true` or `contr = true`, in which case 
the right Kronecker structure is preserved;

(3) `rank[A-λE; C] = n` (observability)  if `minimal = true` or `obs = true`, in which case 
the left Kronecker structure is preserved;

(4) `S(λ)` has no simple infinite eigenvalues if `minimal = true`. 

If conditions (1)-(4) are satisfied, the linearization is called `minimal` and the resulting order `n`
is the least achievable order. If conditions (1)-(3) are satisfied, the linearization is called `irreducible` 
and the resulting order `n` is the least achievable order using orthogonal similarity transformations.
For an irreducible linearization `S(λ)` preserves the pole-zero structure (finite and infinite) and the 
left and right Kronecker structures of `R(λ)`. 

The keyword arguments `atol` and `rtol`, specify, respectively, the absolute and relative tolerance for the 
nonzero coefficients of the matrices `DEN(λ)` and `NUM(λ)`. The default relative tolerance is `nt*ϵ`, 
where `nt` is the size of the square matrix `DEN(λ)` and `ϵ` is the machine epsilon of the element type of its coefficients. 

The structured linearization is built using the methods described in [1].

[1] A. Varga, On computing the Kronecker structure of polynomial and rational matrices using Julia, 2020, 
[arXiv:2006.06825](https://arxiv.org/pdf/2006.06825).
"""
function lpmfd2ls(DEN::Union{AbstractArray{T1,3},AbstractArray{T1,2}},NUM::Union{AbstractArray{T2,3},AbstractArray{T2,2}}; 
                  contr::Bool = false, obs::Bool = false, minimal::Bool = false, 
                  fast::Bool = true, atol::Real = zero(real(T1)), 
                  rtol::Real = size(DEN,1)*eps(real(float(one(T1))))*iszero(atol)) where {T1, T2}
    p = size(NUM,1) 
    m = size(NUM,2)  
    p1 =  size(DEN,1)          
    p1 == size(DEN,2) || throw(DimensionMismatch("DEN(λ) must be a square polynomial matrix"))
    p == p1 || throw(DimensionMismatch("DEN(λ) and NUM(λ) must have the same number of rows"))
    
    return spm2ls(DEN, NUM, Matrix{T1}(I,p,p), zeros(T1,p,m), contr = contr, obs = obs, minimal = minimal, 
                  fast = fast, atol = atol, rtol = rtol) 
end
function lpmfd2ls(DEN::Union{AbstractVecOrMat{<:Polynomial},Polynomial,Number,AbstractVecOrMat{<:Number}}, 
                  NUM::Union{AbstractVecOrMat{<:Polynomial},Polynomial,Number,AbstractVecOrMat{<:Number}}; kwargs...) 
    return lpmfd2ls(poly2pm(DEN),poly2pm(NUM); kwargs...)
end
"""
     rpmfd2ls(DEN, NUM; fast = true, contr = false, obs = false, minimal = false, atol = 0, rtol) -> (A, E, B, C, D)
            
Build a structured linearization as a system matrix `S(λ)` of the form

            | A-λE | B | 
     S(λ) = |------|---|
            |  C   | D |  

of the right polynomial matrix fractional description `R(λ) = NUM(λ)*inv(DEN(λ))`, 
such that `R(λ) = C*inv(λE-A)*B+D`. The resulting linearization `S(λ)` preserves a part, 
if `minimal = false`, or the complete Kronecker structure, if `minimal = true`, of `R(λ)`. 
In the latter case, the order `n` of `A-λE` is the least possible one.

`DEN(λ)` and `NUM(λ)` can be specified as polynomial matrices of the form `X(λ) = X_1 + λ X_2 + ... + λ**k X_(k+1)`, 
for `X = DEN` or `X = NUM`, for which the coefficient matrices `X_i`, `i = 1, ..., k+1`, are stored in 
the 3-dimensional matrices `X`, where `X[:,:,i]` contains the `i`-th coefficient matrix `X_i` (multiplying `λ**(i-1)`). 

`DEN(λ)` and `NUM(λ)` can also be specified as matrices, vectors or scalars of elements of the `Polynomial` type 
provided by the [Polynomials](https://github.com/JuliaMath/Polynomials.jl) package.   
In this case, no check is performed that `N(λ)` and `D(λ)` have the same indeterminates.

The computed structured linearization satisfies:

(1) `A-λE` is regular;

(2) `rank[B A-λE] = n` (controllability) if `minimal = true` or `contr = true`, in which case 
the right Kronecker structure is preserved;

(3) `rank[A-λE; C] = n` (observability)  if `minimal = true` or `obs = true`, in which case 
the left Kronecker structure is preserved;

(4) `S(λ)` has no simple infinite eigenvalues if `minimal = true`. 

If conditions (1)-(4) are satisfied, the linearization is called `minimal` and the resulting order `n`
is the least achievable order. If conditions (1)-(3) are satisfied, the linearization is called `irreducible` 
and the resulting order `n` is the least achievable order using orthogonal similarity transformations.
For an irreducible linearization `S(λ)` preserves the pole-zero structure (finite and infinite) and the 
left and right Kronecker structures of `R(λ)`. 

The keyword arguments `atol` and `rtol`, specify, respectively, the absolute and relative tolerance for the 
nonzero coefficients of the matrices `DEN(λ)` and `NUM(λ)`. The default relative tolerance is `nt*ϵ`, 
where `nt` is the size of the square matrix `DEN(λ)` and `ϵ` is the machine epsilon of the element type of its coefficients. 

The structured linearization is built using the methods described in [1].

[1] A. Varga, On computing the Kronecker structure of polynomial and rational matrices using Julia, 2020, 
[arXiv:2006.06825](https://arxiv.org/pdf/2006.06825).
"""
function rpmfd2ls(DEN::Union{AbstractArray{T1,3},AbstractArray{T1,2}},NUM::Union{AbstractArray{T2,3},AbstractArray{T2,2}}; 
                  contr::Bool = false, obs::Bool = false, minimal::Bool = false, 
                  fast::Bool = true, atol::Real = zero(real(T1)), 
                  rtol::Real = size(DEN,1)*eps(real(float(one(T1))))*iszero(atol)) where {T1, T2}
    p = size(NUM,1) 
    m = size(NUM,2)  
    m1 =  size(DEN,1)          
    m1 == size(DEN,2) || throw(DimensionMismatch("DEN(λ) must be a square polynomial matrix"))
    m == m1 || throw(DimensionMismatch("DEN(λ) and NUM(λ) must have the same number of rows"))
    
    return spm2ls(DEN, Matrix{T1}(I,m,m), NUM, zeros(T1,p,m), contr = contr, obs = obs, minimal = minimal, 
                  fast = fast, atol = atol, rtol = rtol) 
end
function rpmfd2ls(DEN::Union{AbstractVecOrMat{<:Polynomial},Polynomial,Number,AbstractVecOrMat{<:Number}}, 
                  NUM::Union{AbstractVecOrMat{<:Polynomial},Polynomial,Number,AbstractVecOrMat{<:Number}}; kwargs...) 
    return rpmfd2ls(poly2pm(DEN),poly2pm(NUM); kwargs...)
end
"""
     lpmfd2lps(DEN, NUM; fast = true, contr = false, obs = false, minimal = false, atol = 0, rtol) -> (A, E, B, F, C, G, D, H)
            
Build a structured linearization as a system matrix `S(λ)` of the form

            | A-λE | B-λF | 
     S(λ) = |------|------|
            | C-λG | D-λH |  

of the left polynomial matrix fractional description `R(λ) = inv(DEN(λ))*NUM(λ)`, 
such that `R(λ) = (C-λG)*inv(λE-A)*(B-λF)+D-λH`. The resulting linearization `S(λ)` preserves a part, 
if `minimal = false`, or the complete Kronecker structure, if `minimal = true`, of `R(λ)`. 
In the latter case, the order `n` of `A-λE` is the least possible one.

`DEN(λ)` and `NUM(λ)` can be specified as polynomial matrices of the form `X(λ) = X_1 + λ X_2 + ... + λ**k X_(k+1)`, 
for `X = DEN` or `X = NUM`, for which the coefficient matrices `X_i`, `i = 1, ..., k+1`, are stored in 
the 3-dimensional matrices `X`, where `X[:,:,i]` contains the `i`-th coefficient matrix `X_i` (multiplying `λ**(i-1)`). 

`DEN(λ)` and `NUM(λ)` can also be specified as matrices, vectors or scalars of elements of the `Polynomial` type 
provided by the [Polynomials](https://github.com/JuliaMath/Polynomials.jl) package.   
In this case, no check is performed that `N(λ)` and `D(λ)` have the same indeterminates.

The computed structured linearization satisfies:

(1) `A-λE` is regular;

(2) `rank[B-λF A-λE] = n` (strong controllability) if `minimal = true` or `contr = true`, in which case 
the right Kronecker structure is preserved;

(3) `rank[A-λE; C-λG] = n` (strong observability)  if `minimal = true` or `obs = true`, in which case 
the left Kronecker structure is preserved.

If conditions (1)-(3) are satisfied, the linearization is called `strongly minimal`, the resulting order `n`
is the least achievable order and `S(λ)` preserves the pole-zero structure (finite and infinite) and the 
left and right Kronecker structures of `R(λ)`. 

The keyword arguments `atol` and `rtol`, specify, respectively, the absolute and relative tolerance for the 
nonzero coefficients of the matrices `DEN(λ)` and `NUM(λ)`. The default relative tolerance is `nt*ϵ`, 
where `nt` is the size of the square matrix `DEN(λ)` and `ϵ` is the machine epsilon of the element type of its coefficients. 

The structured linearization is built using the methods described in [1].

[1] A. Varga, On computing the Kronecker structure of polynomial and rational matrices using Julia, 2020, 
[arXiv:2006.06825](https://arxiv.org/pdf/2006.06825).
"""
function lpmfd2lps(DEN::Union{AbstractArray{T1,3},AbstractArray{T1,2}},NUM::Union{AbstractArray{T2,3},AbstractArray{T2,2}}; 
                  contr::Bool = false, obs::Bool = false, minimal::Bool = false, 
                  fast::Bool = true, atol::Real = zero(real(T1)), 
                  rtol::Real = size(DEN,1)*eps(real(float(one(T1))))*iszero(atol)) where {T1, T2}
    p = size(NUM,1) 
    m = size(NUM,2)  
    p1 =  size(DEN,1)          
    p1 == size(DEN,2) || throw(DimensionMismatch("DEN(λ) must be a square polynomial matrix"))
    p == p1 || throw(DimensionMismatch("DEN(λ) and NUM(λ) must have the same number of rows"))
    
    return spm2lps(DEN, NUM, Matrix{T1}(I,p,p), zeros(T1,p,m), contr = contr, obs = obs, minimal = minimal, 
                  fast = fast, atol = atol, rtol = rtol) 
end
function lpmfd2lps(DEN::Union{AbstractVecOrMat{<:Polynomial},Polynomial,Number,AbstractVecOrMat{<:Number}}, 
                   NUM::Union{AbstractVecOrMat{<:Polynomial},Polynomial,Number,AbstractVecOrMat{<:Number}}; kwargs...) 
    return lpmfd2lps(poly2pm(DEN),poly2pm(NUM); kwargs...)
end
"""
     rpmfd2lps(DEN, NUM; fast = true, contr = false, obs = false, minimal = false, atol = 0, rtol) -> (A, E, B, F, C, G, D, H)
            
Build a structured linearization as a system matrix `S(λ)` of the form

            | A-λE | B-λF | 
     S(λ) = |------|------|
            | C-λG | D-λH |  

of the right polynomial matrix fractional description `R(λ) = NUM(λ)*inv(DEN(λ))`, 
such that `R(λ) = (C-λG)*inv(λE-A)*(B-λF)+D-λH`. The resulting linearization `S(λ)` preserves a part, 
if `minimal = false`, or the complete Kronecker structure, if `minimal = true`, of `R(λ)`. 
In the latter case, the order `n` of `A-λE` is the least possible one.

`DEN(λ)` and `NUM(λ)` can be specified as polynomial matrices of the form `X(λ) = X_1 + λ X_2 + ... + λ**k X_(k+1)`, 
for `X = DEN` or `X = NUM`, for which the coefficient matrices `X_i`, `i = 1, ..., k+1`, are stored in 
the 3-dimensional matrices `X`, where `X[:,:,i]` contains the `i`-th coefficient matrix `X_i` (multiplying `λ**(i-1)`). 

`DEN(λ)` and `NUM(λ)` can also be specified as matrices, vectors or scalars of elements of the `Polynomial` type 
provided by the [Polynomials](https://github.com/JuliaMath/Polynomials.jl) package.   
In this case, no check is performed that `N(λ)` and `D(λ)` have the same indeterminates.

The computed structured linearization satisfies:

(1) `A-λE` is regular;

(2) `rank[B-λF A-λE] = n` (strong controllability) if `minimal = true` or `contr = true`, in which case 
the right Kronecker structure is preserved;

(3) `rank[A-λE; C-λG] = n` (strong observability)  if `minimal = true` or `obs = true`, in which case 
the left Kronecker structure is preserved.

If conditions (1)-(3) are satisfied, the linearization is called `strongly minimal`, the resulting order `n`
is the least achievable order and `S(λ)` preserves the pole-zero structure (finite and infinite) and the 
left and right Kronecker structures of `R(λ)`. 

The keyword arguments `atol` and `rtol`, specify, respectively, the absolute and relative tolerance for the 
nonzero coefficients of the matrices `DEN(λ)` and `NUM(λ)`. The default relative tolerance is `nt*ϵ`, 
where `nt` is the size of the square matrix `DEN(λ)` and `ϵ` is the machine epsilon of the element type of its coefficients. 

The structured linearization is built using the methods described in [1].

[1] A. Varga, On computing the Kronecker structure of polynomial and rational matrices using Julia, 2020, 
[arXiv:2006.06825](https://arxiv.org/pdf/2006.06825).
"""
function rpmfd2lps(DEN::Union{AbstractArray{T1,3},AbstractArray{T1,2}},NUM::Union{AbstractArray{T2,3},AbstractArray{T2,2}}; 
                  contr::Bool = false, obs::Bool = false, minimal::Bool = false, 
                  fast::Bool = true, atol::Real = zero(real(T1)), 
                  rtol::Real = size(DEN,1)*eps(real(float(one(T1))))*iszero(atol)) where {T1, T2}
    p = size(NUM,1) 
    m = size(NUM,2)  
    m1 =  size(DEN,1)          
    m1 == size(DEN,2) || throw(DimensionMismatch("DEN(λ) must be a square polynomial matrix"))
    m == m1 || throw(DimensionMismatch("DEN(λ) and NUM(λ) must have the same number of rows"))
    
    return spm2lps(DEN, Matrix{T1}(I,m,m), NUM, zeros(T1,p,m), contr = contr, obs = obs, minimal = minimal, 
                  fast = fast, atol = atol, rtol = rtol) 
end
function rpmfd2lps(DEN::Union{AbstractVecOrMat{<:Polynomial},Polynomial,Number,AbstractVecOrMat{<:Number}}, 
                   NUM::Union{AbstractVecOrMat{<:Polynomial},Polynomial,Number,AbstractVecOrMat{<:Number}}; kwargs...) 
    return rpmfd2lps(poly2pm(DEN),poly2pm(NUM); kwargs...)
end
"""
     pminv2ls(P; fast = true, contr = false, obs = false, minimal = false, atol = 0, rtol) -> (A, E, B, C, D)
            
Build a structured linearization as a system matrix `S(λ)` of the form

            | A-λE | B | 
     S(λ) = |------|---|
            |  C   | D |  

of the inverse `R(λ) = inv(P(λ))`, such that `R(λ) = C*inv(λE-A)*B+D`. 
The resulting linearization `S(λ)` preserves a part, if `minimal = false`, or 
the complete Kronecker structure, if `minimal = true`, of `R(λ)`. 
In the latter case, the order `n` of `A-λE` is the least possible one.

`P(λ)` can be specified as a polynomial matrix of the form `P(λ) = P_1 + λ P_2 + ... + λ**k P_(k+1)`, 
for which the coefficient matrices `P_i`, `i = 1, ..., k+1`, are stored in 
the 3-dimensional matrix `P`, where `P[:,:,i]` contains the `i`-th coefficient matrix `P_i` (multiplying `λ**(i-1)`). 

`P(λ)` can also be specified as a matrix or scalar of elements of the `Polynomial` type 
provided by the [Polynomials](https://github.com/JuliaMath/Polynomials.jl) package.   

The computed structured linearization satisfies:

(1) `A-λE` is regular;

(2) `rank[B A-λE] = n` (controllability) if `minimal = true` or `contr = true`, in which case 
the right Kronecker structure is preserved;

(3) `rank[A-λE; C] = n` (observability)  if `minimal = true` or `obs = true`, in which case 
the left Kronecker structure is preserved;

(4) `S(λ)` has no simple infinite eigenvalues if `minimal = true`. 

If conditions (1)-(4) are satisfied, the linearization is called `minimal` and the resulting order `n`
is the least achievable order. If conditions (1)-(3) are satisfied, the linearization is called `irreducible` 
and the resulting order `n` is the least achievable order using orthogonal similarity transformations.
For an irreducible linearization `S(λ)` preserves the pole-zero structure (finite and infinite) and the 
left and right Kronecker structures of `R(λ)`. 

The keyword arguments `atol` and `rtol`, specify, respectively, the absolute and relative tolerance for the 
nonzero coefficients of the matrix `P(λ)`. The default relative tolerance is `nt*ϵ`, 
where `nt` is the size of the square matrix `P(λ)` and `ϵ` is the machine epsilon of the element type of its coefficients. 

The structured linearization is built using the methods described in [1].

[1] A. Varga, On computing the Kronecker structure of polynomial and rational matrices using Julia, 2020, 
[arXiv:2006.06825](https://arxiv.org/pdf/2006.06825).
"""
function pminv2ls(P::Union{AbstractArray{T1,3},AbstractArray{T1,2}}; 
                  contr::Bool = false, obs::Bool = false, minimal::Bool = false, 
                  fast::Bool = true, atol::Real = zero(real(T1)), 
                  rtol::Real = size(P,1)*eps(real(float(one(T1))))*iszero(atol)) where {T1}
    n = size(P,1) 
    n == size(P,2) || throw(DimensionMismatch("P(λ) must be a square polynomial matrix"))
    
    return spm2ls(P, Matrix{T1}(I,n,n), Matrix{T1}(I,n,n), zeros(T1,n,n), contr = contr, obs = obs, minimal = minimal, 
                  fast = fast, atol = atol, rtol = rtol) 
end
function pminv2ls(P::Union{AbstractVecOrMat{<:Polynomial},Polynomial,Number,AbstractVecOrMat{<:Number}}; kwargs...)
    return pminv2ls(poly2pm(P); kwargs...)
end
"""
     pminv2lps(P; fast = true, contr = false, obs = false, minimal = false, atol = 0, rtol) -> (A, E, B, F, C, G, D, H)
            
Build a structured linearization as a system matrix `S(λ)` of the form

            | A-λE | B-λF | 
     S(λ) = |------|------|
            | C-λG | D-λH |  

of the inverse `R(λ) = inv(P(λ))`, such that `R(λ) = (C-λG)*inv(λE-A)*(B-λF)+D-λH`. 
The resulting linearization `S(λ)` preserves a part, if `minimal = false`, or 
the complete Kronecker structure, if `minimal = true`, of `R(λ)`. 
In the latter case, the order `n` of `A-λE` is the least possible one.

`P(λ)` can be specified as a polynomial matrix of the form `P(λ) = P_1 + λ P_2 + ... + λ**k P_(k+1)`, 
for which the coefficient matrices `P_i`, `i = 1, ..., k+1`, are stored in 
the 3-dimensional matrix `P`, where `P[:,:,i]` contains the `i`-th coefficient matrix `P_i` (multiplying `λ**(i-1)`). 

`P(λ)` can also be specified as a matrix or scalar of elements of the `Polynomial` type 
provided by the [Polynomials](https://github.com/JuliaMath/Polynomials.jl) package.   

The computed structured linearization satisfies:

(1) `A-λE` is regular;

(2) `rank[B-λF A-λE] = n` (strong controllability) if `minimal = true` or `contr = true`, in which case 
the right Kronecker structure is preserved;

(3) `rank[A-λE; C-λG] = n` (strong observability)  if `minimal = true` or `obs = true`, in which case 
the left Kronecker structure is preserved.

If conditions (1)-(3) are satisfied, the linearization is called `strongly minimal`, the resulting order `n`
is the least achievable order and `S(λ)` preserves the pole-zero structure (finite and infinite) and the 
left and right Kronecker structures of `R(λ)`. 

The keyword arguments `atol` and `rtol`, specify, respectively, the absolute and relative tolerance for the 
nonzero coefficients of the matrix `P(λ)`. The default relative tolerance is `nt*ϵ`, 
where `nt` is the size of the square matrix `P(λ)` and `ϵ` is the machine epsilon of the element type of its coefficients. 

The structured linearization is built using the methods described in [1].

[1] A. Varga, On computing the Kronecker structure of polynomial and rational matrices using Julia, 2020, 
[arXiv:2006.06825](https://arxiv.org/pdf/2006.06825).
"""
function pminv2lps(P::Union{AbstractArray{T1,3},AbstractArray{T1,2}}; 
                  contr::Bool = false, obs::Bool = false, minimal::Bool = false, 
                  fast::Bool = true, atol::Real = zero(real(T1)), 
                  rtol::Real = size(P,1)*eps(real(float(one(T1))))*iszero(atol)) where {T1}
    n = size(P,1) 
    n == size(P,2) || throw(DimensionMismatch("P(λ) must be a square polynomial matrix"))
    
    return spm2lps(P, Matrix{T1}(I,n,n), Matrix{T1}(I,n,n), zeros(T1,n,n), contr = contr, obs = obs, minimal = minimal, 
                  fast = fast, atol = atol, rtol = rtol) 
end
function pminv2lps(P::Union{AbstractVecOrMat{<:Polynomial},Polynomial,Number,AbstractVecOrMat{<:Number}}; kwargs...)
    return pminv2lps(poly2pm(P); kwargs...)
end
