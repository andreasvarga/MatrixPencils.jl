"""
    rmkstruct(N, D; fast = false, atol::Real = 0, rtol::Real = atol>0 ? 0 : n*ϵ) -> (KRInfo, iz, nfp, ip)
  
Determine the Kronecker-structure and infinite pole-zero structure information of the rational matrix 
`R(λ) := N(λ)./D(λ)` and return an `KRInfo` object, the multiplicities of infinite zeros in `iz`, 
the number of finite poles in `nfp` and the multiplicities of infinite poles in `ip`. 

The numerator `N(λ)` is a polynomial matrix of the form `N(λ) = N_1 + λ N_2 + ... + λ**k N_(k+1)`,   
for which the coefficient matrices `N_i`, `i = 1, ..., k+1` are stored in the 3-dimensional matrix `N`, 
where `N[:,:,i]` contains the `i`-th coefficient matrix `N_i` (multiplying `λ**(i-1)`). 

The denominator `D(λ)` is a polynomial matrix of the form `D(λ) = D_1 + λ D_2 + ... + λ**l D_(l+1)`,  
for which the coefficient matrices `D_i`, `i = 1, ..., l+1`, are stored in the 3-dimensional matrix `D`, 
where `D[:,:,i]` contain the `i`-th coefficient matrix `D_i` (multiplying `λ**(i-1)`). 

Alternatively, `N(λ)` and `D(λ)` can be specified as matrices of elements of the `Polynomial` type 
provided by the [Polynomials](https://github.com/JuliaMath/Polynomials.jl) package. 

The information on the Kronecker-structure of the rational matrix `R(λ)` consists of 
the right Kronecker indices `rki`, left Kronecker indices `lki`, the number of finite zeros `nf` 
and the normal rank `nrank` and can be obtained from `KRInfo` as 
`KRInfo.rki`, `KRInfo.lki`, `KRInfo.nf` and `KRInfo.nrank`, respectively. 
For more details, see  [`KRInfo`](@ref). Additionally, `KRInfo.id` contains the 
infinite elementary divisors of the underlying system matrix pencil `S(λ)` used as linearization.

The Kronecker-structure information of the rational matrix `R(λ)` is obtained from the 
Kronecker-structure information of the underlying linearization using the results of [1] and [2]. 
The determination of the Kronecker-structure information is performed by building an irreducible 
descriptor system realization `(A-λE,B,C,D)` with `A-λE` a regular pencil of order `n`, 
satisfying `R(λ) = C*inv(λE-A)*B+D` and reducing the structured system matrix pencil 
`S(λ) = [A-λE B; C D]` to an appropriate Kronecker-like form (KLF) which exhibits 
the number of its finite eigenvalues (also the finite zeros of `R(λ)`), the multiplicities of the 
infinite eigenvalues (in excess with one to the multiplicities of infinite zeros of `R(λ)`), 
the left and right Kronecker indices (also of `R(λ)`) and the normal rank 
(in excess with `n` to the normal rank of `R(λ)`).
The Kronecker-structure information and pole-zero stucture information on `R(λ)` are recovered from 
the Kronecker-structure information of `S(λ)` and `A-λE`, respectively, using the results of [1] and [2].

The right Kronecker indices are provided in the integer vector `rki`. 
The number of elements of `rki` is the dimension of the right nullspace of the rational matrix `R(λ)` 
and their sum is the least degree of a right polynomial nullspace basis or the least McMillan degree
of a right rational nullspace basis. 

The left Kronecker indices are provided in the integer vector `lki`. 
The number of elements of `lki` is the dimension of the left nullspace of the rational matrix `R(λ)` 
and their sum is the least degree of a left polynomial nullspace basis or the least McMillan degree
of a left rational nullspace basis.  

The multiplicities of infinite eigenvalues of `S(λ)` are provided in the integer vector `id`, 
where each `i`-th element `id[i]` is the order of an infinite elementary divisor 
(i.e., the multiplicity of an infinite eigenvalue). To each `id[i] > 1` corresponds an infinite zero
of `R(λ)` of multiplicity `id[i]-1`. The multiplicities of the infinite zeros of `R(λ)` are returned in `iz`. 

The poles of `R(λ)` are the zeros of the regular pencil `A-λE`. The finite eigenvalues of `A-λE` are the  
finite poles of `R(λ)` and their number is returned in `nfp`. The infinite zeros of `A-λE` are the  
infinite poles of `R(λ)` and their  multiplicities are returned in `ip`. 

The irreducible descriptor system based linearization is built using the methods described in [3] in conjunction with
pencil manipulation algorithms of [4] and [5] to compute reduced order linearizations. These algorithms 
employ rank determinations based on either the use of 
rank revealing QR-decomposition with column pivoting, if `fast = true`, or the SVD-decomposition, if `fast = false`.
The rank decision based on the SVD-decomposition is generally more reliable, but the involved computational effort is higher.

The reductions to appropriate KLFs of `S(λ)` and `A-λE` are performed using orthogonal similarity transformations [5]
and involves rank decisions based  on rank revealing QR-decompositions with column pivoting, if `fast = true`, 
or, the more reliable, SVD-decompositions, if `fast = false`. 

The keyword arguments `atol`  and `rtol` specify the absolute and relative tolerances,  respectively, for the nonzero 
coefficients of `N(λ)` and `D(λ)`. 
The default relative tolerance is `n*ϵ`, where `n` is the size of the smallest dimension of `N(λ)`, and `ϵ` is the 
machine epsilon of the element type of coefficients of `N(λ)`. 

[1] G. Verghese, B. Levy, and T. Kailath, Generalized state-space systems, IEEE Trans. Automat. Control,
26:811-831 (1981).

[2] G. Verghese, Comments on ‘Properties of the system matrix of a generalized state-space system’,
Int. J. Control, Vol.31(5) (1980) 1007–1009.

[3] A. Varga, On computing the Kronecker structure of polynomial and rational matrices using Julia, 2020, 
[arXiv:2006.06825](https://arxiv.org/pdf/2006.06825).

[4] P. Van Dooreen, The generalized eigenstructure problem in linear system theory, 
IEEE Transactions on Automatic Control, vol. AC-26, pp. 111-129, 1981.

[5] A. Varga, Solving Fault Diagnosis Problems - Linear Synthesis Techniques, Springer Verlag, 2017. 
"""
function rmkstruct(N::AbstractArray{T1,3}, D::AbstractArray{T2,3}; fast::Bool = false, atol::Real = zero(real(T1)), 
                   rtol::Real = (min(size(N)...)*eps(real(float(one(T1)))))*iszero(atol)) where {T1,T2}
    A, E, B, C, D, blkdims = rm2ls(N, D; contr = true, obs = true, noseig = false, 
                                   fast = fast, atol = atol, rtol = rtol) 
    info = spkstruct(A, E, B, C, D; fast = fast, atol1 = atol, atol2 = atol, rtol = rtol)
    iz  = info.id .- 1
    iz = iz[iz .> 0]
    n = size(A,1)   
    info.nrank -= n    
    nfp = blkdims[1] 
    if n > nfp
       ii = nfp+1:n
       infop = pkstruct(view(A,ii,ii),view(E,ii,ii), fast = fast, atol1 = atol, atol2 = atol, rtol = rtol)
       ip = infop.id .- 1
       ip = ip[ip .> 0]
    else
      ip = Int[]
    end
    return info, iz, nfp, ip
end 
function rmkstruct(P::AbstractArray{T,3}; kwargs...) where {T} 
   info, iz, ip = pmkstruct(P; kwargs...)
   return info, iz, 0, ip
end
function rmkstruct(N::Union{AbstractVecOrMat{Polynomial{T1}},Polynomial{T1},Number,AbstractVecOrMat}, 
                   D::Union{AbstractVecOrMat{Polynomial{T2}},Polynomial{T2},Number,AbstractVecOrMat}; kwargs...) where {T1,T2} 
   return rmkstruct(poly2pm(N),poly2pm(D); kwargs...)
end
function rmkstruct(P::Union{AbstractVecOrMat{Polynomial{T}},Polynomial{T},Number,AbstractVecOrMat}; kwargs...) where {T} 
   info, iz, ip = pmkstruct(poly2pm(P); kwargs...)
   return info, iz, 0, ip
end
function rmkstruct(N::AbstractArray{T1,3}, 
                   D::Union{AbstractVecOrMat{Polynomial{T2}},Polynomial{T2},Number,AbstractVecOrMat}; kwargs...) where {T1,T2} 
   return rmkstruct(N,poly2pm(D); kwargs...)
end
function rmkstruct(N::Union{AbstractVecOrMat{Polynomial{T1}},Polynomial{T1},Number,AbstractVecOrMat}, 
                   D::AbstractArray{T2,3}; kwargs...) where {T1,T2} 
   return rmkstruct(poly2pm(N),D; kwargs...)
end
"""
    rmzeros(N, D; fast = false, atol::Real = 0, rtol::Real = atol>0 ? 0 : n*ϵ) -> (val, iz, KRInfo)

Return the finite and infinite zeros of the rational matrix `R(λ) := N(λ)./D(λ)` in `val`, 
the multiplicities of infinite zeros in `iz` and the information on the Kronecker-structure of 
the rational matrix `R(λ)` in the `KRInfo` object. 

The information on the Kronecker-structure of the rational matrix `R(λ)` consists of 
the right Kronecker indices `rki`, left Kronecker indices `lki`, the number of finite zeros `nf` 
and the normal rank `nrank` and can be obtained from `KRInfo` as 
`KRInfo.rki`, `KRInfo.lki`, `KRInfo.nf` and `KRInfo.nrank`, respectively. 
For more details, see  [`KRInfo`](@ref). Additionally, `KRInfo.id` contains the 
infinite elementary divisors of the underlying linearization as a structured system matrix pencil.

The Kronecker-structure information of the rational matrix `R(λ)` is obtained from the 
Kronecker-structure information of the underlying linearization using the results of [1] and [2]. 
The determination of the Kronecker-structure information is performed by building an irreducible 
descriptor system realization `(A-λE,B,C,D)` with `A-λE` a regular pencil of order `n`, 
satisfying `R(λ) = C*inv(λE-A)*B+D` and reducing the structured system matrix pencil 
`S(λ) = [A-λE B; C D]` to an appropriate Kronecker-like form (KLF) which exhibits 
the number of its finite eigenvalues (also the finite zeros of `R(λ)`), the multiplicities of the 
infinite eigenvalues (in excess with one to the multiplicities of infinite zeros of `R(λ)`), 
the left and right Kronecker indices (also of `R(λ)`) and the normal rank 
(in excess with `n` to the normal rank of `R(λ)`).

The numerator `N(λ)` is a polynomial matrix of the form `N(λ) = N_1 + λ N_2 + ... + λ**k N_(k+1)`,   
for which the coefficient matrices `N_i`, `i = 1, ..., k+1` are stored in the 3-dimensional matrix `N`, 
where `N[:,:,i]` contains the `i`-th coefficient matrix `N_i` (multiplying `λ**(i-1)`). 

The denominator `D(λ)` is a polynomial matrix of the form `D(λ) = D_1 + λ D_2 + ... + λ**l D_(l+1)`,  
for which the coefficient matrices `D_i`, `i = 1, ..., l+1`, are stored in the 3-dimensional matrix `D`, 
where `D[:,:,i]` contain the `i`-th coefficient matrix `D_i` (multiplying `λ**(i-1)`). 

Alternatively, `N(λ)` and `D(λ)` can be specified as matrices of elements of the `Polynomial` type 
provided by the [Polynomials](https://github.com/JuliaMath/Polynomials.jl) package. 

The irreducible descriptor system based linearization is built using the methods described in [3] in conjunction with
pencil manipulation algorithms of [4] and [5] to compute reduced order linearizations. These algorithms 
employ rank determinations based on either the use of 
rank revealing QR-decomposition with column pivoting, if `fast = true`, or the SVD-decomposition, if `fast = false`.
The rank decision based on the SVD-decomposition is generally more reliable, but the involved computational effort is higher.

The reduction to the appropriate KLF of `S(λ)` is performed using orthogonal similarity transformations [5]
and involves rank decisions based on rank revealing QR-decompositions with column pivoting, if `fast = true`, 
or, the more reliable, SVD-decompositions, if `fast = false`. 

The keyword arguments `atol`  and `rtol` specify the absolute and relative tolerances,  respectively, for the nonzero 
coefficients of `N(λ)` and `D(λ)`. 
The default relative tolerance is `n*ϵ`, where `n` is the size of the smallest dimension of `N(λ)`, and `ϵ` is the 
machine epsilon of the element type of coefficients of `N(λ)`. 

[1] G. Verghese, B. Levy, and T. Kailath, Generalized state-space systems, IEEE Trans. Automat. Control,
26:811-831 (1981).

[2] G. Verghese, Comments on ‘Properties of the system matrix of a generalized state-space system’,
Int. J. Control, Vol.31(5) (1980) 1007–1009.

[3] A. Varga, On computing the Kronecker structure of polynomial and rational matrices using Julia, 2020, 
[arXiv:2006.06825](https://arxiv.org/pdf/2006.06825).

[4] P. Van Dooreen, The generalized eigenstructure problem in linear system theory, 
IEEE Transactions on Automatic Control, vol. AC-26, pp. 111-129, 1981.

[5] A. Varga, Solving Fault Diagnosis Problems - Linear Synthesis Techniques, Springer Verlag, 2017. 
"""
function rmzeros(N::AbstractArray{T1,3}, D::AbstractArray{T2,3}; 
                 fast = false, atol::Real = zero(real(T1)), 
                 rtol::Real = (min(size(N)...)*eps(real(float(one(T1)))))*iszero(atol)) where {T1,T2} 
    A, E, B, C, D, blkdims = rm2ls(N, D; contr = true, obs = true, noseig = false, 
                                   fast = fast, atol = atol, rtol = rtol) 
    val, iz, info = spzeros(A, E, B, C, D; fast = fast, atol1 = atol, atol2 = atol, rtol = rtol)
    info.nrank -= size(A,1)
    return val, iz, info
end
function rmzeros(N::AbstractArray{T,3}; kwargs...) where {T} 
   return pmzeros2(N; kwargs...)
end
function rmzeros(N::Union{AbstractVecOrMat{Polynomial{T1}},Polynomial{T1},Number,AbstractVecOrMat}, 
                 D::Union{AbstractVecOrMat{Polynomial{T2}},Polynomial{T2},Number,AbstractVecOrMat}; kwargs...) where {T1,T2} 
   return rmzeros(poly2pm(N),poly2pm(D); kwargs...)
end
function rmzeros(N::Union{AbstractVecOrMat{Polynomial{T}},Polynomial{T},Number,AbstractVecOrMat}; kwargs...) where {T} 
   return pmzeros2(poly2pm(N); kwargs...)
end
function rmzeros(N::AbstractArray{T1,3}, 
                 D::Union{AbstractVecOrMat{Polynomial{T2}},Polynomial{T2},Number,AbstractVecOrMat}; kwargs...) where {T1,T2} 
   return rmzeros(N,poly2pm(D); kwargs...)
end
function rmzeros(N::Union{AbstractVecOrMat{Polynomial{T1}},Polynomial{T1},Number,AbstractVecOrMat}, 
                 D::AbstractArray{T2,3}; kwargs...) where {T1,T2} 
   return rmzeros(poly2pm(N),D; kwargs...)
end
"""
    rmpoles(N, D; fast = false, atol::Real = 0, rtol::Real = atol>0 ? 0 : n*ϵ) -> (val, ip, id)
   
Return the finite and infinite poles of the rational matrix `R(λ) := N(λ)./D(λ)`  in `val`, 
the multiplicities of infinite poles in `ip` and the infinite elementary divisors of the pole pencil 
of the underlying irreducible descriptor system based linearization of `R(λ)` in `id`.

The information on the finite and infinite poles of the rational matrix `R(λ)` is obtained from the 
Kronecker-structure information of the underlying linearization using the results of [1] and [2]. 
The determination of the Kronecker-structure information is performed by building an irreducible 
descriptor system realization `(A-λE,B,C,D)` with `A-λE` a regular pencil of order `n`, 
satisfying `R(λ) = C*inv(λE-A)*B+D` and reducing the regular pencil `A-λE` 
to an appropriate Kronecker-like form (KLF) which exhibits 
the its finite eigenvalues (also the finite poles of `R(λ)`) and the multiplicities of the 
infinite eigenvalues (in excess with one to the multiplicities of infinite poles of `R(λ)`).
The multiplicities of the infinite zeros of `A-λE` and of infinite poles of `R(λ)` are the same [2] 
and are returned in `ip`. The number of infinite poles in `val` is equal to the sum of multiplicites in `ip`. 
The multiplicities of the infinite eigenvalues of `A-λE` are returned in `id`.

The numerator `N(λ)` is a polynomial matrix of the form `N(λ) = N_1 + λ N_2 + ... + λ**k N_(k+1)`,   
for which the coefficient matrices `N_i`, `i = 1, ..., k+1` are stored in the 3-dimensional matrix `N`, 
where `N[:,:,i]` contains the `i`-th coefficient matrix `N_i` (multiplying `λ**(i-1)`). 

The denominator `D(λ)` is a polynomial matrix of the form `D(λ) = D_1 + λ D_2 + ... + λ**l D_(l+1)`,  
for which the coefficient matrices `D_i`, `i = 1, ..., l+1`, are stored in the 3-dimensional matrix `D`, 
where `D[:,:,i]` contain the `i`-th coefficient matrix `D_i` (multiplying `λ**(i-1)`). 

Alternatively, `N(λ)` and `D(λ)` can be specified as matrices of elements of the `Polynomial` type 
provided by the [Polynomials](https://github.com/JuliaMath/Polynomials.jl) package. 

The irreducible descriptor system based linearization is built using the methods described in [3] in conjunction with
pencil manipulation algorithms of [4] and [5] to compute reduced order linearizations. These algorithms 
employ rank determinations based on either the use of 
rank revealing QR-decomposition with column pivoting, if `fast = true`, or the SVD-decomposition, if `fast = false`.
The rank decision based on the SVD-decomposition is generally more reliable, but the involved computational effort is higher.

The reduction of `A-λE` to the KLF is performed using orthonal similarity transformations and involves rank decisions 
based on rank revealing QR-decompositions with column pivoting, 
if `fast = true`, or, the more reliable, SVD-decompositions, if `fast = false`. For efficiency purposes, the reduction is only
partially performed, without accumulating the performed orthogonal transformations.

The keyword arguments `atol`  and `rtol` specify the absolute and relative tolerances for the nonzero 
coefficients of `N(λ)` and `D(λ)`,  respectively. 
The default relative tolerance is `n*ϵ`, where `n` is the size of the smallest dimension of `N(λ)`, and `ϵ` is the 
machine epsilon of the element type of coefficients of `N(λ)`. 

[1] G. Verghese, B. Levy, and T. Kailath, Generalized state-space systems, IEEE Trans. Automat. Control,
26:811-831 (1981).

[2] G. Verghese, Comments on ‘Properties of the system matrix of a generalized state-space system’,
Int. J. Control, Vol.31(5) (1980) 1007–1009.

[3] A. Varga, On computing the Kronecker structure of polynomial and rational matrices using Julia, 2020, 
[arXiv:2006.06825](https://arxiv.org/pdf/2006.06825).

[4] P. Van Dooreen, The generalized eigenstructure problem in linear system theory, 
IEEE Transactions on Automatic Control, vol. AC-26, pp. 111-129, 1981.

[5] A. Varga, Solving Fault Diagnosis Problems - Linear Synthesis Techniques, Springer Verlag, 2017. 
"""
function rmpoles(N::AbstractArray{T1,3}, D::AbstractArray{T2,3}; 
                 fast = false, atol::Real = zero(real(T1)), 
                 rtol::Real = (min(size(N)...)*eps(real(float(one(T1)))))*iszero(atol)) where {T1,T2} 
   A, E, _, _, _, blkdims = rm2ls(N, D; contr = true, obs = true, noseig = false, 
                                  fast = fast, atol = atol, rtol = rtol) 
   n = size(A,1)                               
   nfp = blkdims[1]
   if nfp < n
      ii = nfp+1:n
      infop = pkstruct(view(A,ii,ii),view(E,ii,ii), fast = fast, atol1 = atol, atol2 = atol, rtol = rtol)
      ip = infop.id .- 1
      ip = ip[ip .> 0]
      id = infop.id
   else
      ip = Int[]
      id = Int[]
   end
   return [eigvals(A[1:nfp,1:nfp]); Inf*ones(eltype(A),sum(ip))], ip, id
end
function rmpoles(N::AbstractArray{T,3}; kwargs...) where T 
   return pmpoles2(N; kwargs...) 
end
function rmpoles(N::Union{AbstractVecOrMat{Polynomial{T1}},Polynomial{T1},Number,AbstractVecOrMat}, 
                 D::Union{AbstractVecOrMat{Polynomial{T2}},Polynomial{T2},Number,AbstractVecOrMat}; kwargs...) where {T1,T2} 
   return rmpoles(poly2pm(N),poly2pm(D); kwargs...)
end
function rmpoles(N::Union{AbstractVecOrMat{Polynomial{T}},Polynomial{T},Number,AbstractVecOrMat}; kwargs...) where {T} 
   return pmpoles2(poly2pm(N); kwargs...)
end
function rmpoles(N::AbstractArray{T1,3}, 
                 D::Union{AbstractVecOrMat{Polynomial{T2}},Polynomial{T2},Number,AbstractVecOrMat}; kwargs...) where {T1,T2} 
   return rmpoles(N,poly2pm(D); kwargs...)
end
function rmpoles(N::Union{AbstractVecOrMat{Polynomial{T1}},Polynomial{T1},Number,AbstractVecOrMat}, 
                 D::AbstractArray{T2,3}; kwargs...) where {T1,T2} 
   return rmpoles(poly2pm(N),D; kwargs...)
end
"""
    rmzeros1(N, D; fast = false, atol::Real = 0, rtol::Real = atol>0 ? 0 : n*ϵ) -> (val, iz, KRInfo)

Return the finite and infinite zeros of the rational matrix `R(λ) := N(λ)./D(λ)` in `val`, 
the multiplicities of infinite zeros in `iz` and the information on the Kronecker-structure of 
the rational matrix `R(λ)` in the `KRInfo` object. 

The information on the Kronecker-structure of the rational matrix `R(λ)` consists of 
the right Kronecker indices `rki`, left Kronecker indices `lki`, the number of finite zeros `nf` 
and the normal rank `nrank` and can be obtained from `KRInfo` as 
`KRInfo.rki`, `KRInfo.lki`, `KRInfo.nf` and `KRInfo.nrank`, respectively. 
For more details, see  [`KRInfo`](@ref). Additionally, `KRInfo.id` contains the 
infinite elementary divisors of the underlying pencil based linearization.

The Kronecker-structure information of the rational matrix `R(λ)` is obtained from the 
Kronecker-structure information of the underlying pencil based linearization using the results of [1] and [2]. 
The determination of the Kronecker-structure information is performed by building a strongly minimal 
pencil based linearization `(A-λE,B-λF,C-λG,D-λH)` with `A-λE` a regular pencil of order `n`, 
satisfying `R(λ) = (C-λG)*inv(λE-A)*(B-λF)+D-λH` and reducing the structured system matrix pencil 
`S(λ) = [A-λE B-λF; C-λG D-λH]` to an appropriate Kronecker-like form (KLF) which exhibits 
the number of its finite eigenvalues (also the finite zeros of `R(λ)`), the multiplicities of the 
infinite eigenvalues (in excess with one to the multiplicities of infinite zeros of `R(λ)`), 
the left and right Kronecker indices (also of `R(λ)`) and the normal rank 
(in excess with `n` to the normal rank of `R(λ)`).

The numerator `N(λ)` is a polynomial matrix of the form `N(λ) = N_1 + λ N_2 + ... + λ**k N_(k+1)`,   
for which the coefficient matrices `N_i`, `i = 1, ..., k+1` are stored in the 3-dimensional matrix `N`, 
where `N[:,:,i]` contains the `i`-th coefficient matrix `N_i` (multiplying `λ**(i-1)`). 

The denominator `D(λ)` is a polynomial matrix of the form `D(λ) = D_1 + λ D_2 + ... + λ**l D_(l+1)`,  
for which the coefficient matrices `D_i`, `i = 1, ..., l+1`, are stored in the 3-dimensional matrix `D`, 
where `D[:,:,i]` contain the `i`-th coefficient matrix `D_i` (multiplying `λ**(i-1)`). 

Alternatively, `N(λ)` and `D(λ)` can be specified as matrices of elements of the `Polynomial` type 
provided by the [Polynomials](https://github.com/JuliaMath/Polynomials.jl) package. 

The strongly minimal pencil based linearization is built using the methods described in [3] in conjunction with
pencil manipulation algorithms of [4] to compute reduced order linearizations. These algorithms 
employ rank determinations based on either the use of 
rank revealing QR-decomposition with column pivoting, if `fast = true`, or the SVD-decomposition, if `fast = false`.
The rank decision based on the SVD-decomposition is generally more reliable, but the involved computational effort is higher.

The reduction to the appropriate KLF of `S(λ)` is performed using orthogonal similarity transformations [5]
and involves rank decisions based on rank revealing QR-decompositions with column pivoting, if `fast = true`, 
or, the more reliable, SVD-decompositions, if `fast = false`. 

The keyword arguments `atol`  and `rtol` specify the absolute and relative tolerances,  respectively, for the nonzero 
coefficients of `N(λ)` and `D(λ)`. 
The default relative tolerance is `n*ϵ`, where `n` is the size of the smallest dimension of `N(λ)`, and `ϵ` is the 
machine epsilon of the element type of coefficients of `N(λ)`. 

[1] G. Verghese, B. Levy, and T. Kailath, Generalized state-space systems, IEEE Trans. Automat. Control,
26:811-831 (1981).

[2] G. Verghese, Comments on ‘Properties of the system matrix of a generalized state-space system’,
Int. J. Control, Vol.31(5) (1980) 1007–1009.

[3] A. Varga, On computing the Kronecker structure of polynomial and rational matrices using Julia, 2020, 
[arXiv:2006.06825](https://arxiv.org/pdf/2006.06825).

[4] F.M. Dopico, M.C. Quintana and P. Van Dooren, Linear system matrices of rational transfer functions, to appear in "Realization and Model Reduction of Dynamical Systems", 
A Festschrift to honor the 70th birthday of Thanos Antoulas", Springer-Verlag. [arXiv:1903.05016](https://arxiv.org/pdf/1903.05016.pdf)

[5] A. Varga, Solving Fault Diagnosis Problems - Linear Synthesis Techniques, Springer Verlag, 2017. 
"""
function rmzeros1(N::AbstractArray{T1,3}, D::AbstractArray{T2,3}; 
                  fast = false, atol::Real = zero(real(T1)), 
                   rtol::Real = (min(size(N)...)*eps(real(float(one(T1)))))*iszero(atol)) where {T1,T2} 
   A, E, B, F, C, G, D, H, blkdims = rm2lps(N, D; obs = true, contr = true, fast = fast, 
                                            atol = atol, rtol = rtol) 
   val, iz, info = pzeros([A B; C D], [E F; G H]; fast = fast, atol1 = atol, atol2 = atol, rtol = rtol)
   info.nrank -= size(A,1)
   return val, iz, info
end
function rmzeros1(N::AbstractArray{T,3}; kwargs...) where {T} 
   return pmzeros1(N; kwargs...)
end
function rmzeros1(N::Union{AbstractVecOrMat{Polynomial{T1}},Polynomial{T1},Number,AbstractVecOrMat}, 
                 D::Union{AbstractVecOrMat{Polynomial{T2}},Polynomial{T2},Number,AbstractVecOrMat}; kwargs...) where {T1,T2} 
   return rmzeros1(poly2pm(N),poly2pm(D); kwargs...)
end
function rmzeros1(N::Union{AbstractVecOrMat{Polynomial{T}},Polynomial{T},Number,AbstractVecOrMat}; kwargs...) where {T} 
   return pmzeros1(poly2pm(N); kwargs...)
end
function rmzeros1(N::AbstractArray{T1,3}, 
                 D::Union{AbstractVecOrMat{Polynomial{T2}},Polynomial{T2},Number,AbstractVecOrMat}; kwargs...) where {T1,T2} 
   return rmzeros1(N,poly2pm(D); kwargs...)
end
function rmzeros1(N::Union{AbstractVecOrMat{Polynomial{T1}},Polynomial{T1},Number,AbstractVecOrMat}, 
                 D::AbstractArray{T2,3}; kwargs...) where {T1,T2} 
   return rmzeros1(poly2pm(N),D; kwargs...)
end
"""
    rmpoles1(N, D;  fast = false, atol::Real = 0, rtol::Real = atol>0 ? 0 : n*ϵ) -> (val, ip, id)
   
Return the finite and infinite poles of the rational matrix `R(λ) := N(λ)./D(λ)`  in `val`, 
the multiplicities of infinite poles in `ip` and the infinite elementary divisors of the pole pencil 
of the underlying strongly irreducible pencil based linearization of  `R(λ)` in `id`.

The information on the finite and infinite poles of the rational matrix `R(λ)` is obtained from the 
pole-structure information of the underlying pencil based linearization using the results of [1] and [2]. 
The determination of the pole-structure information is performed by building a strongly minimal 
pencil based linearization `(A-λE,B-λF,C-λG,D-λH)` with `A-λE` a regular pencil of order `n`, 
satisfying `R(λ) = (C-λG)*inv(λE-A)*(B-λF)+D-λH` and reducing the structured system matrix pencil 
`Sp(λ) = [A-λE -λF; -λG -λH]` to an appropriate Kronecker-like form (KLF) which exhibits 
the number of its finite eigenvalues (also the finite poles of `R(λ)`) and the multiplicities of the 
infinite eigenvalues (in excess with one to the multiplicities of infinite poles of `R(λ)`).
The multiplicities of the infinite zeros of `Sp(λ)` and of infinite poles of `R(λ)` are the same [2] 
and are returned in `ip`. The number of infinite poles in `val` is equal to the sum of multiplicites in `ip`. 
The multiplicities of the infinite eigenvalues of `Sp(λ)` are returned in `id`.

The numerator `N(λ)` is a polynomial matrix of the form `N(λ) = N_1 + λ N_2 + ... + λ**k N_(k+1)`,   
for which the coefficient matrices `N_i`, `i = 1, ..., k+1` are stored in the 3-dimensional matrix `N`, 
where `N[:,:,i]` contains the `i`-th coefficient matrix `N_i` (multiplying `λ**(i-1)`). 

The denominator `D(λ)` is a polynomial matrix of the form `D(λ) = D_1 + λ D_2 + ... + λ**l D_(l+1)`,  
for which the coefficient matrices `D_i`, `i = 1, ..., l+1`, are stored in the 3-dimensional matrix `D`, 
where `D[:,:,i]` contain the `i`-th coefficient matrix `D_i` (multiplying `λ**(i-1)`). 

Alternatively, `N(λ)` and `D(λ)` can be specified as matrices of elements of the `Polynomial` type 
provided by the [Polynomials](https://github.com/JuliaMath/Polynomials.jl) package. 

The strongly minimal pencil based linearization is built using the methods described in [3] in conjunction with
pencil manipulation algorithms of [4] to compute reduced order linearizations. These algorithms 
employ rank determinations based on either the use of 
rank revealing QR-decomposition with column pivoting, if `fast = true`, or the SVD-decomposition, if `fast = false`.
The rank decision based on the SVD-decomposition is generally more reliable, but the involved computational effort is higher.

The reduction of `Sp(λ)` to the KLF is performed using orthonal similarity transformations [5] and involves rank decisions 
based on rank revealing QR-decompositions with column pivoting, 
if `fast = true`, or, the more reliable, SVD-decompositions, if `fast = false`. For efficiency purposes, the reduction is only
partially performed, without accumulating the performed orthogonal transformations.

The keyword arguments `atol`  and `rtol` specify the absolute and relative tolerances,  respectively, for the nonzero 
coefficients of `N(λ)` and `D(λ)`. 
The default relative tolerance is `n*ϵ`, where `n` is the size of the smallest dimension of `N(λ)`, and `ϵ` is the 
machine epsilon of the element type of coefficients of `N(λ)`. 

[1] G. Verghese, B. Levy, and T. Kailath, Generalized state-space systems, IEEE Trans. Automat. Control,
26:811-831 (1981).

[2] G. Verghese, Comments on ‘Properties of the system matrix of a generalized state-space system’,
Int. J. Control, Vol.31(5) (1980) 1007–1009.

[3] A. Varga, On computing the Kronecker structure of polynomial and rational matrices using Julia, 2020, 
[arXiv:2006.06825](https://arxiv.org/pdf/2006.06825).

[4] F.M. Dopico, M.C. Quintana and P. Van Dooren, Linear system matrices of rational transfer functions, to appear in "Realization and Model Reduction of Dynamical Systems", 
A Festschrift to honor the 70th birthday of Thanos Antoulas", Springer-Verlag. [arXiv:1903.05016](https://arxiv.org/pdf/1903.05016.pdf)

[5] A. Varga, Solving Fault Diagnosis Problems - Linear Synthesis Techniques, Springer Verlag, 2017. 
"""
function rmpoles1(N::AbstractArray{T1,3}, D::AbstractArray{T2,3}; 
                  fast = false, atol::Real = zero(real(T1)), 
                  rtol::Real = (min(size(N)...)*eps(real(float(one(T1)))))*iszero(atol)) where {T1,T2} 
   A, E, _, F, _, G, _, H, blkdims = rm2lps(N,D; obs = true, contr = true, fast = fast, atol = atol, rtol = rtol) 
   n = size(A,1)                               
   nfp = blkdims[1]
   if nfp < n
      p, m = size(H)
      ii = nfp+1:n
      ni = n-nfp
      T = promote_type(T1,T2)
      M = [A[ii,ii] zeros(T,ni,p+m); zeros(T,p,ni+m) I; zeros(T,m,ni) I zeros(T,m,p)]
      N = [E[ii,ii] F[ii,:] zeros(T,ni,p); G[:,ii] H zeros(T,p,p); zeros(T,m,ni+m+p)]
      infop = pkstruct(M, N, fast = fast, atol1 = atol, atol2 = atol, rtol = rtol)
      ip = infop.id .- 1
      ip = ip[ip .> 0]
      id = infop.id
   else
      ip = Int[]
      id = Int[]
   end
   return [eigvals(A[1:nfp,1:nfp]); Inf*ones(eltype(A),sum(ip))], ip, id   
   # p, m = size(D)
   # n = size(A,1)     
   # T = eltype(A)  
   # M = [A zeros(T,n,p+m); zeros(T,p,n+m) I; zeros(T,m,n) I zeros(T,m,p)]
   # N = [E F zeros(T,n,p); G H zeros(T,p,p); zeros(T,m,n+m+p)]
   # val, iz, info = pzeros(M, N; fast = fast, atol1 = atol, atol2 = atol, rtol = rtol)
   # return val, iz, info.id
end
function rmpoles1(N::AbstractArray{T,3}; kwargs...) where T 
   return pmpoles1(N; kwargs...) 
end
function rmpoles1(N::Union{AbstractVecOrMat{Polynomial{T1}},Polynomial{T1},Number,AbstractVecOrMat}, 
                 D::Union{AbstractVecOrMat{Polynomial{T2}},Polynomial{T2},Number,AbstractVecOrMat}; kwargs...) where {T1,T2} 
   return rmpoles1(poly2pm(N),poly2pm(D); kwargs...)
end
function rmpoles1(N::Union{AbstractVecOrMat{Polynomial{T}},Polynomial{T},Number,AbstractVecOrMat}; kwargs...) where {T} 
   return pmpoles1(poly2pm(N); kwargs...)
end
function rmpoles1(N::AbstractArray{T1,3}, 
                 D::Union{AbstractVecOrMat{Polynomial{T2}},Polynomial{T2},Number,AbstractVecOrMat}; kwargs...) where {T1,T2} 
   return rmpoles1(N,poly2pm(D); kwargs...)
end
function rmpoles1(N::Union{AbstractVecOrMat{Polynomial{T1}},Polynomial{T1},Number,AbstractVecOrMat}, 
                 D::AbstractArray{T2,3}; kwargs...) where {T1,T2} 
   return rmpoles1(poly2pm(N),D; kwargs...)
end
"""
       rmrank(N, D; fastrank = true, atol = 0, rtol = atol > 0 ? 0 : n*ϵ) 
   
Determine the normal rank of the rational matrix `R(λ) := N(λ)./D(λ)`.

If `fastrank = true`, the rank is evaluated by counting how many singular values of `R(γ)` have magnitude greater 
than `max(atol, rtol*σ₁)`, where `σ₁` is the largest singular value of `R(γ)` and `γ` is a randomly generated 
complex value of magnitude equal to one. 
If `fastrank = false`, first a structured linearization of `R(λ)` is built in the form of a descriptor system matrix 
`S(λ) = [A-λE B; C D]` with `A-λE` an order `n` regular subpencil, and then its normal rank `nr` is determined. 
The normal rank of `R(λ)` is `nr - n`.   
   
The numerator `N(λ)` is a polynomial matrix of the form `N(λ) = N_1 + λ N_2 + ... + λ**k N_(k+1)`,   
for which the coefficient matrices `N_i`, `i = 1, ..., k+1` are stored in the 3-dimensional matrix `N`, 
where `N[:,:,i]` contains the `i`-th coefficient matrix `N_i` (multiplying `λ**(i-1)`). 

The denominator `D(λ)` is a polynomial matrix of the form `D(λ) = D_1 + λ D_2 + ... + λ**l D_(l+1)`,  
for which the coefficient matrices `D_i`, `i = 1, ..., l+1`, are stored in the 3-dimensional matrix `D`, 
where `D[:,:,i]` contain the `i`-th coefficient matrix `D_i` (multiplying `λ**(i-1)`). 

Alternatively, `N(λ)` and `D(λ)` can be specified as matrices of elements of the `Polynomial` type 
provided by the [Polynomials](https://github.com/JuliaMath/Polynomials.jl) package. 

The irreducible descriptor system based linearization is built using the methods described in [1] in conjunction with
pencil manipulation algorithms of [2] and [3] to compute reduced order linearizations. These algorithms 
employ rank determinations based on either the use of 
rank revealing QR-decomposition with column pivoting, if `fast = true`, or the SVD-decomposition, if `fast = false`.
The rank decision based on the SVD-decomposition is generally more reliable, but the involved computational effort is higher.

The reduction to the appropriate KLF of `S(λ)` is performed using orthogonal similarity transformations [3]
and involves rank decisions based on rank revealing SVD-decompositions. 

The keyword arguments `atol`  and `rtol` specify the absolute and relative tolerances for the nonzero 
coefficients of `N(λ)` and `D(λ)`,  respectively. 
The default relative tolerance is `n*ϵ`, where `n` is the size of the smallest dimension of `N(λ)`, and `ϵ` is the 
machine epsilon of the element type of coefficients of `N(λ)`. 

[1] A. Varga, On computing the Kronecker structure of polynomial and rational matrices using Julia, 2020, 
[arXiv:2006.06825](https://arxiv.org/pdf/2006.06825).

[2] P. Van Dooreen, The generalized eigenstructure problem in linear system theory, 
IEEE Transactions on Automatic Control, vol. AC-26, pp. 111-129, 1981.

[3] A. Varga, Solving Fault Diagnosis Problems - Linear Synthesis Techniques, Springer Verlag, 2017. 
"""
function rmrank(N::AbstractArray{T1,3}, D::AbstractArray{T2,3}; fastrank::Bool = true, atol::Real = zero(real(T1)), 
                rtol::Real = (min(size(N)...)*eps(real(float(one(T1)))))*iszero(atol)) where {T1,T2} 
   if fastrank
       return rank(rmeval(N,D,exp(rand()*im)), atol = atol, rtol = rtol)
   else
      A, E, B, C, D, blkdims = rm2ls(N, D; fast = false, atol = atol, rtol = rtol) 
      r = sprank(A, E, B, C, D, fastrank = false, atol1 = atol, atol2 = atol, rtol = rtol)
      return r-size(A,1)
   end
end
function rmrank(N::AbstractArray{T,3}; kwargs...) where T
    pmrank(N; kwargs...)
end
function rmrank(N::Union{AbstractVecOrMat{Polynomial{T1}},Polynomial{T1},Number,AbstractVecOrMat}, 
                D::Union{AbstractVecOrMat{Polynomial{T2}},Polynomial{T2},Number,AbstractVecOrMat}; kwargs...) where {T1,T2} 
   return rmrank(poly2pm(N),poly2pm(D); kwargs...)
end
function rmrank(N::Union{AbstractVecOrMat{Polynomial{T}},Polynomial{T},Number,AbstractVecOrMat}; kwargs...) where {T}
    pmrank(poly2pm(N); kwargs...)
end
function rmrank(N::AbstractArray{T1,3}, 
                D::Union{AbstractVecOrMat{Polynomial{T2}},Polynomial{T2},Number,AbstractVecOrMat}; kwargs...) where {T1,T2} 
   return rmrank(N,poly2pm(D); kwargs...)
end
function rmrank(N::Union{AbstractVecOrMat{Polynomial{T1}},Polynomial{T1},Number,AbstractVecOrMat}, 
                D::AbstractArray{T2,3}; kwargs...) where {T1,T2} 
   return rmrank(poly2pm(N),D; kwargs...)
end




