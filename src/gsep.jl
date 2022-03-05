"""
    fihess(A, E; fast = true, finite_infinite = false, atol1 = 0, atol2 = 0, rtol, withQ = true, withZ = true) -> (At, Et, Q, Z, ν, blkdims)

Reduce the regular matrix pencil `A - λE` to an equivalent form `At - λEt = Q'*(A - λE)*Z` using 
orthogonal or unitary transformation matrices `Q` and `Z` such that the transformed matrices `At` and `Et` are in one of the
following block upper-triangular forms:

(1) if `finite_infinite = false`, then
 
                   | Ai-λEi |   *     |
        At - λEt = |--------|---------|, 
                   |    O   | Af-λEf  |
 
where the `ni x ni` subpencil `Ai-λEi` contains the infinite elementary divisors and 
the `nf x nf` subpencil `Af-λEf` contains the finite eigenvalues of the pencil `A-λE`.

The subpencil `Ai-λEi` is in a staircase form, with `Ai` nonsingular and upper triangular and `Ei` nilpotent and upper triangular. 
The `nb`-dimensional vector `ν` contains the dimensions of the diagonal blocks
of the staircase form  `Ai-λEi` such that `i`-th block has dimensions `ν[i] x ν[i]`. 
The difference `ν[i]-ν[i+1]` for `i = 1, 2, ..., nb` is the number of infinite elementary divisors of degree `i` 
(with `ν[nb+1] = 0`).

The subpencil `Af-λEf` is with `Af` in an upper Hessenberg form and `Ef` nonsingular and upper triangular. 

The dimensions of the diagonal blocks are returned in `blkdims = (ni, nf)`.   

(2) if `finite_infinite = true`, then
 
                   | Af-λEf |   *    |
        At - λEt = |--------|--------|, 
                   |   O    | Ai-λEi |
 
where the `nf x nf` subpencil `Af-λEf`, with `Ef` nonsingular and upper triangular, 
contains the finite eigenvalues of the pencil `A-λE` and the `ni x ni` subpencil `Ai-λEi` 
contains the infinite elementary divisors.

The subpencil `Af-λEf` is with `Af` in an upper Hessenberg form and `Ef` nonsingular and upper triangular.  

The subpencil `Ai-λEi` is in a staircase form, with `Ai` nonsingular and upper triangular  and `Ei` nilpotent and upper triangular. 
The `nb`-dimensional vectors `ν` contains the dimensions of the diagonal blocks
of the staircase form `Ai-λEi` such that `i`-th block has dimensions `ν[i] x ν[i]`. 
The difference `ν[nb-j+1]-ν[nb-j]` for `j = 1, 2, ..., nb` is the number of infinite elementary 
divisors of degree `j` (with `ν[0] = 0`).

The dimensions of the diagonal blocks are returned in `blkdims = (nf, ni)`.   

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of `A`, the absolute tolerance for the nonzero elements of `E`,  and the relative tolerance 
for the nonzero elements of `A` and `E`. 

The reduction is performed using rank decisions based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.

The performed left orthogonal or unitary transformations are accumulated in the matrix `Q` if `withQ = true`. 
Otherwise, `Q` is set to `nothing`.   
The performed right orthogonal or unitary transformations are accumulated in the matrix `Z` if `withZ = true`. 
Otherwise, `Z` is set to `nothing`.  
"""
function fihess(A::AbstractMatrix, E::AbstractMatrix; 
                   fast::Bool = true, finite_infinite::Bool = false, 
                   atol1::Real = zero(real(eltype(A))), atol2::Real = zero(real(eltype(E))), 
                   rtol::Real = (size(A,1)*eps(real(float(one(eltype(A))))))*iszero(min(atol1,atol2)), 
                   withQ::Bool = true, withZ::Bool = true)

    n = LinearAlgebra.checksquare(A)
    n == LinearAlgebra.checksquare(E) || throw(DimensionMismatch("A and E must have the same dimensions"))          
    T = promote_type(eltype(A), eltype(E))
    T <: BlasFloat || (T = promote_type(Float64,T))

    A1 = copy_oftype(A,T)   
    E1 = copy_oftype(E,T)

    withQ ? (Q = Matrix{T}(I,n,n)) : (Q = nothing)
    withZ ? (Z = Matrix{T}(I,n,n)) : (Z = nothing)

    ν, blkdims = fisplit!(A1, E1, Q, Z, missing, missing; 
                         fast  = fast, finite_infinite = finite_infinite, 
                         atol1 = atol1, atol2 = atol2, rtol = rtol, withQ = withQ, withZ = withZ)
    (ilo, ihi) = finite_infinite ? (1, blkdims[1]) : (blkdims[1]+1, n)
    ilo > ihi || gghrd!(withQ ? 'V' : 'N', withZ ? 'V' : 'N', ilo, ihi, A1, E1, Q, Z)
   
    return A1, E1, Q, Z, ν, blkdims                                             
end
"""
    fischur(A, E; fast = true, finite_infinite = false, atol1 = 0, atol2 = 0, rtol, withQ = true, withZ = true) -> (At, Et, Q, Z, ν, blkdims)

Reduce the regular matrix pencil `A - λE` to an equivalent form `At - λEt = Q'*(A - λE)*Z` using 
orthogonal or unitary transformation matrices `Q` and `Z` such that the transformed matrices `At` and `Et` are in one of the
following block upper-triangular forms:

(1) if `finite_infinite = false`, then
 
                   | Ai-λEi |   *     |
        At - λEt = |--------|---------|, 
                   |    O   | Af-λEf  |
 
where the `ni x ni` subpencil `Ai-λEi` contains the infinite elementary divisors and 
the `nf x nf` subpencil `Af-λEf` contains the finite eigenvalues of the pencil `A-λE`.

The subpencil `Ai-λEi` is in a staircase form, with `Ai` nonsingular and upper triangular and `Ei` nilpotent and upper triangular. 
The `nb`-dimensional vector `ν` contains the dimensions of the diagonal blocks
of the staircase form  `Ai-λEi` such that `i`-th block has dimensions `ν[i] x ν[i]`. 
The difference `ν[i]-ν[i+1]` for `i = 1, 2, ..., nb` is the number of infinite elementary divisors of degree `i` 
(with `ν[nb+1] = 0`).

The subpencil `Af-λEf` is with the pair `(Af,Ef)` in a generalized Schur form, with `Af` (quasi) upper triangular and 
`Ef` nonsingular and upper triangular.  

The dimensions of the diagonal blocks are returned in `blkdims = (ni, nf)`.   

(2) if `finite_infinite = true`, then
 
                   | Af-λEf |   *    |
        At - λEt = |--------|--------|, 
                   |   O    | Ai-λEi |
 
where the `nf x nf` subpencil `Af-λEf`, with `Ef` nonsingular and upper triangular, 
contains the finite eigenvalues of the pencil `A-λE` and the `ni x ni` subpencil `Ai-λEi` 
contains the infinite elementary divisors.

The subpencil `Af-λEf` is with the pair `(Af,Ef)` in a generalized Schur form, with `Af` (quasi) upper triangular and 
`Ef` nonsingular and upper triangular.  

The subpencil `Ai-λEi` is in a staircase form, with `Ai` nonsingular and upper triangular 
and `Ei` nilpotent and upper triangular. 
The `nb`-dimensional vectors `ν` contains the dimensions of the diagonal blocks
of the staircase form `Ai-λEi` such that `i`-th block has dimensions `ν[i] x ν[i]`. 
The difference `ν[nb-j+1]-ν[nb-j]` for `j = 1, 2, ..., nb` is the number of infinite elementary 
divisors of degree `j` (with `ν[0] = 0`).

The dimensions of the diagonal blocks are returned in `blkdims = (nf, ni)`.   

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of `A`, the absolute tolerance for the nonzero elements of `E`,  and the relative tolerance 
for the nonzero elements of `A` and `E`. 

The reduction is performed using rank decisions based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.

The performed left orthogonal or unitary transformations are accumulated in the matrix `Q` if `withQ = true`. 
Otherwise, `Q` is set to `nothing`.   
The performed right orthogonal or unitary transformations are accumulated in the matrix `Z` if `withZ = true`. 
Otherwise, `Z` is set to `nothing`.  
"""
function fischur(A::AbstractMatrix, E::AbstractMatrix; 
                      fast::Bool = true, finite_infinite::Bool = false, 
                      atol1::Real = zero(real(eltype(A))), atol2::Real = zero(real(eltype(E))), 
                      rtol::Real = (size(A,1)*eps(real(float(one(eltype(A))))))*iszero(min(atol1,atol2)), 
                      withQ::Bool = true, withZ::Bool = true)

    n = LinearAlgebra.checksquare(A)
    n == LinearAlgebra.checksquare(E) || throw(DimensionMismatch("A and E must have the same dimensions"))          
    T = promote_type(eltype(A), eltype(E))
    T <: BlasFloat || (T = promote_type(Float64,T))

    A1 = copy_oftype(A,T)   
    E1 = copy_oftype(E,T)

    withQ ? (Q = Matrix{T}(I,n,n)) : (Q = nothing)
    withZ ? (Z = Matrix{T}(I,n,n)) : (Z = nothing)

    ν, blkdims = fisplit!(A1, E1, Q, Z, missing, missing; 
                         fast  = fast, finite_infinite = finite_infinite, 
                         atol1 = atol1, atol2 = atol2, rtol = rtol, withQ = withQ, withZ = withZ)
    (ilo, ihi) = finite_infinite ? (1, blkdims[1]) : (blkdims[1]+1, n)
    if ilo < ihi 
        compq = withQ ? 'V' : 'N'                    
        compz = withZ ? 'V' : 'N'                    
        gghrd!(compq, compz, ilo, ihi, A1, E1, Q, Z)
        hgeqz!(compq, compz, ilo, ihi, A1, E1, Q, Z)
    end
   
    return A1, E1, Q, Z, ν, blkdims                                             
end
"""
    fischursep(A, E; smarg, disc = false, fast = true, finite_infinite = false, stable_unstable = false, atol1 = 0, atol2 = 0, rtol, withQ = true, withZ = true) -> (At, Et, Q, Z, ν, blkdims)

Reduce the regular matrix pencil `A - λE` to an equivalent block upper triangular form `At - λEt = Q'*(A - λE)*Z` 
using orthogonal or unitary transformation matrices `Q` and `Z` such that the transformed matrices `At` and `Et` 
have separated infinite, stable and unstable eigenvalues with respect to a stability domain `Cs` defined by the stability margin parameter `smarg` and 
the stability type parameter `disc`. If `disc = false`, `Cs` is the set of complex numbers with real parts less than `smarg`, 
while if `disc = true`, `Cs` is the set of complex numbers with moduli less than `smarg` (i.e., the interior of a disc 
of radius `smarg` centered in the origin). If `smarg = missing`, the default value used is `smarg = 0`, if  `disc = false`,
and `smarg = 1`, if `disc = true`.

The pencil `At - λEt` results in one of the following block upper-triangular forms:

(1) if `finite_infinite = false`, then
 
                   | Ai-λEi   *      *    |
        At - λEt = |    O   A1-λE1   *    |
                   |    0     0    A2-λE2 |
 
where the `ni x ni` subpencil `Ai-λEi` contains the infinite elementary divisors, 
the `n1 x n1` subpencil `A1-λE1` is with the pair `(A1,E1)` in a generalized Schur form, and the 
`n2 x n2` subpencil `A2-λE2` is with the pair `(A2,E2)` in a generalized Schur form. 
The pencil `A1-λE1` has unstable finite eigenvalues and `A2-λE2` has stable finite eigenvalues if `stable_unstable = false`,
while `A1-λE1` has stable finite eigenvalues and `A2-λE2` has unstable finite eigenvalues if `stable_unstable = true`.

The subpencil `Ai-λEi` is in a staircase form, with `Ai` nonsingular and upper triangular and `Ei` nilpotent and upper triangular. 
The `nb`-dimensional vector `ν` contains the dimensions of the diagonal blocks
of the staircase form  `Ai-λEi` such that `i`-th block has dimensions `ν[i] x ν[i]`. 
The difference `ν[i]-ν[i+1]` for `i = 1, 2, ..., nb` is the number of infinite elementary divisors of degree `i` 
(with `ν[nb+1] = 0`).

The dimensions of the diagonal blocks are returned in `blkdims = (ni, n1, n2)`.   

(2) if `finite_infinite = true`, then
 
                   | A1-λE1   *      *    |
        At - λEt = |    O   A2-λE2   *    |
                   |    0     0    Ai-λEi |
 
where the `ni x ni` subpencil `Ai-λEi` contains the infinite elementary divisors, 
the `n1 x n1` subpencil `A1-λE1` is with the pair `(A1,E1)` in a generalized Schur form, and the 
`n2 x n2` subpencil `A2-λE2` is with the pair `(A2,E2)` in a generalized Schur form. 
The pencil `A1-λE1` has unstable finite eigenvalues and `A2-λE2` has stable finite eigenvalues if `stable_unstable = false`,
while `A1-λE1` has stable finite eigenvalues and `A2-λE2` has unstable finite eigenvalues if `stable_unstable = true`.

The subpencil `Ai-λEi` is in a staircase form, with `Ai` nonsingular and upper triangular 
and `Ei` nilpotent and upper triangular. 
The `nb`-dimensional vectors `ν` contains the dimensions of the diagonal blocks
of the staircase form `Ai-λEi` such that `i`-th block has dimensions `ν[i] x ν[i]`. 
The difference `ν[nb-j+1]-ν[nb-j]` for `j = 1, 2, ..., nb` is the number of infinite elementary 
divisors of degree `j` (with `ν[0] = 0`).

The dimensions of the diagonal blocks are returned in `blkdims = (n1, n2, ni)`.   

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of `A`, the absolute tolerance for the nonzero elements of `E`,  and the relative tolerance 
for the nonzero elements of `A` and `E`. 

The reduction is performed using rank decisions based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.

The performed left orthogonal or unitary transformations are accumulated in the matrix `Q` if `withQ = true`. 
Otherwise, `Q` is set to `nothing`.   
The performed right orthogonal or unitary transformations are accumulated in the matrix `Z` if `withZ = true`. 
Otherwise, `Z` is set to `nothing`.  
"""
function fischursep(A::AbstractMatrix, E::AbstractMatrix; 
                    smarg::Union{Real,Missing} = missing, disc::Bool = false, 
                    fast::Bool = true, finite_infinite::Bool = false, stable_unstable::Bool = false, 
                    atol1::Real = zero(real(eltype(A))), atol2::Real = zero(real(eltype(E))), 
                    rtol::Real = (size(A,1)*eps(real(float(one(eltype(A))))))*iszero(min(atol1,atol2)), 
                    withQ::Bool = true, withZ::Bool = true)

    n = LinearAlgebra.checksquare(A)
    n == LinearAlgebra.checksquare(E) || throw(DimensionMismatch("A and E must have the same dimensions"))    
    disc && (!ismissing(smarg) && smarg < 0 && error("sdeg must be non-negative if disc = true"))
      
    T = promote_type(eltype(A), eltype(E))
    T <: BlasFloat || (T = promote_type(Float64,T))

    A1 = copy_oftype(A,T)   
    E1 = copy_oftype(E,T)

    withQ ? (Q = Matrix{T}(I,n,n)) : (Q = nothing)
    withZ ? (Z = Matrix{T}(I,n,n)) : (Z = nothing)

    ν, blkdims = fisplit!(A1, E1, Q, Z, missing, missing; 
                         fast  = fast, finite_infinite = finite_infinite, 
                         atol1 = atol1, atol2 = atol2, rtol = rtol, withQ = withQ, withZ = withZ)
    (ilo, ihi) = finite_infinite ? (1, blkdims[1]) : (blkdims[1]+1, n)
    if ilo <= ihi 
        compq = withQ ? 'V' : 'N'                    
        compz = withZ ? 'V' : 'N'                    
        gghrd!(compq, compz, ilo, ihi, A1, E1, Q, Z)
        _, _, α, β, _, _ = hgeqz!(compq, compz, ilo, ihi, A1, E1, Q, Z)
        i2 = ilo:ihi
        ismissing(smarg) && (smarg = disc ? one(real(T)) : zero(real(T)))
        select2 = disc ? abs.(α[i2]) .< smarg*abs.(β[i2]) : real.(α[i2] ./ β[i2]) .< smarg
        stable_unstable || (select2 = .!select2)

        if finite_infinite
           n3 = blkdims[2]
           n1 = length(select2[select2 .== true])
           n2 = n-n3-n1
           select = [Int.(select2); zeros(Int,n3)] 
        else
           n1 = blkdims[1]
           n2 = length(select2[select2 .== true])
           n3 = n-n2-n1
           select = [ones(Int,n1);Int.(select2)]
        end
        tgsen!(withQ, withZ, select, A1, E1, Q, Z) 
    else
        (n1,n2,n3) = finite_infinite ? (0,0,blkdims[2]) : (blkdims[1],0,0)
    end
 
    return A1, E1, Q, Z, ν, (n1,n2,n3)                                             
end
"""
    sfischursep(A, E; smarg, disc = false, fast = true, finite_infinite = false, stable_unstable = false, atol1 = 0, atol2 = 0, rtol, withQ = true, withZ = true) -> (At, Et, Q, Z, ν, blkdims)

Reduce the regular matrix pencil `A - λE` to an equivalent block upper triangular form `At - λEt = Q'*(A - λE)*Z` 
using orthogonal or unitary transformation matrices `Q` and `Z` such that the transformed matrices `At` and `Et` 
have separated infinite, stable and unstable eigenvalues with respect to a stability domain `Cs` defined by the stability margin parameter `smarg` and 
the stability type parameter `disc`. If `disc = false`, `Cs` is the set of complex numbers with real parts less than `smarg`, 
while if `disc = true`, `Cs` is the set of complex numbers with moduli less than `smarg` (i.e., the interior of a disc 
of radius `smarg` centered in the origin). If `smarg = missing`, the default value used is `smarg = 0`, if  `disc = false`,
and `smarg = 1`, if `disc = true`.

The pencil `At - λEt` results in one of the following block upper-triangular forms:

(1) if `finite_infinite = true`, then
 
                   | Ai1   *      *       *     |
        At - λEt = |  0  A1-λE1   *       *     |
                   |  0    0    A2-λE2    *     |
                   |  0    0      0    Ai2-λEi2 |
 
where the `ni1 x ni1` matrix `Ai1` and the  `ni2 x ni2` subpencil `Ai2-λEi2` contain the infinite elementary divisors, 
the `n1 x n1` subpencil `A1-λE1` is with the pair `(A1,E1)` in a generalized Schur form, and the 
`n2 x n2` subpencil `A2-λE2` is with the pair `(A2,E2)` in a generalized Schur form. 
The pencil `A1-λE1` has unstable finite eigenvalues and `A2-λE2` has stable finite eigenvalues if `stable_unstable = false`,
while `A1-λE1` has stable finite eigenvalues and `A2-λE2` has unstable finite eigenvalues if `stable_unstable = true`.

The subpencil `Ai2-λEi2` is in a staircase form, with `Ai2` nonsingular and upper triangular and `Ei2` nilpotent and upper triangular. 
The `nb`-dimensional vector `ν` contains the dimensions of the diagonal blocks
of the staircase form  `Ai2-λEi2` such that `i`-th block has dimensions `ν[i] x ν[i]`. 
The difference `ν[nb-j+2]-ν[nb-j+1]` for `j = 1, 2, ..., nb+1` is the number of infinite elementary 
divisors of degree `j` (with `ν[0] := 0` and `ν[nb+1] := ni1`).

The dimensions of the diagonal blocks are returned in `blkdims = (ni1, n1, n2, ni2)`.   

(2) if `finite_infinite = false`, then
 
                   | Ai1-λEi1   *      *     *  |
                   |    0     A1-λE1   *     *  |
        At - λEt = |    0       0    A2-λE2  *  |
                   |    0       0      0    Ai2 |
 
where the `ni1 x ni1` subpencil `Ai1-λEi1` and the `ni2 x ni2` upper triangular and nonsingular matrix `Ai2` contain
the infinite elementary divisors of `A-λE`, 
the `n1 x n1` subpencil `A1-λE1` is with the pair `(A1,E1)` in a generalized Schur form, and the 
`n2 x n2` subpencil `A2-λE2` is with the pair `(A2,E2)` in a generalized Schur form. 
The pencil `A1-λE1` has unstable finite eigenvalues and `A2-λE2` has stable finite eigenvalues if `stable_unstable = false`,
while `A1-λE1` has stable finite eigenvalues and `A2-λE2` has unstable finite eigenvalues if `stable_unstable = true`.

The subpencil `Ai1-λEi1` is in a staircase form, with `Ai1` nonsingular and upper triangular 
and `Ei1` nilpotent and upper triangular. 
The `nb`-dimensional vectors `ν` contains the dimensions of the diagonal blocks
of the staircase form `Ai1-λEi1` such that `i`-th block has dimensions `ν[i] x ν[i]`. 
The difference `ν[i]-ν[i+1]` for `i = 0, 1, 2, ..., nb` is the number of infinite elementary divisors of degree `i` 
(with `ν[nb+1] = 0` and `ν[0] = ni2`).

The dimensions of the diagonal blocks are returned in `blkdims = (ni1, n1, n2, ni2)`.   

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of `A`, the absolute tolerance for the nonzero elements of `E`,  and the relative tolerance 
for the nonzero elements of `A` and `E`. 

The reduction is performed using rank decisions based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.

The performed left orthogonal or unitary transformations are accumulated in the matrix `Q` if `withQ = true`. 
Otherwise, `Q` is set to `nothing`.   
The performed right orthogonal or unitary transformations are accumulated in the matrix `Z` if `withZ = true`. 
Otherwise, `Z` is set to `nothing`.  
"""
function sfischursep(A::AbstractMatrix, E::AbstractMatrix; 
                    smarg::Union{Real,Missing} = missing, disc::Bool = false, 
                    fast::Bool = true, finite_infinite::Bool = false, stable_unstable::Bool = false, 
                    atol1::Real = zero(real(eltype(A))), atol2::Real = zero(real(eltype(E))), 
                    rtol::Real = (size(A,1)*eps(real(float(one(eltype(A))))))*iszero(min(atol1,atol2)), 
                    withQ::Bool = true, withZ::Bool = true)

    n = LinearAlgebra.checksquare(A)
    n == LinearAlgebra.checksquare(E) || throw(DimensionMismatch("A and E must have the same dimensions"))    
    disc && (!ismissing(smarg) && smarg < 0 && error("sdeg must be non-negative if disc = true"))
      
    T = promote_type(eltype(A), eltype(E))
    T <: BlasFloat || (T = promote_type(Float64,T))

    A1 = copy_oftype(A,T)   
    E1 = copy_oftype(E,T)

    withQ ? (Q = Matrix{T}(I,n,n)) : (Q = nothing)
    withZ ? (Z = Matrix{T}(I,n,n)) : (Z = nothing)

    ν, blkdims1 = sfisplit!(A1, E1, Q, Z, missing, missing; 
                         fast  = fast, finite_infinite = finite_infinite, 
                         atol1 = atol1, atol2 = atol2, rtol = rtol, withQ = withQ, withZ = withZ)
    ni1 = blkdims1[1]
    nf  = blkdims1[2]
    ni2 = blkdims1[3]
    (ilo, ihi) = (ni1+1, ni1+nf)
    if ilo <= ihi 
        compq = withQ ? 'V' : 'N'                    
        compz = withZ ? 'V' : 'N'                    
        gghrd!(compq, compz, ilo, ihi, A1, E1, Q, Z)
        _, _, α, β, _, _ = hgeqz!(compq, compz, ilo, ihi, A1, E1, Q, Z)
        i2 = ilo:ihi
        ismissing(smarg) && (smarg = disc ? one(real(T)) : zero(real(T)))
        select2 = disc ? abs.(α[i2]) .< smarg*abs.(β[i2]) : real.(α[i2] ./ β[i2]) .< smarg
        stable_unstable || (select2 = .!select2)

        select = [ones(Int,ni1); Int.(select2); zeros(Int,ni2) ]
        n1 = length(select2[select2 .== true])
        n2 = nf-n1
        blkdims = (ni1, n1, n2, ni2)
        tgsen!(withQ, withZ, select, A1, E1, Q, Z) 
    else
        blkdims = (ni1, 0, 0, ni2)
    end
 
    return A1, E1, Q, Z, ν, blkdims                                             
end
"""
    fiblkdiag(A, E, B, C; fast = true, finite_infinite = false, trinv = false, atol1 = 0, atol2 = 0, rtol, withQ = true, withZ = true) -> (At, Et, Bt, Ct, Q, Z, ν, blkdims, sep)

Reduce the regular matrix pencil `A - λE` to an equivalent form `At - λEt = Q*(A - λE)*Z` using 
the transformation matrices `Q` and `Z` such that the transformed matrices `At` and `Et` are in one of the
following block diagonal forms:

(1) if `finite_infinite = false`, then
 
                   | Ai-λEi |   0     |
        At - λEt = |--------|---------|, 
                   |    O   | Af-λEf  |
 
where the `ni x ni` subpencil `Ai-λEi` contains the infinite elementary 
divisors and the `nf x nf` subpencil `Af-λEf`, with `Ef` nonsingular and upper triangular, contains the finite eigenvalues of the pencil `A-λE`.

The subpencil `Ai-λEi` is in a staircase form, with `Ai` nonsingular and upper triangular and `Ei` nilpotent and upper triangular. 
The `nb`-dimensional vector `ν` contains the dimensions of the diagonal blocks
of the staircase form  `Ai-λEi` such that `i`-th block has dimensions `ν[i] x ν[i]`. 
The difference `ν[i]-ν[i+1]` for `i = 1, 2, ..., nb` is the number of infinite elementary divisors of degree `i` 
(with `ν[nb+1] = 0`).

The dimensions of the diagonal blocks are returned in `blkdims = (ni, nf)`.   

(2) if `finite_infinite = true`, then
 
                   | Af-λEf |   0    |
        At - λEt = |--------|--------|, 
                   |   O    | Ai-λEi |
 
where the `nf x nf` subpencil `Af-λEf`, with `Ef` nonsingular and upper triangular, 
contains the finite eigenvalues of the pencil `A-λE` and the `ni x ni` subpencil `Ai-λEi` 
contains the infinite elementary divisors.

The subpencil `Ai-λEi` is in a staircase form, with `Ai` nonsingular and upper triangular  and `Ei` nilpotent and upper triangular. 
The `nb`-dimensional vectors `ν` contains the dimensions of the diagonal blocks
of the staircase form `Ai-λEi` such that `i`-th block has dimensions `ν[i] x ν[i]`. 
The difference `ν[nb-j+1]-ν[nb-j]` for `j = 1, 2, ..., nb` is the number of infinite elementary 
divisors of degree `j` (with `ν[0] = 0`).

The dimensions of the diagonal blocks are returned in `blkdims = (nf, ni)`.   

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of `A`, the absolute tolerance for the nonzero elements of `E`,  and the relative tolerance 
for the nonzero elements of `A` and `E`. 

The reduction is performed using rank decisions based on rank revealing QR-decomdiagpositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.

If `withQ = true`, `Q` contains the left transformation matrix, if `trinv = false`, or its inverse, if `trinv = true`. 
If `withQ = false`, `Q` is set to `nothing`.   
If `withZ = true`, `Z` contains the right transformation matrix, if `trinv = false`, or its inverse, if `trinv = true`. 
If `withZ = false`, `Z` is set to `nothing`.   

`Bt = Q*B`, unless `B = missing`, in which case `Bt = missing` is returned, and `Ct = C*Z`, 
unless `C = missing`, in which case `Ct = missing` is returned .              

An estimation of the separation of the spectra of `Ai-λEi` and `Af-λEf` is returned in `sep`, where  `0 < sep ≤ 1`.
"""
function fiblkdiag(A::AbstractMatrix, E::AbstractMatrix, B::Union{AbstractMatrix,Missing}, C::Union{AbstractMatrix,Missing}; 
                   fast::Bool = true, finite_infinite::Bool = false, trinv::Bool = false, 
                   atol1::Real = zero(real(eltype(A))), atol2::Real = zero(real(eltype(E))), 
                   rtol::Real = (size(A,1)*eps(real(float(one(eltype(A))))))*iszero(min(atol1,atol2)), 
                   withQ::Bool = true, withZ::Bool = true)

    compq = withQ || !ismissing(B) 
    compz = withZ || !ismissing(C) 
    A1, E1, Q, Z, ν, blkdims = fischur(A, E; fast = fast, finite_infinite = finite_infinite, atol1 = atol1, atol2 = atol2,
                                      rtol = rtol, withQ = compq, withZ = compz) 
    T = eltype(A1)
    ONER = one(real(T))
    ismissing(B) ? B1 = missing : B1 = Q'*copy_oftype(B,T)
    ismissing(C) ? C1 = missing : C1 = copy_oftype(C,T)*Z
    
    minimum(blkdims) == 0 && (return A1, E1, B1, C1, withQ ? (trinv ? Q : adjis!(Q)) : nothing, 
                              withZ ? (trinv ? adjis!(Z) : Z) : nothing, ν, blkdims, ONER)   

    trinv ? (withZ && adjis!(Z)) : (withQ && adjis!(Q))  

    n1 = blkdims[1]
    i1 = 1:n1
    i2 = n1+1:size(A,1)
    X = view(E1,i1,i2)
    Y = view(A1,i1,i2)
    _, _, scale = tgsyl!(view(A1,i1,i1), view(A1,i2,i2), Y, view(E1,i1,i1), view(E1,i2,i2), X) 
    
    ONE = one(T)
    ZERO = zero(T)
    scale != 0 && (scale = ONE/scale)

    ismissing(B) || mul!(view(B1,i1,:),X,view(B1,i2,:),scale,ONE)
    ismissing(C) || mul!(view(C1,:,i2),view(C1,:,i1),Y,-scale,ONE)

    if trinv
       withQ && mul!(view(Q,:,i2),view(Q,:,i1),X,-scale,ONE) 
       withZ && mul!(view(Z,i1,:),Y,view(Z,i2,:),scale,ONE)
    else
       withQ && mul!(view(Q,i1,:),X,view(Q,i2,:),scale,ONE) 
       withZ && mul!(view(Z,:,i2),view(Z,:,i1),Y,-scale,ONE) 
    end  

    scale == 0 ? sep = ONER : sep = ONER/max(1+norm(X)*abs(scale),1+norm(Y)*abs(scale))
   
    fill!(X,ZERO)
    fill!(Y,ZERO)
 
    return A1, E1, B1, C1, Q, Z, ν, blkdims, sep                                             
                       
end
"""
    gsblkdiag(A, E, B, C; smarg, disc = false, fast = true, finite_infinite = false, stable_unstable = false, trinv = false, 
              atol1 = 0, atol2 = 0, rtol, withQ = true, withZ = true) -> (At, Et, Bt, Ct, Q, Z, ν, blkdims, sep)

Reduce the regular matrix pencil `A - λE` to an equivalent block diagonal triangular form `At - λEt = Q*(A - λE)*Z` 
using the transformation matrices `Q` and `Z` such that the transformed matrices `At` and `Et` 
have separated infinite, stable and unstable eigenvalues with respect to a stability domain `Cs` 
defined by the stability margin parameter `smarg` and the stability type parameter `disc`. 
If `disc = false`, `Cs` is the set of complex numbers with real parts less than `smarg`, 
while if `disc = true`, `Cs` is the set of complex numbers with moduli less than `smarg` (i.e., the interior of a disc 
of radius `smarg` centered in the origin). If `smarg = missing`, the default value used is `smarg = 0`, if  `disc = false`,
and `smarg = 1`, if `disc = true`.

The pencil `At - λEt` results in one of the following block upper-triangular forms:

(1) if `finite_infinite = false`, then
 
                   | Ai-λEi   *      0    |
        At - λEt = |    O   A1-λE1   0    |
                   |    0     0    A2-λE2 |
 
where the `ni x ni` subpencil `Ai-λEi` contains the infinite elementary divisors, 
the `n1 x n1` subpencil `A1-λE1` is with the pair `(A1,E1)` in a generalized Schur form, and the 
`n2 x n2` subpencil `A2-λE2` is with the pair `(A2,E2)` in a generalized Schur form. 
The pencil `A1-λE1` has unstable finite eigenvalues and `A2-λE2` has stable finite eigenvalues if `stable_unstable = false`,
while `A1-λE1` has stable finite eigenvalues and `A2-λE2` has unstable finite eigenvalues if `stable_unstable = true`.

The subpencil `Ai-λEi` is in a staircase form, with `Ai` nonsingular and upper triangular and `Ei` nilpotent and upper triangular. 
The `nb`-dimensional vector `ν` contains the dimensions of the diagonal blocks
of the staircase form  `Ai-λEi` such that `i`-th block has dimensions `ν[i] x ν[i]`. 
The difference `ν[i]-ν[i+1]` for `i = 1, 2, ..., nb` is the number of infinite elementary divisors of degree `i` 
(with `ν[nb+1] = 0`).

The dimensions of the diagonal blocks are returned in `blkdims = (ni, n1, n2)`.   

(2) if `finite_infinite = true`, then
 
                   | A1-λE1   0      0    |
        At - λEt = |    O   A2-λE2   *    |
                   |    0     0    Ai-λEi |
 
where the `ni x ni` subpencil `Ai-λEi` contains the infinite elementary divisors, 
the `n1 x n1` subpencil `A1-λE1` is with the pair `(A1,E1)` in a generalized Schur form, and the 
`n2 x n2` subpencil `A2-λE2` is with the pair `(A2,E2)` in a generalized Schur form. 
The pencil `A1-λE1` has unstable finite eigenvalues and `A2-λE2` has stable finite eigenvalues if `stable_unstable = false`,
while `A1-λE1` has stable finite eigenvalues and `A2-λE2` has unstable finite eigenvalues if `stable_unstable = true`.

The subpencil `Ai-λEi` is in a staircase form, with `Ai` nonsingular and upper triangular 
and `Ei` nilpotent and upper triangular. 
The `nb`-dimensional vectors `ν` contains the dimensions of the diagonal blocks
of the staircase form `Ai-λEi` such that `i`-th block has dimensions `ν[i] x ν[i]`. 
The difference `ν[nb-j+1]-ν[nb-j]` for `j = 1, 2, ..., nb` is the number of infinite elementary 
divisors of degree `j` (with `ν[0] = 0`).

The dimensions of the diagonal blocks are returned in `blkdims = (n1, n2, ni)`.   

The keyword arguments `atol1`, `atol2`, and `rtol`, specify, respectively, the absolute tolerance for the 
nonzero elements of `A`, the absolute tolerance for the nonzero elements of `E`,  and the relative tolerance 
for the nonzero elements of `A` and `E`. 

The reduction is performed using rank decisions based on rank revealing QR-decompositions with column pivoting 
if `fast = true` or the more reliable SVD-decompositions if `fast = false`.

If `withQ = true`, `Q` contains the left transformation matrix, if `trinv = false`, or its inverse, if `trinv = true`. 
If `withQ = false`, `Q` is set to `nothing`.   
If `withZ = true`, `Z` contains the left transformation matrix, if `trinv = false`, or its inverse, if `trinv = true`. 
If `withZ = false`, `Z` is set to `nothing`.   

`Bt = Q*B`, unless `B = missing`, in which case `Bt = missing` is returned, and `Ct = C*Z`, 
unless `C = missing`, in which case `Ct = missing` is returned .              

An estimation of the separation of the spectra of the two underlying diagonal blocks is returned in `sep`, 
where  `0 ≤ sep ≤ 1`. A value `sep ≈ 0` indicates that `A1-λE1` and `A2-λE2` have some almost equal eigenvalues. 
"""
function gsblkdiag(A::AbstractMatrix, E::AbstractMatrix, B::Union{AbstractMatrix,Missing}, C::Union{AbstractMatrix,Missing}; 
                   smarg::Union{Real,Missing} = missing, disc::Bool = false, trinv::Bool = false, 
                   fast::Bool = true, finite_infinite::Bool = false, stable_unstable::Bool = false, 
                   atol1::Real = zero(real(eltype(A))), atol2::Real = zero(real(eltype(E))), 
                   rtol::Real = (size(A,1)*eps(real(float(one(eltype(A))))))*iszero(min(atol1,atol2)), 
                   withQ::Bool = true, withZ::Bool = true)

    compq = withQ || !ismissing(B)  
    compz = withZ || !ismissing(C) 
    A1, E1, Q, Z, ν, blkdims = fischursep(A, E; smarg = smarg, disc = disc, 
                                      fast = fast, finite_infinite = finite_infinite, stable_unstable = stable_unstable,
                                      atol1 = atol1, atol2 = atol2, rtol = rtol, withQ = compq, withZ = compz) 
    T = eltype(A1)
    ONER = one(real(T))
    ismissing(B) ? B1 = missing : B1 = Q'*copy_oftype(B,T)
    ismissing(C) ? C1 = missing : C1 = copy_oftype(C,T)*Z

    finite_infinite ? n1 = blkdims[1] : n1 = blkdims[1] + blkdims[2]                     
    (n1 == 0 || size(A1,1) == n1) && (return A1, E1, B1, C1, withQ ? (trinv ? Q : adjis!(Q)) : nothing, 
               withZ ? (trinv ? adjis!(Z) : Z) : nothing, ν, blkdims,ONER)   

    trinv ? (withZ && adjis!(Z)) : (withQ && adjis!(Q))  

    i1 = 1:n1
    i2 = n1+1:size(A,1)
    X = view(E1,i1,i2)
    Y = view(A1,i1,i2)
    _, _, scale = tgsyl!(view(A1,i1,i1), view(A1,i2,i2), Y, view(E1,i1,i1), view(E1,i2,i2), X) 
    
    ONE = one(T)
    ZERO = zero(T)
    scale != 0 && (scale = ONE/scale)

    ismissing(B) || mul!(view(B1,i1,:),X,view(B1,i2,:),scale,ONE)
    ismissing(C) || mul!(view(C1,:,i2),view(C1,:,i1),Y,-scale,ONE)

    if trinv
       withQ && mul!(view(Q,:,i2),view(Q,:,i1),X,-scale,ONE) 
       withZ && mul!(view(Z,i1,:),Y,view(Z,i2,:),scale,ONE)
    else
       withQ && mul!(view(Q,i1,:),X,view(Q,i2,:),scale,ONE) 
       withZ && mul!(view(Z,:,i2),view(Z,:,i1),Y,-scale,ONE) 
    end  

    scale == 0 ? sep = ONER : sep = ONER/max(1+norm(X)*abs(scale),1+norm(Y)*abs(scale))
   
    fill!(X,ZERO)
    fill!(Y,ZERO)

    return A1, E1, B1, C1, Q, Z, ν, blkdims, sep                                             
        
end
@inline function adjis!(A::AbstractMatrix) 
    # in-situ adjoint of a square matrix
    n = LinearAlgebra.checksquare(A)
    for i = 1:n
        for j = i:n
            t = A[i,j]'
            A[i,j] = A[j,i]'
            A[j,i] = t
        end
    end
    return A 
end
"""
    ssblkdiag(A, B, C; smarg, disc = false, stable_unstable = false, withQ = true, withZ = true) -> (At, Bt, Ct, Q, Z, blkdims, sep)

Reduce the regular matrix pencil `A - λI` to an equivalent block diagonal triangular form `At - λI = Q*(A - λI)*Z` 
using the transformation matrices `Q` and `Z`, where `Q = inv(Z)`, such that the transformed matrix `At` have 
separated stable and unstable eigenvalues with respect to a stability domain `Cs` defined by the 
stability margin parameter `smarg` and the stability type parameter `disc`. 
If `disc = false`, `Cs` is the set of complex numbers with real parts less than `smarg`, 
while if `disc = true`, `Cs` is the set of complex numbers with moduli less than `smarg` (i.e., the interior of a disc 
of radius `smarg` centered in the origin). If `smarg = missing`, the default value used is `smarg = 0`, if  `disc = false`,
and `smarg = 1`, if `disc = true`.
The matrix `At` results in the following block diagonal form
 
        At = | A1  0  |
             | 0   A2 |
 
where the `n1 x n1` matrix `A1` and the `n2 x n2` matrix `A2` are in Schur form. 
The matrix `A1` has unstable eigenvalues and `A2` has stable eigenvalues if `stable_unstable = false`,
while `A1` has stable eigenvalues and `A2` has unstable eigenvalues if `stable_unstable = true`.
The dimensions of the diagonal blocks are returned in `blkdims = (n1, n2)`.   
If `withQ = true`, `Q` contains the left transformation matrix. If `withQ = false`, `Q` is set to `nothing`.   
If `withZ = true`, `Z` contains the right transformation matrix. If `withZ = false`, `Z` is set to `nothing`.   
`Bt = Q*B`, unless `B = missing`, in which case `Bt = missing` is returned, and `Ct = C*Z`, 
unless `C = missing`, in which case `Ct = missing` is returned .              
An estimation of the separation of the spectra of the two underlying diagonal blocks is returned in `sep`, 
where  `0 ≤ sep ≤ 1`. A value `sep ≈ 0` indicates that `A1` and `A2` have some almost equal eigenvalues. 
"""
function ssblkdiag(A::AbstractMatrix{T}, B::Union{AbstractMatrix{T},Missing}, C::Union{AbstractMatrix{T},Missing}; 
                   smarg::Union{Real,Missing} = missing, disc::Bool = false, stable_unstable::Bool = false, 
                   withQ::Bool = true, withZ::Bool = true ) where T <: BlasFloat
    
    n = LinearAlgebra.checksquare(A)
    S = schur(A)
    ONER = real(float(one(T)))
    if disc
       ismissing(smarg) && (smarg = ONER)
       stable_unstable ? select = abs.(S.values) .< smarg : select = abs.(S.values) .> smarg 
    else
       ismissing(smarg) && (smarg = real(float(zero(T))))
       stable_unstable ? select = real(S.values) .< smarg : select = real(S.values) .> smarg 
    end
    n1 = length(filter(y-> y == true,select)) 

    (n1 == 0 || n1 == n) && (return A, B, C, withQ ? copy(S.Z') : nothing, withZ ? S.Z : nothing, (n1,n-n1), ONER)   

    ordschur!(S, select)

    i1 = 1:n1
    i2 = n1+1:size(A,1)
    Y = view(S.T,i1,i2)  # Y will contain the negative of solution -Y
    _, scale = LAPACK.trsyl!('N', 'N', view(S.T,i1,i1), view(S.T,i2,i2), Y, -1) 
 
    ONE = one(T)
    ZERO = zero(T)
    scale != 0 && (scale = ONE/scale)

    ismissing(B) ? B1 = missing : (B1 = S.Z'*B; mul!(view(B1,i1,:),Y,view(B1,i2,:),scale,ONE))
    ismissing(C) ? C1 = missing : (C1 = C*S.Z; mul!(view(C1,:,i2),view(C1,:,i1),Y,-scale,ONE))

    withQ ? Q = copy(S.Z') : Q = nothing 
    withQ && mul!(view(Q,i1,:),Y,view(Q,i2,:),scale,ONE) 
    withZ && mul!(view(S.Z,:,i2),view(S.Z,:,i1),Y,-scale,ONE)

    scale == 0 ? sep = zero(real(T)) : sep = ONER/(1+norm(Y)*abs(scale))
   
    fill!(Y,ZERO)
    return S.T, B1, C1, Q, withZ ? S.Z : nothing, (n1, n-n1), sep                                                    
end
