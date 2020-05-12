module LapackUtil2

const liblapack = Base.liblapack_name

import LinearAlgebra.BLAS.@blasfunc

import LinearAlgebra: BlasFloat, BlasInt, LAPACKException,
    DimensionMismatch, SingularException, PosDefException, chkstride1, checksquare

using Base: iszero, has_offset_axes

export larfg!, larfgl!, larf!

function chkside(side::AbstractChar)
    # Check that left/right hand side multiply is correctly specified
    side == 'L' || side == 'R' ||
        throw(ArgumentError("side argument must be 'L' (left hand multiply) or 'R' (right hand multiply), got $side"))
    side
end


## Tools to compute and apply elementary reflectors
for (larfg, elty) in
    ((:dlarfg_, Float64),
     (:slarfg_, Float32),
     (:zlarfg_, ComplexF64),
     (:clarfg_, ComplexF32))
    @eval begin
        # 
        #    larfg!(x) -> (τ, β)
        #
        # Wrapper to LAPACK function family _LARFG.F to compute the parameters 
        # v and τ of a Householder reflector H = I - τ*v*v' which annihilates 
        # the N-1 trailing elements of x and to provide τ and β, where β is the  
        # first element of the transformed vector H*x. The vector v has its first 
        # component set to 1 and is returned in x.  
        #
        #        .. Scalar Arguments ..
        #        INTEGER            incx, n
        #        DOUBLE PRECISION   alpha, tau
        #        ..
        #        .. Array Arguments ..
        #        DOUBLE PRECISION   x( * )
        function larfg!(x::AbstractVector{$elty})
            N    = BlasInt(length(x))
            incx = stride(x, 1)
            τ    = Ref{$elty}(0)
            ccall((@blasfunc($larfg), liblapack), Cvoid,
                (Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ref{$elty}),
                N, x, pointer(x, 2), incx, τ)
            β = x[1]
            @inbounds x[1] = one($elty)
            return τ[], β
        end
    end
end

for (larfg, elty) in
    ((:dlarfg_, Float64),
     (:slarfg_, Float32),
     (:zlarfg_, ComplexF64),
     (:clarfg_, ComplexF32))
    @eval begin
        # 
        #    larfgl!(x) -> (τ, β)
        #
        # Wrapper to LAPACK function family _LARFG.F to compute the parameters 
        # v and τ of a Householder reflector H = I - τ*v*v' which annihilates 
        # the N-1 leading elements of x and to provide τ and β, where β is the  
        # last element of the transformed vector H*x. The vector v has its last 
        # component set to 1 and is returned in x.  
        #
        #        .. Scalar Arguments ..
        #        INTEGER            incx, n
        #        DOUBLE PRECISION   alpha, tau
        #        ..
        #        .. Array Arguments ..
        #        DOUBLE PRECISION   x( * )
        function larfgl!(x::AbstractVector{$elty})
            N    = BlasInt(length(x))
            incx = stride(x, 1)
            τ    = Ref{$elty}(0)
            ccall((@blasfunc($larfg), liblapack), Cvoid,
                (Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ref{$elty}),
                N, pointer(x, N), x, incx, τ)
            β = x[N]
            @inbounds x[N] = one($elty)
            return τ[], β
        end
    end
end

for (larf, elty) in
    ((:dlarf_, Float64),
     (:slarf_, Float32),
     (:zlarf_, ComplexF64),
     (:clarf_, ComplexF32))
    @eval begin
        # 
        #    larf!(side,v,τ,C) 
        #
        # Wrapper to LAPACK function family _LARF.F to apply a Householder reflector 
        # H = I - τ*v*v' from left, if side = 'L', or from right, if side = 'R', to 
        # the matrix C. H*C, if side = 'L' or C*H, if side = 'R', are returned in C. 
        #  
        #        .. Scalar Arguments ..
        #        CHARACTER          side
        #        INTEGER            incv, ldc, m, n
        #        DOUBLE PRECISION   tau
        #        ..
        #        .. Array Arguments ..
        #        DOUBLE PRECISION   c( ldc, * ), v( * ), work( * )
        function larf!(side::AbstractChar, v::AbstractVector{$elty},
                       τ::$elty, C::AbstractMatrix{$elty}, work::AbstractVector{$elty})
            m, n = size(C)
            chkside(side)
            ldc = max(1, stride(C, 2))
            l = side == 'L' ? n : m
            #incv  = BlasInt(1)
            incv = stride(v, 1)
            ccall((@blasfunc($larf), liblapack), Cvoid,
                (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                 Ref{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Clong),
                side, m, n, v, incv,
                τ, C, ldc, work, 1)
            return C
        end

        function larf!(side::AbstractChar, v::AbstractVector{$elty},
                       τ::$elty, C::AbstractMatrix{$elty})
            m, n = size(C)
            chkside(side)
            lwork = side == 'L' ? n : m
            return larf!(side, v, τ, C, Vector{$elty}(undef,lwork))
        end
    end
end

end
