module LapackUtil2

const liblapack = VERSION < v"1.7" ? Base.liblapack_name : "libblastrampoline"

import LinearAlgebra.BLAS.@blasfunc

import LinearAlgebra: BlasFloat, BlasReal, BlasComplex, BlasInt, LAPACKException, 
       DimensionMismatch, SingularException, PosDefException, chkstride1, checksquare

using Base: iszero, has_offset_axes

export larfg!, larfgl!, larf!

function chkside(side::AbstractChar)
    # Check that left/right hand side multiply is correctly specified
    side == 'L' || side == 'R' ||
        throw(ArgumentError("side argument must be 'L' (left hand multiply) or 'R' (right hand multiply), got $side"))
    side
end
function chklapackerror(ret::BlasInt)
    ret == 0 ? (return) : 
              (ret < 0 ? throw(ArgumentError("invalid argument #$(-ret) to LAPACK call")) : throw(LAPACKException(ret)))
end

for (fn, elty) in ((:dlanv2_, :Float64),
                   (:slanv2_, :Float32))
    @eval begin
        function lanv2(A::$elty, B::$elty, C::$elty, D::$elty)
           """
           SUBROUTINE DLANV2( A, B, C, D, RT1R, RT1I, RT2R, RT2I, CS, SN )

           DOUBLE PRECISION A, B, C, CS, D, RT1I, RT1R, RT2I, RT2R, SN
           """
           RT1R = Ref{$elty}(1.0)
           RT1I = Ref{$elty}(1.0)
           RT2R = Ref{$elty}(1.0)
           RT2I = Ref{$elty}(1.0)
           CS = Ref{$elty}(1.0)
           SN = Ref{$elty}(1.0)
           ccall((@blasfunc($fn), liblapack), Cvoid,
                 (Ref{$elty},Ref{$elty},Ref{$elty},Ref{$elty},
                 Ref{$elty},Ref{$elty},Ref{$elty},Ref{$elty},Ref{$elty},Ref{$elty}),
                 A, B, C, D,
                 RT1R, RT1I, RT2R, RT2I, CS, SN)
           return RT1R[], RT1I[], RT2R[], RT2I[], CS[], SN[]
        end
    end
end
"""
    lanv2(A, B, C, D) -> (RT1R, RT1I, RT2R, RT2I, CS, SN)

Compute the Schur factorization of a real 2-by-2 nonsymmetric matrix `[A,B;C,D]` in
standard form. `A`, `B`, `C`, `D` are overwritten on output by the corresponding elements of the
standardised Schur form. `RT1R+im*RT1I` and `RT2R+im*RT2I` are the resulting eigenvalues.
`CS` and `SN` are the parameters of the rotation matrix.
Interface to the LAPACK subroutines DLANV2/SLANV2.
"""
lanv2(A::BlasReal, B::BlasReal, C::BlasReal, D::BlasReal)


for (fn, elty) in ((:dlag2_, :Float64),
                   (:slag2_, :Float32))
    @eval begin
        function lag2(A::StridedMatrix{$elty}, B::StridedMatrix{$elty}, SAFMIN::$elty)
           """
           SUBROUTINE DLAG2( A, LDA, B, LDB, SAFMIN, SCALE1, SCALE2, WR1, WR2, WI )

           INTEGER            LDA, LDB
           DOUBLE PRECISION   SAFMIN, SCALE1, SCALE2, WI, WR1, WR2
           DOUBLE PRECISION   A( LDA, * ), B( LDB, * )
           """
           LDA = stride(A,2)
           LDB = stride(B,2)
           SCALE1 = Ref{$elty}(1.0)
           SCALE2 = Ref{$elty}(1.0)
           WR1 = Ref{$elty}(1.0)
           WR2 = Ref{$elty}(1.0)
           WI = Ref{$elty}(1.0)
           ccall((@blasfunc($fn), liblapack), Cvoid,
                (Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ref{$elty},
                Ref{$elty},Ref{$elty},Ref{$elty},Ref{$elty},Ref{$elty}),
                A, LDA, B, LDB, SAFMIN,
                SCALE1, SCALE2, WR1, WR2, WI)
           return SCALE1[], SCALE2[], WR1[], WR2[], WI[]
        end
    end
end
"""
    lag2(A, B, SAFMIN) -> (SCALE1, SCALE2, WR1, WR2, WI)

Compute the eigenvalues of a 2-by-2 generalized real eigenvalue problem for
the matrix pair `(A,B)`, with scaling as necessary to avoid over-/underflow.
`SAFMIN` is the smallest positive number s.t. `1/SAFMIN` does not overflow.
If `WI = 0`, `WR1/SCALE1` and `WR2/SCALE2` are the resulting real eigenvalues, while
if `WI <> 0`, then `(WR1+/-im*WI)/SCALE1` are the resulting complex eigenvalues.
Interface to the LAPACK subroutines DLAG2/SLAG2.
"""
lag2(A::StridedMatrix{BlasReal}, B::StridedMatrix{BlasReal}, SAFMIN::BlasReal) 

#lag2(A::StridedMatrix{T}, B::StridedMatrix{T}) where T <: BlasReal = lag2(A,B,safemin(T))

function safemin(::Type{T}) where T <: BlasReal
    SMLNUM = (T == Float64) ? reinterpret(Float64, 0x2000000000000000) : reinterpret(Float32, 0x20000000)
    return SMLNUM * 2/ eps(T)
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

for (gghrd, elty) in
    ((:dgghrd_, Float64),
     (:sgghrd_, Float32),
     (:zgghrd_, ComplexF64),
     (:cgghrd_, ComplexF32))
    @eval begin
        #       SUBROUTINE DGGHRD( COMPQ, COMPZ, N, ILO, IHI, A, LDA, B, LDB, Q,
        #                          LDQ, Z, LDZ, INFO )
        #
        #       .. Scalar Arguments ..
        #       CHARACTER          COMPQ, COMPZ
        #       INTEGER            IHI, ILO, INFO, LDA, LDB, LDQ, LDZ, N
        #       ..
        #       .. Array Arguments ..
        #       DOUBLE PRECISION   A( LDA, * ), B( LDB, * ), Q( LDQ, * ), Z( LDZ, * )
        function gghrd!(compq::AbstractChar, compz::AbstractChar,  ilo::BlasInt, ihi::BlasInt, 
                        A::AbstractMatrix{$elty}, B::AbstractMatrix{$elty}, Q::AbstractMatrix{$elty}, Z::AbstractMatrix{$elty})
            chkstride1(A, B, Q, Z)
            n, nb, nq, nz = checksquare(A, B, Q, Z)
            n == nb || throw(DimensionMismatch("dimensions of A, ($n,$n), and B, ($nb,$nb), must match"))
            n == nq || throw(DimensionMismatch("dimensions of A, ($n,$n), and Q, ($nq,$nq), must match"))
            n == nz || throw(DimensionMismatch("dimensions of A, ($n,$n), and Z, ($nz,$nz), must match"))
            lda = max(1, stride(A, 2))
            ldb = max(1, stride(B, 2))
            ldq = max(1, stride(Q, 2))
            ldz = max(1, stride(Z, 2))
            compq == 'N' || compq == 'V' || compq == 'I' ||
                  throw(ArgumentError("compq argument must be 'N', 'V' or 'I', got $compq"))
            compz == 'N' || compz == 'V' || compz == 'I' ||
                  throw(ArgumentError("compz argument must be 'N', 'V' or 'I', got $compz"))
            n > 0 && (ilo < 1 || ihi > n || ihi < ilo) && 
                    throw(ArgumentError("ilo and ihi arguments must satisfy 1 ≤ ilo ≤ ihi ≤ $n, got ilo = $ilo and ihi = $ihi"))
            n == 0 && (ilo != 1 || ihi != 0) && 
                    throw(ArgumentError("for n = 0, ilo must be 1 and ihi must be zero, got ilo = $ilo and ihi = $ihi"))
            info = Ref{BlasInt}()
            ccall((@blasfunc($gghrd), liblapack), Cvoid,
                      (Ref{UInt8}, Ref{UInt8},  Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                      Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                      Ptr{BlasInt}, Clong, Clong),
                      compq, compz, n, ilo, ihi, 
                      A, lda, B, ldb, Q, ldq, Z, ldz, 
                      info, 1, 1)
                chklapackerror(info[])
           return A, B, Q, Z
        end
    end
end

for (hgeqz, elty) in
    ((:dhgeqz_,:Float64),
     (:shgeqz_,:Float32))
     @eval begin
        #       SUBROUTINE DHGEQZ( JOB, COMPQ, COMPZ, N, ILO, IHI, H, LDH, T, LDT,
        #                          ALPHAR, ALPHAI, BETA, Q, LDQ, Z, LDZ, WORK, LWORK, INFO )
        #
        #       .. Scalar Arguments ..
        #       CHARACTER          COMPQ, COMPZ, JOB
        #       INTEGER            IHI, ILO, INFO, LDH, LDQ, LDT, LDZ, LWORK, N
        #       ..
        #       .. Array Arguments ..
        #       DOUBLE PRECISION   ALPHAI( * ), ALPHAR( * ), BETA( * ),
        #      $                   H( LDH, * ), Q( LDQ, * ), T( LDT, * ),
        #      $                   WORK( * ), Z( LDZ, * )
        function hgeqz!(compq::AbstractChar, compz::AbstractChar,  ilo::BlasInt, ihi::BlasInt,  
                        H::AbstractMatrix{$elty}, T::AbstractMatrix{$elty}, Q::AbstractMatrix{$elty}, Z::AbstractMatrix{$elty})
            chkstride1(H, T, Q, Z)
            n, nt, nq, nz = checksquare(H, T, Q, Z)
            n == nt || throw(DimensionMismatch("dimensions of H, ($n,$n), and T, ($nt,$nt), must match"))
            n == nq || throw(DimensionMismatch("dimensions of H, ($n,$n), and Q, ($nq,$nq), must match"))
            n == nz || throw(DimensionMismatch("dimensions of H, ($n,$n), and Z, ($nz,$nz), must match"))
            compq == 'N' || compq == 'V' || compq == 'I' ||
                  throw(ArgumentError("compq argument must be 'N', 'V' or 'I', got $compq"))
            compz == 'N' || compz == 'V' || compz == 'I' ||
                  throw(ArgumentError("compz argument must be 'N', 'V' or 'I', got $compz"))
            n > 0 && (ilo < 1 || ihi > n || ihi < ilo) && 
                    throw(ArgumentError("ilo and ihi arguments must satisfy 1 ≤ ilo ≤ ihi ≤ $n, got ilo = $ilo and ihi = $ihi"))
            n == 0 && (ilo != 1 || ihi != 0) && 
                    throw(ArgumentError("for n = 0, ilo must be 1 and ihi must be zero, got ilo = $ilo and ihi = $ihi"))
            alphar = similar(H, $elty, n)
            alphai = similar(H, $elty, n)
            beta = similar(H, $elty, n)
            ldh = max(1, stride(H, 2))
            ldt = max(1, stride(T, 2))
            ldq = compq == 'N' ? 1 : max(1, stride(Q, 2))
            ldz = compz == 'N' ? 1 : max(1, stride(Z, 2))
            work = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            info = Ref{BlasInt}()
            for i = 1:2  # first call returns lwork as work[1]
                ccall((@blasfunc($hgeqz), liblapack), Cvoid,
                      (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                       Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, 
                       Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, 
                       Ptr{$elty}, Ref{BlasInt}, 
                       Ptr{BlasInt}, Clong, Clong, Clong),
                    'S', compq, compz, n, ilo, ihi, 
                    H, ldh, T, ldt, alphar, alphai, beta, 
                    Q, ldq, Z, ldz, 
                    work, lwork, 
                    info, 1, 1, 1)
                chklapackerror(info[])
                if i == 1
                    lwork = BlasInt(real(work[1]))
                    resize!(work, lwork)
                end
            end
            H, T, complex.(alphar, alphai), beta, Q, Z
        end
    end
end
for (hgeqz, elty, relty) in
    ((:zhgeqz_,:ComplexF64,:Float64),
     (:chgeqz_,:ComplexF32,:Float32))
     @eval begin
        #       SUBROUTINE ZHGEQZ( JOB, COMPQ, COMPZ, N, ILO, IHI, H, LDH, T, LDT,
        #                          ALPHA, BETA, Q, LDQ, Z, LDZ, WORK, LWORK, RWORK, INFO )
        #
        #       .. Scalar Arguments ..
        #       CHARACTER          COMPQ, COMPZ, JOB
        #       INTEGER            IHI, ILO, INFO, LDH, LDQ, LDT, LDZ, LWORK, N
        #       ..
        #       .. Array Arguments ..
        #       REAL               RWORK( * )
        #       COMPLEX*16         ALPHA( * ), BETA( * ), H( LDH, * ),
        #      $                   Q( LDQ, * ), T( LDT, * ), WORK( * ),
        #      $                   Z( LDZ, * )
        function hgeqz!(compq::AbstractChar, compz::AbstractChar,  ilo::BlasInt, ihi::BlasInt,  
                        H::AbstractMatrix{$elty}, T::AbstractMatrix{$elty}, Q::AbstractMatrix{$elty}, Z::AbstractMatrix{$elty})
            chkstride1(H, T, Q, Z)
            n, nt, nq, nz = checksquare(H, T, Q, Z)
            n == nt || throw(DimensionMismatch("dimensions of H, ($n,$n), and T, ($nt,$nt), must match"))
            n == nq || throw(DimensionMismatch("dimensions of H, ($n,$n), and Q, ($nq,$nq), must match"))
            n == nz || throw(DimensionMismatch("dimensions of H, ($n,$n), and Z, ($nz,$nz), must match"))
            compq == 'N' || compq == 'V' || compq == 'I' ||
                  throw(ArgumentError("compq argument must be 'N', 'V' or 'I', got $compq"))
            compz == 'N' || compz == 'V' || compz == 'I' ||
                  throw(ArgumentError("compz argument must be 'N', 'V' or 'I', got $compz"))
            n > 0 && (ilo < 1 || ihi > n || ihi < ilo) && 
                    throw(ArgumentError("ilo and ihi arguments must satisfy 1 ≤ ilo ≤ ihi ≤ n, got ilo = $ilo and ihi = $ihi"))
            n == 0 && (ilo != 1 || ihi != 0) && 
                    throw(ArgumentError("for n = 0, ilo must be 1 and ihi must be zero, got ilo = $ilo and ihi = $ihi"))
            alpha = similar(H, $elty, n)
            beta = similar(H, $elty, n)
            ldh = max(1, stride(H, 2))
            ldt = max(1, stride(T, 2))
            ldq = compq == 'N' ? 1 : max(1, stride(Q, 2))
            ldz = compz == 'N' ? 1 : max(1, stride(Z, 2))
            work = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            rwork = Vector{$relty}(undef, n)
            info = Ref{BlasInt}()
            for i = 1:2  # first call returns lwork as work[1]
                ccall((@blasfunc($hgeqz), liblapack), Cvoid,
                      (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                       Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ptr{$elty},  
                       Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, 
                       Ptr{$elty}, Ref{BlasInt}, Ptr{$relty}, 
                       Ptr{BlasInt}, Clong, Clong, Clong),
                    'S', compq, compz, n, ilo, ihi, 
                    H, ldh, T, ldt, alpha, beta, 
                    Q, ldq, Z, ldz, 
                    work, lwork, rwork,
                    info, 1, 1, 1)
                chklapackerror(info[])
                if i == 1
                    lwork = BlasInt(real(work[1]))
                    resize!(work, lwork)
                end
            end
            H, T, alpha, beta, Q, Z
        end
    end
end

for (tgexc, elty) in ((:dtgexc_, :Float64), (:stgexc_, :Float32))
    @eval begin
        #
        #       SUBROUTINE DTGEXC( WANTQ, WANTZ, N, A, LDA, B, LDB, Q, LDQ, Z,
        #                          LDZ, IFST, ILST, WORK, LWORK, INFO )
        #
        #       .. Scalar Arguments ..
        #       LOGICAL            WANTQ, WANTZ
        #       INTEGER            IFST, ILST, INFO, LDA, LDB, LDQ, LDZ, LWORK, N
        #       ..
        #       .. Array Arguments ..
        #       DOUBLE PRECISION   A( LDA, * ), B( LDB, * ), Q( LDQ, * ),
        #      $                   WORK( * ), Z( LDZ, * )
        function tgexc!(wantq::Bool, wantz::Bool,  ifst::BlasInt, ilst::BlasInt, 
                        S::AbstractMatrix{$elty}, T::AbstractMatrix{$elty}, Q::AbstractMatrix{$elty}, Z::AbstractMatrix{$elty})
            chkstride1(S, T, Q, Z)
            n, nt, nq, nz = checksquare(S, T, Q, Z)
            n == nt || throw(DimensionMismatch("dimensions of S, ($n,$n), and T, ($nt,$nt), must match"))
            n == nq || throw(DimensionMismatch("dimensions of S, ($n,$n), and Q, ($nq,$nq), must match"))
            n == nz || throw(DimensionMismatch("dimensions of S, ($n,$n), and Z, ($nz,$nz), must match"))
            (ifst < 1 || ifst > n || ilst < 1 || ilst > n) && 
                    throw(ArgumentError("ifst and ilst arguments must satisfy 1 ≤ ifst, ilst ≤ $n, got ifst = $ifst and ilst = $ilst"))
            lds = max(1, stride(S, 2))
            ldt = max(1, stride(T, 2))
            ldq = wantq ? max(1, stride(Q, 2)) : 1
            ldz = wantz ? max(1, stride(Z, 2)) : 1
            lwork = BlasInt(-1)
            work = Vector{$elty}(undef, 1)
            info = Ref{BlasInt}()
            for i = 1:2  # first call returns lwork as work[1] 
                ccall((@blasfunc($tgexc), liblapack), Cvoid,
                      (Ref{BlasInt},  Ref{BlasInt},  Ref{BlasInt},
                      Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                      Ref{BlasInt}, Ref{BlasInt},
                      Ptr{$elty}, Ref{BlasInt},
                      Ptr{BlasInt}),
                      BlasInt(wantq), BlasInt(wantz), n, 
                      S, lds, T, ldt, Q, ldq, Z, ldz, 
                      ifst, ilst, 
                      work, lwork,
                      info)
                chklapackerror(info[])
                if i == 1 # only estimated optimal lwork
                    lwork  = BlasInt(real(work[1]))
                    resize!(work, lwork)
                end
            end
            return S, T, Q, Z
        end
    end
end
for (tgexc, elty) in ((:ztgexc_, :ComplexF64), (:ctgexc_, :ComplexF32))
    @eval begin
        #
        #       SUBROUTINE ZTGEXC( WANTQ, WANTZ, N, A, LDA, B, LDB, Q, LDQ, Z,
        #                          LDZ, IFST, ILST, INFO )
        #
        #       .. Scalar Arguments ..
        #       LOGICAL            WANTQ, WANTZ
        #       INTEGER            IFST, ILST, INFO, LDA, LDB, LDQ, LDZ, N
        #       ..
        #       .. Array Arguments ..
        #       COMPLEX*16   A( LDA, * ), B( LDB, * ), Q( LDQ, * ), Z( LDZ, * )
        function tgexc!(wantq::Bool, wantz::Bool,  ifst::BlasInt, ilst::BlasInt, 
                        S::AbstractMatrix{$elty}, T::AbstractMatrix{$elty}, Q::AbstractMatrix{$elty}, Z::AbstractMatrix{$elty})
            chkstride1(S, T, Q, Z)
            n, nt, nq, nz = checksquare(S, T, Q, Z)
            n == nt || throw(DimensionMismatch("dimensions of S, ($n,$n), and T, ($nt,$nt), must match"))
            n == nq || throw(DimensionMismatch("dimensions of S, ($n,$n), and Q, ($nq,$nq), must match"))
            n == nz || throw(DimensionMismatch("dimensions of S, ($n,$n), and Z, ($nz,$nz), must match"))
            (ifst < 1 || ifst > n || ilst < 1 || ilst > n) && 
                    throw(ArgumentError("ifst and ilst arguments must satisfy 1 ≤ ifst, ilst ≤ $n, got ifst = $ifst and ilst = $ilst"))
            lds = max(1, stride(S, 2))
            ldt = max(1, stride(T, 2))
            ldq = wantq ? max(1, stride(Q, 2)) : 1
            ldz = wantz ? max(1, stride(Z, 2)) : 1
            info = Ref{BlasInt}()
            ccall((@blasfunc($tgexc), liblapack), Cvoid,
                  (Ref{BlasInt},  Ref{BlasInt},  Ref{BlasInt},
                   Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                    Ref{BlasInt}, Ref{BlasInt},
                    Ptr{BlasInt}),
                    BlasInt(wantq), BlasInt(wantz), n, 
                    S, lds, T, ldt, Q, ldq, Z, ldz, 
                    ifst, ilst,
                    info)
            chklapackerror(info[])
            return S, T, Q, Z
        end
    end
end

"""
    tgexc!(wantq, wantz, ifst, ilst, S, T, Q, Z) -> (S, T, Q, Z)

Reorder the generalized Schur factorization of a matrix pair (S,T) in generalized Schur form, 
so that the diagonal blocks of (S, T) with row index `ifst` are moved to row `ilst`. 
If `wantq = true`, the left Schur vectors `Q` are reordered, and if `wantq = false` they are not modified. 
If `wantz = true`, the right Schur vectors `Z` are reordered, and if `wantz = false` they are not modified. 
"""
tgexc!(wantq::Bool, wantz::Bool, ifst::BlasInt, ilst::BlasInt, 
       S::AbstractMatrix, T::AbstractMatrix, Q::AbstractMatrix, Z::AbstractMatrix)

for (tgsen, elty) in
    ((:dtgsen_, :Float64),
     (:stgsen_, :Float32))
    @eval begin
        #        .. Scalar Arguments ..
        #        LOGICAL            WANTQ, WANTZ
        #        INTEGER            IJOB, INFO, LDA, LDB, LDQ, LDZ, LIWORK, LWORK,
        #       $                   M, N
        #        DOUBLE PRECISION   PL, PR
        #        ..
        #        .. Array Arguments ..
        #        LOGICAL            SELECT( * )
        #        INTEGER            IWORK( * )
        #        DOUBLE PRECISION   A( LDA, * ), ALPHAI( * ), ALPHAR( * ),
        #       $                   B( LDB, * ), BETA( * ), DIF( * ), Q( LDQ, * ),
        #       $                   WORK( * ), Z( LDZ, * )
        #        ..
        function tgsen!(wantq::Bool, wantz::Bool, select::AbstractVector{BlasInt}, S::AbstractMatrix{$elty}, T::AbstractMatrix{$elty},
                        Q::AbstractMatrix{$elty}, Z::AbstractMatrix{$elty})
            chkstride1(select, S, T, Q, Z)
            n, nt, nq, nz = checksquare(S, T, Q, Z)
            n == nt || throw(DimensionMismatch("dimensions of S, ($n,$n), and T, ($nt,$nt), must match"))
            n == nq || throw(DimensionMismatch("dimensions of S, ($n,$n), and Q, ($nq,$nq), must match"))
            n == nz || throw(DimensionMismatch("dimensions of S, ($n,$n), and Z, ($nz,$nz), must match"))
            lds = max(1, stride(S, 2))
            ldt = max(1, stride(T, 2))
            ldq = max(1, stride(Q, 2))
            ldz = max(1, stride(Z, 2))
            m = sum(select)
            alphai = similar(T, $elty, n)
            alphar = similar(T, $elty, n)
            beta = similar(T, $elty, n)
            lwork = BlasInt(-1)
            work = Vector{$elty}(undef, 1)
            liwork = BlasInt(-1)
            iwork = Vector{BlasInt}(undef, 1)
            info = Ref{BlasInt}()
            select = convert(Array{BlasInt}, select)
            for i = 1:2  # first call returns lwork as work[1] and liwork as iwork[1]
                ccall((@blasfunc($tgsen), liblapack), Cvoid,
                       (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{BlasInt},
                        Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                        Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ptr{$elty},
                        Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                        Ref{BlasInt}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid},
                        Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt},
                        Ptr{BlasInt}),
                    0, BlasInt(wantq), BlasInt(wantz), select,
                    n, S, lds, T,
                    ldt, alphar, alphai, beta,
                    Q, ldq, Z, ldz,
                    m, C_NULL, C_NULL, C_NULL,
                    work, lwork, iwork, liwork,
                    info)
                chklapackerror(info[])
                if i == 1 # only estimated optimal lwork, liwork
                    lwork  = BlasInt(real(work[1]))
                    resize!(work, lwork)
                    liwork = BlasInt(real(iwork[1]))
                    resize!(iwork, liwork)
                end
            end
            S, T, complex.(alphar, alphai), beta, Q, Z
        end
    end
end
for (tgsen, elty, relty) in
    ((:ztgsen_, :ComplexF64, :Float64),
     (:ctgsen_, :ComplexF32, :Float32))
    @eval begin
        function tgsen!(wantq::Bool, wantz::Bool, select::AbstractVector{BlasInt}, S::AbstractMatrix{$elty}, T::AbstractMatrix{$elty},
                        Q::AbstractMatrix{$elty}, Z::AbstractMatrix{$elty})
            chkstride1(select, S, T, Q, Z)
            n, nt, nq, nz = checksquare(S, T, Q, Z)
            n == nt || throw(DimensionMismatch("dimensions of S, ($n,$n), and T, ($nt,$nt), must match"))
            n == nq || throw(DimensionMismatch("dimensions of S, ($n,$n), and Q, ($nq,$nq), must match"))
            n == nz || throw(DimensionMismatch("dimensions of S, ($n,$n), and Z, ($nz,$nz), must match"))
            lds = max(1, stride(S, 2))
            ldt = max(1, stride(T, 2))
            ldq = max(1, stride(Q, 2))
            ldz = max(1, stride(Z, 2))
            m = sum(select)
            alpha = similar(T, $elty, n)
            beta = similar(T, $elty, n)
            lwork = BlasInt(-1)
            work = Vector{$elty}(undef, 1)
            liwork = BlasInt(-1)
            iwork = Vector{BlasInt}(undef, 1)
            info = Ref{BlasInt}()
            select = convert(Array{BlasInt}, select)
            for i = 1:2  # first call returns lwork as work[1] and liwork as iwork[1]
                ccall((@blasfunc($tgsen), liblapack), Cvoid,
                       (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{BlasInt},
                        Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                        Ref{BlasInt}, Ptr{$elty}, Ptr{$elty},
                        Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                        Ref{BlasInt}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid},
                        Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt},
                        Ptr{BlasInt}),
                    0, BlasInt(wantq), BlasInt(wantz), select,
                    n, S, lds, T,
                    ldt, alpha, beta,
                    Q, ldq, Z, ldz,
                    m, C_NULL, C_NULL, C_NULL,
                    work, lwork, iwork, liwork,
                    info)
                chklapackerror(info[])
                if i == 1 # only estimated optimal lwork, liwork
                    lwork  = BlasInt(real(work[1]))
                    resize!(work, lwork)
                    liwork = BlasInt(real(iwork[1]))
                    resize!(iwork, liwork)
                end
            end
            S, T, alpha, beta, Q, Z
        end
    end
end
"""
    tgsen!(wantq, wantz, select, S, T, Q, Z) -> (S, T, Q, Z)

Reorder the diagonal 1x1 and 2x2 blocks of a matrix pair (S,T) in generalized Schur form, such that 
their eigenvalues correspond to the cluster specified in `select`.
If `wantq = true`, the left Schur vectors `Q` are reordered, and if `wantq = false` they are not modified. 
If `wantz = true`, the right Schur vectors `Z` are reordered, and if `wantz = false` they are not modified. 
"""
tgsen!(wantq::Bool, wantz::Bool, select::AbstractVector, S::AbstractMatrix, T::AbstractMatrix, Q::AbstractMatrix, Z::AbstractMatrix)


for (fn, elty, relty) in ((:dtgsyl_, :Float64, :Float64),
                   (:stgsyl_, :Float32, :Float32),
                   (:ztgsyl_, :ComplexF64, :Float64),
                   (:ctgsyl_, :ComplexF32, :Float32))
    @eval begin
        function tgsyl!(trans::AbstractChar, A::AbstractMatrix{$elty}, B::AbstractMatrix{$elty}, C::AbstractMatrix{$elty},
                        D::AbstractMatrix{$elty}, E::AbstractMatrix{$elty}, F::AbstractMatrix{$elty})
            @assert !has_offset_axes(A, B, C, D, E, F)
            chkstride1(A, B, C, D, E, F)
            m, n = checksquare(A, B)
            lda = max(1, stride(A, 2))
            ldb = max(1, stride(B, 2))
            ldc = max(1, stride(C, 2))
            m1, n1 = size(C)
            (m == m1 && n == n1) ||
                throw(DimensionMismatch("dimensions of A($m,$m),  B($n,$n), and C($m1,$n1) must match"))
            m2, n2 = checksquare(D, E)
            m == m2 || throw(DimensionMismatch("dimensions of A($m,$m) and D($m2,$m2) must match"))
            n == n2 || throw(DimensionMismatch("dimensions of B($n,$n) and E($n2,$n2), must match"))
            ldd = max(1, stride(D, 2))
            lde = max(1, stride(E, 2))
            ldf = max(1, stride(F, 2))
            m3, n3 = size(F)
            (m2 == m3 && n2 == n3) ||
                throw(DimensionMismatch("dimensions of D($m,$m),  E($n,$n), and F($m3,$n3) must match"))
            dif = Vector{$relty}(undef, 1)
            scale = Vector{$relty}(undef, 1)
            info  = Ref{BlasInt}()
            ijob = 0
            work = Vector{$elty}(undef, 1)
            lwork = 1
            iwork = Vector{BlasInt}(undef,m+n+6)
            #SUBROUTINE DTGSYL( TRANS, IJOB, M, N, A, LDA, B, LDB, C, LDC, D,
            #       LDD, E, LDE, F, LDF, SCALE, DIF, WORK, LWORK,
            #       IWORK, INFO )
            ccall((@blasfunc($fn), liblapack), Cvoid,
                (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                 Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                 Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                 Ptr{$relty}, Ptr{$relty}, Ptr{$relty}, Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}, Clong),
                 trans, ijob, m, n,
                 A, lda, B, ldb, C, ldc, D, ldd, E, lde, F, ldf,
                 scale, dif, work, lwork, iwork, info, 1)
            chklapackerror(info[])
            C, F, scale[1]
        end
    end
end

"""
    tgsyl!(A, B, C, D, E, F) -> (C, F, scale)

Solve the Sylvester system of matrix equations

      AX - YB = scale*C
      DX - YE = scale*F ,

where `X` and `Y` are unknown matrices, the pairs `(A, D)`, `(B, E)` and  `(C, F)`
have the same sizes, and the pairs `(A, D)` and `(B, E)` are in
generalized (real) Schur canonical form, i.e. `A`, `B` are upper quasi
triangular and `D`, `E` are upper triangular.
Returns `X` (overwriting `C`), `Y` (overwriting `F`) and `scale`.

    tgsyl!(trans, A, B, C, D, E, F) -> (C, F, scale)

Solve for `trans = 'T'` and real matrices or for `trans = 'C'` and complex
matrices,  the (adjoint) Sylvester system of matrix equations

      A'X + D'Y = scale*C
      XB' + YE' = scale*(-F) .

`tgsyl!('N', A, B, C, D, E, F)` corresponds to the call `tgsyl!(A, B, C, D, E, F)`.

Interface to the LAPACK subroutines DTGSYL/STGSYL/ZTGSYL/CTGSYL.
"""
tgsyl!(trans::AbstractChar,A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix, E::AbstractMatrix, F::AbstractMatrix)

tgsyl!(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix, E::AbstractMatrix, F::AbstractMatrix) =
tgsyl!('N',A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix, E::AbstractMatrix, F::AbstractMatrix)

end # Module
