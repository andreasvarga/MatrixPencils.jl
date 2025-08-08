import Base: *, eltype, size
import LinearAlgebra: adjoint, mul!, rmul!, lmul!

# Householder reflectors suite: computation and applications from left and right;
# adapted from Ralph Smith's GenericSchur.jl and Andreas Noack's GenericLinearAlgebra.jl

function _reflector!(x::AbstractVector{T},first::Bool = true) where {T<:Real}
    # allow "First Element Pivot" or "Last Element Pivot"
    require_one_based_indexing(x)
    n = length(x)
    n <= 1 && return zero(T)
    sfmin = 2floatmin(T) / eps(T)
    @inbounds begin
        α = first ? x[1] : x[end] 
        v = first ? view(x,2:n) : view(x,1:n-1)
        xnorm = LinearAlgebra.norm(v)
        if iszero(xnorm)
            return zero(T)
        end
        β = -copysign(hypot(α, xnorm), α)
        kount = 0
        smallβ = abs(β) < sfmin
        if smallβ
            # recompute xnorm and β if needed for accuracy
            rsfmin = one(T) / sfmin
            while smallβ
                kount += 1
                for j in 1:n-1
                    v[j] *= rsfmin
                end
                β *= rsfmin
                α *= rsfmin
                # CHECKME: is 20 adequate for BigFloat?
                smallβ = (abs(β) < sfmin) && (kount < 20)
            end
            # now β ∈ [sfmin,1]
            xnorm = LinearAlgebra.norm(v)
            β = -copysign(hypot(α, xnorm), α)
        end
        τ = (β - α) / β
        t = one(T) / (α - β)
        for j in 1:n-1
            v[j] *= t
        end
        for j in 1:kount
            β *= sfmin
        end
        first ? x[1] = β : x[end] = β
    end
    return τ 
end

function _reflector!(x::AbstractVector{T}, first::Bool = true) where {T<:Complex}
    require_one_based_indexing(x)
    n = length(x)
    # we need to make subdiagonals real so the n=1 case is nontrivial for complex eltype
    n < 1 && return zero(T)
    RT = real(T)
    sfmin = floatmin(RT) / eps(RT)
    @inbounds begin
        α = first ? x[1] : x[end]
        v = first ? view(x,2:n) : view(x,1:n-1)
        αr, αi = reim(α)
        xnorm = LinearAlgebra.norm(v)

        if iszero(xnorm) && iszero(αi)
            return zero(T)
        end
        β = -copysign(_hypot3(αr, αi, xnorm), αr)
        #β = -copysign(hypot3(αr, αi, xnorm), αr)
        kount = 0
        smallβ = abs(β) < sfmin
        if smallβ
            # recompute xnorm and β if needed for accuracy
            rsfmin = one(real(T)) / sfmin
            while smallβ
                kount += 1
                for j in 1:n-1
                    v[j] *= rsfmin
                end
                β *= rsfmin
                αr *= rsfmin
                αi *= rsfmin
                smallβ = (abs(β) < sfmin) && (kount < 20)
            end
            # now β ∈ [sfmin,1]
            xnorm = LinearAlgebra.norm(v)
            α = complex(αr, αi)
            β = -copysign(_hypot3(αr, αi, xnorm), αr)
            #β = -copysign(hypot3(αr, αi, xnorm), αr)
        end
        τ = complex((β - αr) / β, -αi / β)
        t = one(T) / (α - β)
        for j in 1:n-1
            v[j] *= t
        end
        for j in 1:kount
            β *= sfmin
        end
        first ? x[1] = β : x[end] =  β
    end
    return τ
end
function _hypot3(x::T, y::T, z::T) where {T}
    xa = abs(x)
    ya = abs(y)
    za = abs(z)
    w = max(xa, ya, za)
    rw = one(real(T)) / w
    r::real(T) = w * sqrt((rw * xa)^2 + (rw * ya)^2 + (rw * za)^2)
    return r
end


"""
Householder reflection represented as the essential part of the
vector, the normalizing factor and pivot position information (first or last)
"""
struct Householder{T,S<:AbstractVector}
    v::S
    τ::T
    first::Bool
    function Householder(v::AbstractVector{T}, τ::T, first::Bool = true) where {T}
       new{T,typeof(v)}(v,τ,first)
    end
end

Base.size(H::MatrixPencils.Householder) = (length(H.v)+1, length(H.v)+1)
Base.size(H::MatrixPencils.Householder, i::Integer) = i <= 2 ? length(H.v)+1 : 1
Base.length(H::MatrixPencils.Householder) = 1

Base.eltype(H::MatrixPencils.Householder{T}) where {T} = T

LinearAlgebra.adjoint(H::MatrixPencils.Householder{T}) where {T} = Adjoint{T,typeof(H)}(H)

Base.convert(::Type{Matrix}, H::MatrixPencils.Householder{T}) where {T} = lmul!(H, Matrix{T}(I, size(H, 1), size(H, 1)))
Base.convert(::Type{Matrix{T}}, H::MatrixPencils.Householder{T}) where {T} = lmul!(H, Matrix{T}(I, size(H, 1), size(H, 1)))
Base.convert(::Type{Matrix}, H::Adjoint{<:Any,<:MatrixPencils.Householder{T}}) where {T} = lmul!(H, Matrix{T}(I, size(parent(H), 1), size(parent(H), 1)))
Base.convert(::Type{Matrix{T}}, H::Adjoint{<:Any,<:MatrixPencils.Householder{T}}) where {T} = lmul!(H, Matrix{T}(I, size(parent(H), 1), size(parent(H), 1)))



function LinearAlgebra.lmul!(H::MatrixPencils.Householder, A::AbstractMatrix)
    m, n = size(A)
    size(H,1) == m || throw(DimensionMismatch("A: $m,$n H: $(size(H))"))
    v = view(H.v, :)
    τ = H.τ
    if H.first
       @inbounds begin
        for j = 1:n
           va = A[1,j]
           Aj = view(A, 2:m, j)
           va += dot(v, Aj)
           va = τ*va
           A[1,j] -= va
           axpy!(-va, v, Aj)
        end
       end
    else
       @inbounds begin
        for j = 1:n
           va = A[m,j]
           Aj = view(A, 1:m-1, j)
           va += dot(v, Aj)
           va = τ*va
           A[m,j] -= va
           axpy!(-va, v, Aj)
        end
       end
    end
    A
end

function LinearAlgebra.lmul!(adjH::Adjoint{<:Any,<:MatrixPencils.Householder}, A::AbstractMatrix)
    H = parent(adjH)
    m, n = size(A)
    size(H,1) == m || throw(DimensionMismatch("A: $m,$n H: $(size(H))"))
    v = view(H.v, :)
    τ = H.τ
    if H.first
       @inbounds begin
        for j = 1:n
           va = A[1,j]
           Aj = view(A, 2:m, j)
           va += dot(v, Aj)
           va = τ'*va
           A[1,j] -= va
           axpy!(-va, v, Aj)
        end
       end
    else
       @inbounds begin
        for j = 1:n
           va = A[m,j]
           Aj = view(A, 1:m-1, j)
           va += dot(v, Aj)
           va = τ'*va
           A[m,j] -= va
           axpy!(-va, v, Aj)
        end
       end
    end
    A
end

function LinearAlgebra.rmul!(A::AbstractMatrix, H::MatrixPencils.Householder) 
    m, n = size(A)
    size(H,2) == n || throw(DimensionMismatch("A: $m,$n H: $(size(H))"))
    v = view(H.v, :)
    τ = H.τ
    if H.first
       @inbounds begin
        for i = 1:m
            Aiv = A[i, 1]
            for j = 2:n
                Aiv += A[i, j] * v[j-1]
            end
            Aiv = Aiv * τ
            A[i, 1] -= Aiv
            for j = 2:n
                A[i, j] -= Aiv * v[j-1]'
            end
        end
       end
    else
       @inbounds begin
        for i = 1:m
            Aiv = A[i, n]
            for j = 1:n-1
                Aiv += A[i, j] * v[j]
            end
            Aiv = Aiv * τ
            A[i, n] -= Aiv
            for j = 1:n-1
                A[i, j] -= Aiv * v[j]'
            end
        end
       end
    end
    return A
end

