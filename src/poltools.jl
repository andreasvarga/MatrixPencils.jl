function qrsolve!(A::AbstractMatrix{T},b::AbstractVector{T}) where {T}
    # Fast least-squares solver for full column rank Hessenberg-like matrices
    #VERSION >= v"1.2.0" && require_one_based_indexing(A)
    m, n = size(A) 
    m < n && error("Column dimension exceeds row dimension") 
    _, τ = LinearAlgebra.LAPACK.geqrf!(A)
    T <: Complex ? tran = 'C' : tran = 'T'
    LinearAlgebra.LAPACK.ormqr!('L',tran,A,τ,view(b,:,1:1))
    return UpperTriangular(triu(A[1:n,:]))\b[1:n]
end

function poldeg1(v::Vector{T}) where {T <: Number}
    # Degree of a polynomial plus one.
    n1 = findlast(!iszero,v) # degree(v) + 1
    n1 === nothing && (n1 = 0)
    return n1
end
poldeg(v::Vector{T}) where {T <: Number} = poldeg1(v)-1  # Degree of a polynomial
function poldiv(a::Vector{T}, b::Vector{S}) where {T <: Number, S <: Number }
    # Quotient of the exact division of two polynomials a and b.
    # q = poldiv(a,b) returns the quotient q = a/b, which is computed by solving the 
    # linear least-square problem ||C(b)*q - a|| = min, where C(b) is the convolution 
    # matrix of appropriate size.  

    na1 = poldeg1(a) # degree(a) + 1
    nb1 = poldeg1(b) # degree(b) + 1
    nb1 == 0 && throw(DivideError())
    na1 < nb1 && error("Degree of second polynomial exceeds the degree of first polynomial")
    a1, b1  = promote(float(a[1:na1]), float(b[1:nb1]))
    #return Matrix(Toeplitz([den;zeros(n1-length(den))],[den[1];zeros(n1-m1)]))\num[1:n1] 
    #return convmtx(b[1:nb1],na1-nb1+1)\a[1:na1] 
    return qrsolve!(convmtx(b1,na1-nb1+1),a1) 
end
function convmtx(v::Vector{T}, n::Int) where {T <: Number}

    #   Convolution matrix.
    #   C = convmtx(v,n) returns the convolution matrix C for a vector v. 
    #   If q is a column vector of length n, then C*q is the same as conv(v,q). 
    
    #   Form C as the Toeplitz matrix 
    #   C = Toeplitz([v; zeros(n-1)],[v[1]; zeros(n-1));  put Toeplitz code inline
    
    nv = length(v)-1
    C = zeros(T, n+nv, n)
    @inbounds for j = 1:n
        C[j:j+nv,j] = v  
    end
    return C
end
function poldivrem(a::Vector{T}, b::Vector{S}) where {T <: Number, S <: Number }
    # Quotient and remainder of the division of two polynomials a and b.
    # q, r = poldivrem(a,b) returns the quotient q and remainder r such that a = q*b + r.
    na1 = poldeg1(a) # degree(a) + 1
    nb1 = poldeg1(b) # degree(b) + 1
    nb1 == 0 && throw(DivideError())
    
    R = eltype(one(T)/one(S))

    if na1 < nb1
       return zeros(R, 1), a
    end
    m = nb1-1
    q = zeros(R, na1-m)
    r = R[ a[i] for i in 1:na1 ]
    #r = copy_oftype(num[1:n1],R)

    @inbounds for i in na1:-1:nb1
        s = r[i] / b[nb1]
        q[i-m] = s
        @inbounds for j in 1:m
            r[i-nb1+j] -= b[j] * s
        end
    end

    return q, m == 0 ? zeros(R,1) : r[1:m]   
end
function polgcdvw(a::Vector{T}, b::Vector{S}; atol::Real = 0, rtol::Real = Base.rtoldefault(float(real(T))), maxnit::Int = 0) where {T <: Number, S <: Number }
    # Greatest common divisor of two polynomials.
    #     polgcdvw(a, b; atol = 0, rtol, maxnit = 0) -> (d, v, w, δ)
    # Compute the greatest common divisor `d` of two polynomials `a` and `b`, and 
    # the polynomials v and w such that a = d * v * ka and b = d * w *kb, where ka and kb
    # are scalar scaling factors (ka and kb are not explicitly provided). 
    # The accuracy estimate δ is computed as δ = norm([a/ka-d*v; b/kb-d*w]). 
    # A SVD-based method adapted from [1] is employed and an iterative accuracy refinement
    # of δ, with a maximum number of maxnit iterations, is additionally performed.
    #  [1] Z. Zeng, The numerical greatest common divisor of univariate polynomials, 
    #      in: L. Gurvits, P. Pébay, J.M. Rojas, D. Thompson (Eds.), 
    #      Randomization, Relaxation,and Complexity in Polynomial Equation Solving, 
    #      Contemporary Mathematics,vol.556,AMS,2011,pp.187–217.
    
    na1 = poldeg1(a) # degree(a) + 1
    nb1 = poldeg1(b) # degree(b) + 1

    R = eltype(one(T)/one(S))

    a1, b1  = promote(float(a[1:na1]), float(b[1:nb1]))

    na1 <= 0 && (return ones(R,1), zeros(R,1), b1, zero(R))
    nb1 <= 0 && (return ones(R,1), a1, zeros(R,1), zero(R))
    a1 = a1/norm(a1)
    b1 = b1/norm(b1)
    switch = (na1 < nb1)
    switch && ((a1, b1, na1, nb1) = (b1, a1, nb1, na1))

    atol == 0 ? tol = rtol : tol = atol
    # determine the degree of GCD as the nullity of the Sylvester matrix 
    nd = na1 + nb1 - 2 - rank([convmtx(a1,nb1-1) convmtx(b1,na1-1)], atol = tol)
    nd == 0 && (switch ? (return [one(R)], b1, a1, zero(R)) : (return [one(R)], a1, b1, zero(R))) 
    # determine [w; -v] from an orthogonal/unitary nullspace basis of dimension 1 
    # of a reduced Sylvester matrix using the last row of Vt from its (partial) SVD 
    _, sv, Vt = LAPACK.gesvd!('N', 'A', [convmtx(a1,nb1-nd) convmtx(b1,na1-nd)])
    k = na1 + nb1 -2*nd - count(x -> x > tol, sv) # expected nullity k = 1
    k == 1 || error("GCD computation failure")
    wv = Vt[end,:] 
    # wv = LAPACK.gesvd!('N', 'A', [convmtx(a1,nb1-nd) convmtx(b1,na1-nd)])[3][end,:] 
    eltype(wv) <: Complex && (wv = conj(wv))  
    v = -wv[nb1-nd+1:end]
    w = wv[1:nb1-nd]
    # determine the GCD d with high accuracy as the solution of a well-conditioned 
    # linear least-squares problem
    # d = convmtx(w,nd+1)\b1 
    # d = [convmtx(v,nd+1) ; convmtx(w,nd+1)] \ [a1; b1]
    d = qrsolve!([convmtx(v,nd+1) ; convmtx(w,nd+1)],[a1; b1])
    if maxnit > 0
       d, v, w, δ = gcdvwupd(a1, b1, d, v, w, maxnit = maxnit)
    else
       δ = norm( [a1; b1] - [ conv(v,d); conv(w,d) ])
    end

    switch ? (return d, w, v, δ) : (return d, v, w, δ)
end 
function gcdvwupd(a::Vector{T}, b::Vector{T}, d::Vector{T}, v::Vector{T}, w::Vector{T}; maxnit::Int = 10) where {T <: Number}
    # Iterative refinement of the accuracy of the greatest common divisor of two polynomials.
    #     gcdvwupd(a, b, d, v, w; maxnit = 0) -> (dupd, vupd, wupd, δ)
    # Given the polynomials a and b, and a triple of polynomials (d, v, w) such that 
    # `d` is an approximation of a greatest common divisor of `a` and `b`, and 
    # v and w are approximate quotients of a/d and b/d, respectively, compute the updated triple 
    # (dupd, vupd, wupd) which minimizes the accuracy estimate δ = norm([a-dupd*vupd; b-dupd*wupd]).
    # A maximum number of maxnit iterations are performed using the Gauss-Newton iteration method, 
    # in the form proposed in [1].
    #
    #  [1] Z. Zeng, The numerical greatest common divisor of univariate polynomials, 
    #      in: L. Gurvits, P. Pébay, J.M. Rojas, D. Thompson (Eds.), 
    #      Randomization, Relaxation,and Complexity in Polynomial Equation Solving, 
    #      Contemporary Mathematics,vol.556,AMS,2011,pp.187–217.

    na1 = length(a) 
    nb1 = length(b) 
    nd1 = length(d)
    nv1 = length(v)
    nw1 = length(w)
    na1 == nd1+nv1-1 || error("Incompatible dimensions between a, d and v")
    nb1 == nd1+nw1-1 || error("Incompatible dimensions between b, d and w")

    h = d/norm(d)^2
    f = [a;b;one(T)]
    fh = [conv(v,d); conv(w,d); dot(h,d)]
    Δf = f-fh;
    δ = norm(Δf)
    d0 = copy(d); v0 = copy(v); w0 = copy(w);
    nvw1 = nv1+nw1
    ndvw1 = nd1+nvw1 
    nab1 = na1+nb1
    Jh = [zeros(nab1,ndvw1); zeros(1,nvw1) h' ]

    j1 = 1:nv1
    j2 = nv1+1:nvw1
    j3 = nvw1+1:ndvw1
    i1 = 1:nv1+nd1-1
    i2 = nv1+nd1:nvw1+2*nd1-2
    Jh11 = view(Jh,i1,j1)
    Jh22 = view(Jh,i2,j2)
    Jh13 = view(Jh,i1,j3)
    Jh23 = view(Jh,i2,j3)

    
    for it = 1:maxnit
        # Jh = [ convmtx(d0,nv1)          0     convmtx(v0,nd1) ] 
        #      [  0             convmtx(u0,nw1) convmtx(w0,nd1) ]
        #      [  0                       0            h'       ] 
        Jh11[:,:] = convmtx(d0,nv1)
        Jh22[:,:] = convmtx(d0,nw1)
        Jh13[:,:] = convmtx(v0,nd1) 
        Jh23[:,:] = convmtx(w0,nd1) 
        # Δz = Jh\Δf;   # this does not exploit the zero structure
        Δz = qrsolve!(copy(Jh),copy(Δf));  # this is probably the fastest
        # Δz = qr(Jh)\Δf;  # this is usually slower, but faster for full matrices
        v1 = v0 + Δz[j1];
        w1 = w0 + Δz[j2];
        d1 = d0 + Δz[j3];
        fh1 = [conv(v1,d1); conv(w1,d1); dot(h,d1)];
        Δf1 = f-fh1;
        δ1 = norm(Δf1)
        if δ1 < δ
           d0 = copy(d1); v0 = copy(v1); w0 = copy(w1); δ = copy(δ1); Δf = copy(Δf1);
        else
           break
        end
    end
    return d0, v0, w0, δ 
end
function pollcm(a::Vector{T}, b::Vector{S}; atol::Real = 0, rtol::Real = Base.rtoldefault(float(real(T))), maxnit::Int = 10) where {T <: Number, S <: Number }
    # Least common multiple of two polynomials.
    #     m = pollcm(a,b; atol = 0, rtol, maxnit = 0) 
    # Compute the least common multiple `m` of two polynomials `a` and `b`. Employ iterative refinement of
    # the accuracy of GCD(a,b) if maxnit > 0. 
      
    na1 = poldeg1(a) # degree(a) + 1
    nb1 = poldeg1(b) # degree(b) + 1
    
    na1 == 0 && (return zeros(T,1))
    nb1 == 0 && (return zeros(S,1))
    na1 == 1 && (return b)
    nb1 == 1 && (return a)
     
    return conv(a, polgcdvw(a, b, atol = atol, rtol = rtol, maxnit = maxnit)[3])
end    
function conv(a::Vector{T}, b::Vector{S}) where {T <: Number, S <: Number }
    # Convolution of two vectors (or product of two polynomials).
    # c = conv(a,b) returns the convolution of vectors a and b.

    na1 = length(a) 
    nb1 = length(b) 

    R = promote_type(T, S)
    c = zeros(R, na1 + nb1 - 1)
    for i in 1:na1, j in 1:nb1
        @inbounds c[i + j - 1] += a[i] * b[j]
    end
    return c
end   
function polcoeffval(r::AbstractVector{T},val::Number) where {T}
    # Compute the coefficients of a polynomial from its roots and evaluate the polynomial
    # for a given value of its argument. Both are equal to one for an empty vector r.
    T1 = promote_type(T,eltype(val))
    n = length(r)
    c = zeros(T1, n+1)
    ONE = one(T1)
    c[1] = ONE
    pval = ONE
    for j = 1:n
        pval = pval*(val-r[j])
        for i = j:-1:1
            c[i+1] = c[i+1]-r[j]*c[i]
        end
    end
    return reverse(c), pval
 end
