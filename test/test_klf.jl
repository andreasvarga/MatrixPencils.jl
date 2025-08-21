module Test_klf

using Random
using LinearAlgebra
using SparseArrays
using MatrixPencils
using Test
using GenericLinearAlgebra

   
Random.seed!(2351);
println("Test_klf")

@testset "Matrix Pencils Utilities" begin


@testset "pbalance!" begin

n = 10;  k = 20; 
for k in (10,20)

lambda = exp.(im.*rand(n));
la = real(lambda); li = imag(lambda); la[end] = 0; li[end] = 0;  
D = sort(la[1:end-1]./li[1:end-1])

La = Diagonal(la); Li = Diagonal(li); 
Tl = randn(n,n).^k; Tr = randn(n,n).^k;
A = Tl*La*Tr; E = Tl*Li*Tr;
atol1 = 1.e-4; atol2 = 1.e-4;
for iter = 1:10
    ev = sort(pzeros(A,E;atol1,atol2)[1],by=real)
    length(ev) == n-1 && break
    atol1 *= 10; atol2 *= 10
end
ev = sort(pzeros(A,E;atol1,atol2)[1],by=real)
corig = norm(abs.(ev-D)./sqrt.(1. .+ ev .^2)./sqrt.(1. .+ D .^2))
isfinite(corig) || (corig = Inf)

# if any(isinf.(ev))
#    corig = Inf
# else
#    corig = norm(abs.(ev-D)./sqrt.(1. .+ ev .^2)./sqrt.(1. .+ D .^2))
#    isfinite(corig) || (corig = Inf)
# end

qsorig = qS1(abs.(A)+abs.(E))
AA = copy(A); EE = copy(E); 
@time D1, D2 = regbalance!(AA,EE)
@test AA == D1*A*D2 && EE == D1*E*D2 
qsfin = qS1(abs.(AA)+abs.(EE))
@test qsfin < qsorig 
atol1 = 1.e-5; atol2 = 1.e-9;
for iter = 1:10
    evs = sort(pzeros(AA,EE;atol1,atol2)[1],by=real)
    length(evs) == n-1 && all(isfinite.(evs)) && break
    (!all(isfinite.(evs)) || length(evs) > n-1) && (atol1 *= 10; atol2 *= 10)
    (!all(isfinite.(evs)) || length(evs) < n-1) && (atol1 /= 10; atol2 /= 10)
end
evs = sort(pzeros(AA,EE;atol1,atol2)[1],by=real)
#@test evs ≈ D
# compute the chordal distance between exact and computed eigenvalues
cofin = norm(abs.(evs-D)./sqrt.(1. .+ evs .^2)./sqrt.(1. .+ D .^2))
#println("k = $k cofin = $cofin")
@test ev ≈ D ? cofin/10 < corig :  cofin < corig 
end


# Example 5.6 of Dopico et al. SIMAX, 2022
M = [1. 1 0; 1 0 0;0 0 1];
MM = copy(M);
dleft, dright = rcsumsbal!(MM,r = ones(3),c = ones(3))
@test dleft*M*dright ≈ MM
@test qS1(MM) < qS1(M)
# println("qs = $(qS1(M)) qsbal = $(qS1(MM))")

# diagonal regularization
M = [1. 1 0; 1 0 0;0 0 1];
α = 0.1
for α in (1.,0.5,0.1)
W = [α^2*I M; M' α^2*I] 
dleft1, dright1 = rcsumsbal!(W,r = ones(6),c = ones(6),tol=0.001)
Dl = Diagonal(dleft1.diag[1:3]); Dr = Diagonal(dright1.diag[4:6])
@test qS1(Dl*M*Dr) < qS1(M)
# println("α = $α  qs = $(qS1(Dl*M*Dr))")
end

# regularization of Dopico et al. SIMAX, 2022
M = [1. 1 0; 1 0 0;0 0 1];
qorig = qS1(M)
α = 0.1
for α in (1.,0.5,0.1)
W = [fill((α/3)^2,3,3) M; M' fill((α/3)^2,3,3)] 
dleft1, dright1 = rcsumsbal!(W,r = ones(6),c = ones(6),maxiter=1000,tol=0.001)
Dl = Diagonal(dleft1.diag[1:3]); Dr = Diagonal(dright1.diag[4:6])
@test qS1(Dl*M*Dr) < qorig
# println("α = $α  qs = $qorig qsbal = $(qS1(Dl*M*Dr))")
end


α = 0.1
for α in (1.,0.5,0.1)
M =  [-1. 0 0; 0 0 0;0 0 -1]; N =  [0. 1 0; 1 0 0;0 0 0];
MM = copy(M); NN = copy(N)
dleft, dright = pbalance!(MM, NN; maxiter=1000, tol=0.001, regpar=α, pow2 = false)
@test dleft*M*dright ≈ MM && dleft*N*dright ≈ NN
@test qS1(abs.(MM)+abs.(NN)) < qS1(abs.(M)+abs.(N)) 
# println("diagreg = false α = $α  qs = $(qS1(abs.(MM)+abs.(NN)))")

MM = copy(M); NN = copy(N)
dleft, dright = pbalance!(MM, NN; diagreg = true, maxiter=1000, tol=0.001, regpar=α, pow2 = false)
@test dleft*M*dright ≈ MM && dleft*N*dright ≈ NN
@test qS1(abs.(MM)+abs.(NN)) < qS1(abs.(M)+abs.(N)) 
# println("diagreg = true α = $α  qs = $(qS1(abs.(MM)+abs.(NN)))")

end

# Example 5.9 of Dopica et al. SIMAX, 2022
M = [1. 1 1; 0 0 1];
qorig = qS1(M)
MM = copy(M);
dleft, dright = rcsumsbal!(MM,r = 3*ones(2),c = 2*ones(3))
@test dleft*M*dright ≈ MM
@test qS1(MM) < qorig
# println("qs = $qorig qsbal = $(qS1(MM))")

M = [1. 1 1; 0 0 1];
qorig = qS1(M)
m, n = size(M)
α = 1.e-10
for α in (0.5,0.1,0.01,1.e-4,1.e-10)
W = [α^2*I M; M' α^2*I] 
WW = copy(W)
v = [3*ones(2);2*ones(3)]
dleft1, dright1 = rcsumsbal!(WW, r = v, c = v, maxiter=1000,tol=0.001)
Dl = Diagonal(dleft1.diag[1:m]); Dr = Diagonal(dright1.diag[m+1:m+n])
@test qS1(Dl*M*Dr) < qorig 
# println("α = $α  qs = $qorig qsbal = $(qS1(Dl*M*Dr))")
end


M = [1. 1 1; 0 0 1];
qorig = qS1(M)
m, n = size(M)
α = 1.e-10
for α in (0.5,0.1,0.01,1.e-4,1.e-10)
# W = [α*I M; M' α*I] 
# dleft, dright = rcsumsbal!(W, r = [3*ones(2);2*ones(3)], c = [3*ones(2);2*ones(3)],tol=0.001)
W = [fill((α/m)^2,m,m) M; M' fill((α/n)^2,n,n)] 
WW = copy(W)
v = [3*ones(2);2*ones(3)]
dleft1, dright1 = rcsumsbal!(WW,r = v, c = v, maxiter=1000,tol=0.001)
Dl = Diagonal(dleft1.diag[1:m]); Dr = Diagonal(dright1.diag[m+1:m+n])
@test qS1(Dl*M*Dr) < qorig 
# println("α = $α  qs = $qorig qsbal = $(qS1(Dl*M*Dr))")
end


α = 0.1
M =  [0. -1 -1; 0 0 0]; N =  [1. 0 0; 0 0 1];
r = 3. *ones(2); c = 2. *ones(3); 
for α in (0.5,0.1,0.01,1.e-4,1.e-10)
MM = copy(M); NN = copy(N)
dleft, dright = pbalance!(MM, NN; r, c, maxiter=1000, tol=0.001, regpar=α, pow2 = false)
@test dleft*M*dright ≈ MM && dleft*N*dright ≈ NN
@test qS1(abs.(MM)+abs.(NN)) < qS1(abs.(M)+abs.(N)) 
# println("diagreg = false α = $α  qs = $(qS1(abs.(MM)+abs.(NN)))")

MM = copy(M); NN = copy(N)
dleft, dright = pbalance!(MM, NN; diagreg = true, r, c, maxiter=1000, tol=0.001, regpar=α, pow2 = false)
@test dleft*M*dright ≈ MM && dleft*N*dright ≈ NN
@test qS1(abs.(MM)+abs.(NN)) < qS1(abs.(M)+abs.(N)) 
# println("diagreg = true α = $α  qs = $(qS1(abs.(MM)+abs.(NN)))")

end





# examples from Dopico et al. SIMAX, 43:1213-1237, 2022.    
n = 50; k = -16; 
for k in (-16 : 5 : 15)

T = rand(n,n);
T[1, 2 : n] = 10. ^(k)*T[1, 2 : n]
T[4:n, 3] = 10. ^(k)*T[4:n, 3]
D = sort(round.(Int,1. ./ rand(n)))
A = T*Diagonal(D); E = T;
ev = sort(eigvals(A,E),by=real)
# println("norm(evs-ev) = $(norm(ev - D))")
# compute the chordal distance between exact and computed eigenvalues
corig = norm(abs.(ev-D)./sqrt.(1. .+ ev .^2)./sqrt.(1. .+ D .^2))
isfinite(corig) || (corig = Inf)
# println("k = $k corig = $corig")

M = abs.(A)+abs.(E); M[2,:] .= 0; M[4,:] .= 0; M[:,4] .= 0;
MM = copy(M);
dleft, dright = rcsumsbal!(MM)
@test dleft*M*dright ≈ MM
@test qS1(MM) < qS1(M)


qsorig = qS1(abs.(A)+abs.(E))
AA = copy(A); EE = copy(E); 
@time D1, D2 = pbalance!(AA,EE)
@test AA == D1*A*D2 && EE == D1*E*D2 
qsfin = qS1(abs.(AA)+abs.(EE))
@test qsfin < qsorig 
#println("qsorig/qsfin = $(qsorig/qsfin)")
@test 2. .^round.(Int,log2.(D1.diag)) == D1.diag && 2. .^round.(Int,log2.(D2.diag)) == D2.diag
evs = sort(eigvals(AA,EE),by=real)
@test evs ≈ D
# compute the chordal distance between exact and computed eigenvalues
cofin = norm(abs.(evs-D)./sqrt.(1. .+ evs .^2)./sqrt.(1. .+ D .^2))
# println("k = $k pow2 = true cofin = $cofin")
@test ev ≈ D ? cofin/10 < corig :  cofin < corig 


AA = copy(A); EE = copy(E); 
@time D1, D2 = pbalance!(AA,EE; pow2 = false)
@test AA ≈ D1*A*D2 && EE ≈ D1*E*D2 
qsfin = max(qS1(AA),qS1(EE))
@test qsfin < qsorig 
#println("qsorig/qsfin = $(qsorig/qsfin)")
evs = sort(eigvals(AA,EE),by=real)
@test evs ≈ D
cofin = norm(abs.(evs-D)./sqrt.(1. .+ evs .^2)./sqrt.(1. .+ D .^2))
# println("k = $k pow2 = false cofin = $cofin")
end

#  Example 1, Ward 1981, pp.148f 

A = [-2.0e+1 -1.0e+4 -2.0e+0 -1.0e+6 -1.0e+1 -2.0e+5
6.0e-3 4.0e+0 6.0e-4 2.0e+2 3.0e-3 3.0e+1
-2.0e-1 -3.0e+2 -4.0e-2 -1.0e+4 0.0e+0 3.0e+3
6.0e-5 4.0e-2 9.0e-6 9.0e+0 3.0e-5 5.0e-1
6.0e-2 5.0e+1 8.0e-3 -4.0e+3 8.0e-2 0.0e+0
0.0e+0 1.0e+3 7.0e-1 -2.0e+5 1.3e+1 -6.0e+4 ];

E = [-2.0e+1 -1.0e+4 2.0e+0 -2.0e+6 1.0e+1 -1.0e+5
5.0e-3 3.0e+0 -2.0e-4 4.0e+2 -1.0e-3 3.0e+1
0.0e+0 -1.0e+2 -8.0e-2 2.0e+4 -4.0e-1 0.0e+0
5.0e-5 3.0e-2 2.0e-6 4.0e+0 2.0e-5 1.0e-1
4.0e-2 3.0e+1 -1.0e-3 3.0e+3 -1.0e-2 6.0e+2
-1.0e+0 0.0e+0 4.0e-1 -1.0e+5 4.0e+0 2.0e+4 ]; 

D = collect(1.:1.:6.)  # exact eigenvalues

qsorig = qS1(abs.(A)+abs.(E))
AA = copy(A); EE = copy(E); 
@time D1, D2 = pbalance!(AA,EE; pow2 = false)
@test AA ≈ D1*A*D2 && EE ≈ D1*E*D2 
qsfin = qS1(abs.(AA)+abs.(EE))
ev = sort(eigvals(A,E),by=real)
corig = norm(abs.(ev-D)./sqrt.(1. .+ ev .^2)./sqrt.(1. .+ D .^2))
# println("corig = $corig")
evs = sort(eigvals(AA,EE),by=real)
cofin = norm(abs.(evs-D)./sqrt.(1. .+ evs .^2)./sqrt.(1. .+ D .^2))
# println("cofin = $cofin")
@test 100000*cofin < corig && 100000*qsfin < qsorig 


# Example 3 (graded matrix), see Ward, 1981, pp.148f

A = [    1.0000e+00   1.0000e+01            0            0            0            0            0
1.0000e+01   1.0000e+02   1.0000e+03            0            0            0            0
         0   1.0000e+03   1.0000e+04   1.0000e+05            0            0            0
         0            0   1.0000e+05   1.0000e+06   1.0000e+07            0            0
         0            0            0   1.0000e+07   1.0000e+08   1.0000e+09            0
         0            0            0            0   1.0000e+09   1.0000e+10   1.0000e+11
         0            0            0            0            0   1.0000e+11   1.0000e+12];
E = 1. *Matrix(I(7)); 
qsorig = qS1(abs.(A)+abs.(E))
AA = copy(A); EE = copy(E); 
@time D1, D2 = pbalance!(AA,EE; pow2 = false)
@test AA ≈ D1*A*D2 && EE ≈ D1*E*D2 
qsfin = qS1(abs.(AA)+abs.(EE))
@test 100000*qsfin < qsorig 

As = sparse(A); Es = sparse(E)
qsorig = qS1(abs.(As)+abs.(Es))
AA = copy(As); EE = copy(Es); 
@time D1, D2 = pbalance!(AA,EE; pow2 = false)
@test AA ≈ D1*As*D2 && EE ≈ D1*Es*D2 
qsfin = qS1(abs.(AA)+abs.(EE))
@test 100000*qsfin < qsorig 


end


println("klf_rlsplit")      
@testset "klf_rlsplit" begin

fast = true
for fast in (false,true)

M2 = zeros(0,0); N2 = zeros(0,0);
M = copy(M2); N = copy(N2); 

@time M1, N1, Q, Z, ν, μ, n, m, p  = klf_rlsplit(M, N, fast = fast)
@test norm(Q'*M2*Z-M1) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N1) < sqrt(eps(1.)) &&
      ν == [] && μ == [] && n == 0 && m == 0  && p == 0

M2 = zeros(3,0); N2 = zeros(3,0);
M = copy(M2); N = copy(N2); 

@time M1, N1, Q, Z, ν, μ, n, m, p  = klf_rlsplit(M, N, fast = fast, finite_infinite = false)
@test norm(Q'*M2*Z-M1) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N1) < sqrt(eps(1.)) &&
      ν == [] && μ == [] && n == 0 && m == 0  && p == 3

M2 = zeros(3,0); N2 = zeros(3,0);
M = copy(M2); N = copy(N2); 

@time M1, N1, Q, Z, ν, μ, n, m, p  = klf_rlsplit(M, N, fast = fast, finite_infinite = true)
@test norm(Q'*M2*Z-M1) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N1) < sqrt(eps(1.)) &&
      ν == [3] && μ == [0] && n == 0 && m == 0  && p == 0


M2 = zeros(0,3); N2 = zeros(0,3);
M = copy(M2); N = copy(N2); 

@time M1, N1, Q, Z, ν, μ, n, m, p  = klf_rlsplit(M, N, fast = fast, finite_infinite = false)
@test norm(Q'*M2*Z-M1) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N1) < sqrt(eps(1.)) &&
      ν == [0] && μ == [3] && n == 0 && m == 0  && p == 0

M2 = zeros(0,3); N2 = zeros(0,3);
M = copy(M2); N = copy(N2); 

@time M1, N1, Q, Z, ν, μ, n, m, p  = klf_rlsplit(M, N, fast = fast, finite_infinite = true)
@test norm(Q'*M2*Z-M1) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N1) < sqrt(eps(1.)) &&
      ν == [] && μ == [] && n == 0 && m == 3  && p == 0

M2 = ones(1,1); N2 = zeros(1,1); 
M = copy(M2); N = copy(N2); 

@time M1, N1, Q, Z, ν, μ, n, m, p  = klf_rlsplit(M, N, fast = fast, finite_infinite = false)
@test norm(Q'*M2*Z-M1) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N1) < sqrt(eps(1.)) &&
      ν == [1] && μ == [1] && n == 0 && m == 0  && p == 0

M2 = ones(1,1); N2 = zeros(1,1); 
M = copy(M2); N = copy(N2); 

@time M1, N1, Q, Z, ν, μ, n, m, p  = klf_rlsplit(M, N, fast = fast, finite_infinite = true)
@test norm(Q'*M2*Z-M1) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N1) < sqrt(eps(1.)) &&
      ν == [1] && μ == [1] && n == 0 && m == 0  && p == 0


M2 = zeros(1,1); N2 = ones(1,1); 
M = copy(M2); N = copy(N2); 

@time M1, N1, Q, Z, ν, μ, n, m, p  = klf_rlsplit(M, N, fast = fast, finite_infinite = false)
@test norm(Q'*M2*Z-M1) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N1) < sqrt(eps(1.)) &&
      ν == [] && μ == [] && n == 1 && m == 0  && p == 0

M2 = zeros(1,1); N2 = ones(1,1); 
M = copy(M2); N = copy(N2); 

@time M1, N1, Q, Z, ν, μ, n, m, p  = klf_rlsplit(M, N, fast = fast, finite_infinite = true)
@test norm(Q'*M2*Z-M1) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N1) < sqrt(eps(1.)) &&
      ν == [] && μ == [] && n == 1 && m == 0  && p == 0

# Test Suite 1 (Kagstrom)
M2 = [  22  34  31   31  17
        45  45  42   19  29
        39  47  49   26  34
        27  31  26   21  15
        38  44  44   24  30 ];
    
N2 = [   13  26  25  17  24 
        31  46  40  26  37 
        26  40  19  25  25 
        16  25  27  14  23 
        24  35  18  21  22  ];

M = copy(M2); N = copy(N2); 

@time M1, N1, Q, Z, ν, μ, n, m, p  = klf_rlsplit(M, N, fast = fast, finite_infinite = false)
@test norm(Q'*M2*Z-M1) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N1) < sqrt(eps(1.)) &&
      ν == [1] && μ == [2] && n == 3 && m == 0  && p == 1

M = copy(M2); N = copy(N2); 

@time M1, N1, Q, Z, ν, μ, n, m, p  = klf_rlsplit(M, N, fast = fast, finite_infinite = true)
@test norm(Q'*M2*Z-M1) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N1) < sqrt(eps(1.)) &&
      ν == [2] && μ == [1] && n == 3 && m == 1  && p == 0

# Test data for MB04UD
M2 = [
   2.0  0.0  2.0 -2.0
   0.0 -2.0  0.0  2.0
   2.0  0.0 -2.0  0.0
   2.0 -2.0  0.0  2.0];
N2 = [
   1.0  0.0  1.0 -1.0
   0.0 -1.0  0.0  1.0
   1.0  0.0 -1.0  0.0
   1.0 -1.0  0.0  1.0];

M = copy(M2); N = copy(N2); 

@time M1, N1, Q, Z, ν, μ, n, m, p  = klf_rlsplit(M, N, fast = fast, finite_infinite = false)
@test norm(Q'*M2*Z-M1) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N1) < sqrt(eps(1.)) &&
      ν == [] && μ == [] && n == 4 && m == 0  && p == 0

M = copy(M2); N = copy(N2); 

@time M1, N1, Q, Z, ν, μ, n, m, p  = klf_rlsplit(M, N, fast = fast, finite_infinite = true)
@test norm(Q'*M2*Z-M1) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N1) < sqrt(eps(1.)) &&
      ν == [] && μ == [] && n == 4 && m == 0  && p == 0

# Test Suite 2 (Example 2.2.1, Beelen)
M = zeros(14,16); 
i1 = [1 2 3 6 7 8 9 10 11 12 13 13 14]; 
j1 = [4 6 7 8 9 10 11 12 13 14 15 16 16];
aval = [1 1 1 1 1 1 1 1 1 2 3 1 3];
for k = 1:length(i1)
    M[i1[k],j1[k]] = aval[k]
end
N = zeros(14,16); 
i1 = [1 2 3 5 6 7 10 12 13 14]; j1 = [3 5 6 8 9 10 13 14 15 16];
for k = 1:length(i1)
    N[i1[k],j1[k]] = 1
end
 
Q = qr(rand(14,14)).Q; Z = qr(rand(16,16)).Q;
M2 = Q*M*Z; N2 = Q*N*Z;

M = copy(M2); N = copy(N2); 

@time M1, N1, Q, Z, ν, μ, n, m, p  = klf_rlsplit(M, N, atol1 = 1.e-7, atol2 = 1.e-7, fast = fast, finite_infinite = false)
@test norm(Q'*M2*Z-M1) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N1) < sqrt(eps(1.)) &&
      ν == [4, 2, 0 ] && μ == [6, 3, 1]  && n == 6 && m == 0  && p == 2

M = copy(M2); N = copy(N2); 

@time M1, N1, Q, Z, ν, μ, n, m, p  = klf_rlsplit(M, N, fast = fast, finite_infinite = true, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q'*M2*Z-M1) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N1) < sqrt(eps(1.)) &&
      ν == [1, 1, 2, 4] && μ == [0, 1, 2, 3] && n == 6 && m == 4  && p == 0

Ty = Float64      
for Ty in (Float64, Complex{Float64}, BigFloat, Complex{BigFloat})

abstol = sqrt(eps(one(real(Ty))))

mr = 2; nr = mr+1; ni = 4; nf = 10; nl = 4; ml = nl+1; 
mM = mr+ni+nf+ml;
nM = nr+ni+nf+nl;
M2 = [rand(Ty,mr,nM); 
     zeros(Ty,ni,nr) Matrix{Ty}(I,ni,ni) rand(Ty,ni,nf+nl); 
     zeros(Ty,nf,nr+ni) rand(Ty,nf,nf+nl);
     zeros(Ty,ml,nr+ni+nf) rand(Ty,ml,nl)]
N2 = [rand(Ty,mr,nM); 
     zeros(Ty,ni,nr) diagm(1 => ones(Ty,ni-1)) rand(Ty,ni,nf+nl); 
     zeros(Ty,nf,nr+ni) rand(Ty,nf,nf+nl);
     zeros(Ty,ml,nr+ni+nf) rand(Ty,ml,nl)]
Q = qr(rand(Ty,mM,mM)).Q;
Z = qr(rand(Ty,nM,nM)).Q; 
M2 = Q*M2*Z;
N2 = Q*N2*Z;


M = copy(M2); N = copy(N2);
atol1 = 1.e-7; atol2 = 1.e-7; rtol = 1.e-7;
@time M1, N1, Q1, Z1, νr, μr, νi, nfe, νl, μl = klf(M, N, finite_infinite=true,fast = true, atol1 = 0*atol1, atol2 = 0*atol2, rtol = rtol)
@test norm(Q1'*M2*Z1-M1) < atol1 &&
      norm(Q1'*N2*Z1-N1) < atol2 &&
      νr == [ones(Int,mr); [0]] && μr == ones(Int,nr) && 
      νi == ones(Int,ni) && νl == ones(Int,ml) && μl == [[0]; ones(Int,nl)] && nfe == nf

M = copy(M2); N = copy(N2); 

@time M1, N1, Q, Z, ν, μ, n, m, p  = klf_rlsplit(M, N, atol1 = 1.e-7, atol2 = 1.e-7, fast = fast, finite_infinite = false)
@test norm(Q'*M2*Z-M1) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N1) < sqrt(eps(1.)) &&
      ν == [2, 2, 1, 1] && μ == [2, 2, 2, 1]  && n == 14 && m == 0  && p == 1

M = copy(M2); N = copy(N2); 

@time M1, N1, Q, Z, ν, μ, n, m, p  = klf_rlsplit(M, N, atol1 = 1.e-7, atol2 = 1.e-7, fast = fast, finite_infinite = true)
@test norm(Q'*M2*Z-M1) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N1) < sqrt(eps(1.)) &&
      ν == [1, 2, 2, 2, 2] && μ == [0, 2, 2, 2, 2]  && n == 12 && m == 1  && p == 0
end

end
end

@testset "klf_left and klf_right" begin
println("klf_left and klf_right")      

fast = true
for fast in (false,true)

M2 = zeros(0,0); N2 = zeros(0,0);
M = copy(M2); N = copy(N2); 

@time M1, N1, Q, Z, ν, μ, nf, νl, μl = klf_left(M, N, fast = fast)
@test norm(Q'*M2*Z-M1) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N1) < sqrt(eps(1.)) &&
      ν == [] && μ == [] && νl == [] && μl == []  && nf == 0


M2 = zeros(0,0); N2 = zeros(0,0);
M = copy(M2); N = copy(N2);

@time M1, N1, Q, Z, νr, μr, nf, ν, μ  = klf_right(M, N, fast = fast)
@test norm(Q'*M2*Z-M) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N) < sqrt(eps(1.)) &&
      νr == [] && μr == [] && ν == [] && μ == []  && nf == 0


M2 = zeros(3,0); N2 = zeros(3,0);
M = copy(M2); N = copy(N2);

@time M1, N1, Q, Z, ν, μ, nf, νl, μl = klf_left(M, N, fast = fast)
@test norm(Q'*M2*Z-M1) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N1) < sqrt(eps(1.)) &&
      ν == [] && μ == [] && νl == [3] && μl == [0]  && nf == 0   

M2 = zeros(3,0); N2 = zeros(3,0);
M = copy(M2); N = copy(N2);

@time M1, N1, Q, Z, νr, μr, nf, ν, μ  = klf_right(M, N, fast = fast)
@test norm(Q'*M2*Z-M1) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N1) < sqrt(eps(1.)) &&
      νr == [] && μr == [] && ν == [3] && μ == [0] && nf == 0


M2 = zeros(0,3); N2 = zeros(0,3);
M = copy(M2); N = copy(N2);

@time M1, N1, Q, Z, ν, μ, nf, νl, μl = klf_left(M, N, fast = fast)
@test norm(Q'*M2*Z-M1) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N1) < sqrt(eps(1.)) &&
      ν == [0] && μ == [3] && νl == [] && μl == [] && nf == 0

M2 = zeros(0,3); N2 = zeros(0,3);
M = copy(M2); N = copy(N2);
@time M1, N1, Q, Z, νr, μr, nf, ν, μ  = klf_right(M, N, fast = fast)
@test norm(Q'*M2*Z-M1) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N1) < sqrt(eps(1.)) &&
      νr == [0] && μr == [3] && ν == [] && μ == [] && nf == 0

N2 = zeros(2,1); M2 = [zeros(1,1); ones(1,1)];  
M = copy(M2); N = copy(N2);
@time M1, N1, Q, Z, ν, μ, nf, νl, μl = klf_left(M, N, fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q'*M2*Z-M1) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N1) < sqrt(eps(1.)) &&
      ν == [1] && μ == [1] && νl == [1] && μl == [0] && nf == 0    

N2 = zeros(2,1); M2 = [zeros(1,1); ones(1,1)];  
M = copy(M2); N = copy(N2);
@time M1, N1, Q, Z, νr, μr, nf, ν, μ  = klf_right(M, N, fast = fast)
@test norm(Q'*M2*Z-M1) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N1) < sqrt(eps(1.)) &&
      νr == [] && μr == [] && ν == [2] && μ == [1] && nf == 0

M2 = zeros(1,1); N2 = zeros(1,1); 
M = copy(M2); N = copy(N2);
@time M1, N1, Q, Z, ν, μ, nf, νl, μl = klf_left(M, N, fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q'*M2*Z-M1) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N1) < sqrt(eps(1.)) &&
      ν == [0] && μ == [1]  && νl == [1] && μl == [0] && nf == 0   

M2 = zeros(1,1); N2 = zeros(1,1); 
M = copy(M2); N = copy(N2);
@time M1, N1, Q, Z, νr, μr, nf, ν, μ  = klf_right(M, N, fast = fast)
@test norm(Q'*M2*Z-M1) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N1) < sqrt(eps(1.)) &&
      νr == [0] && μr == [1] && ν == [1] && μ == [0] && nf == 0 
      
# Test Suite 1 (Kagstrom)
M2 = [  22  34  31   31  17
        45  45  42   19  29
        39  47  49   26  34
        27  31  26   21  15
        38  44  44   24  30 ];
    
N2 = [   13  26  25  17  24 
        31  46  40  26  37 
        26  40  19  25  25 
        16  25  27  14  23 
        24  35  18  21  22  ];

M = copy(M2); N = copy(N2); 
@time M1, N1, Q1, Z1, ν, μ, nf, νl, μl = klf_left(M, N, fast = fast)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      ν == [1] && μ == [2] && νl == [1] && μl == [0] && nf == 3   

M = copy(M2); N = copy(N2); 
@time M1, N1, Q1, Z1, νr, μr, nf, ν, μ = klf_right(M, N, fast = fast)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [0] && μr == [1] && ν == [2] && μ == [1] && nf == 3 

# Test data for MB04UD
M2 = [
   2.0  0.0  2.0 -2.0
   0.0 -2.0  0.0  2.0
   2.0  0.0 -2.0  0.0
   2.0 -2.0  0.0  2.0];
N2 = [
   1.0  0.0  1.0 -1.0
   0.0 -1.0  0.0  1.0
   1.0  0.0 -1.0  0.0
   1.0 -1.0  0.0  1.0];


M = copy(M2); N = copy(N2); 
@time M1, N1, Q1, Z1, ν, μ, nf, νl, μl = klf_left(M, N, fast = fast)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      ν == [ ] && μ == [ ]  && νl == [] && μl == [] && nf == 4  

M = copy(M2); N = copy(N2); 
@time M1, N1, Q1, Z1, νr, μr, nf, ν, μ  = klf_right(M, N, fast = fast)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      ν == [ ] && μ == [ ]  && νl == [] && μl == [] && nf == 4  
      
# Test Suite 2 (Example 2.2.1, Beelen)
M = zeros(14,16); 
i1 = [1 2 3 6 7 8 9 10 11 12 13 13 14]; 
j1 = [4 6 7 8 9 10 11 12 13 14 15 16 16];
aval = [1 1 1 1 1 1 1 1 1 2 3 1 3];
for k = 1:length(i1)
    M[i1[k],j1[k]] = aval[k]
end
N = zeros(14,16); 
i1 = [1 2 3 5 6 7 10 12 13 14]; j1 = [3 5 6 8 9 10 13 14 15 16];
for k = 1:length(i1)
    N[i1[k],j1[k]] = 1
end
 
Q = qr(rand(14,14)).Q; Z = qr(rand(16,16)).Q;
M2 = Q*M*Z; N2 = Q*N*Z;


M = copy(M2); N = copy(N2); 
@time M1, N1, Q1, Z1, ν, μ, nf, νl, μl = klf_left(M, N, fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      ν == [4, 2, 0 ] && μ == [6, 3, 1] && νl == [1, 1, 1, 2] && μl == [0, 1, 1, 1] && nf == 3

M = copy(M2); N = copy(N2); 
@time M1, N1, Q1, Z1, ν, μ, nf, νl, μl = klf_left(M, N, fast = fast, ut = true, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      ν == [4, 2, 0 ] && μ == [6, 3, 1] && νl == [1, 1, 1, 2] && μl == [0, 1, 1, 1] && nf == 3

M = copy(M2); N = copy(N2); 
@time M1, N1, Q1, Z1, n, m, νi, νl, μl = klf_leftinf(M, N, fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      n == 6 && m == 4 && νl == [1, 1, 1, 2] && μl == [0, 1, 1, 1] && νi == [2, 1]

M = copy(M2); N = copy(N2); 
@time M1, N1, Q1, Z1, n, m, νi, νl, μl = klf_leftinf(M, N, fast = fast, ut = true, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      n == 6 && m == 4 && νl == [1, 1, 1, 2] && μl == [0, 1, 1, 1] && νi == [2, 1]

M = copy(M2); N = copy(N2); 
@time M1, N1, Q1, Z1, νr, μr, nf, ν, μ  = klf_right(M, N, fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [2, 1, 0] && μr == [4, 2, 1]  && ν == [1, 1, 2, 4] && μ == [0, 1, 2, 3] && nf == 3

M = copy(M2); N = copy(N2); 
@time M1, N1, Q1, Z1, νr, μr, nf, ν, μ  = klf_right(M, N, fast = fast, ut = true, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [2, 1, 0] && μr == [4, 2, 1]  && ν == [1, 1, 2, 4] && μ == [0, 1, 2, 3] && nf == 3

M = copy(M2); N = copy(N2); 
@time M1, N1, Q1, Z1, νr, μr, νi, n, p  = klf_rightinf(M, N, fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [2, 1, 0] && μr == [4, 2, 1]  && νi == [1, 2] && n == 6 && p == 2

M = copy(M2); N = copy(N2); 
@time M1, N1, Q1, Z1, νr, μr, νi, n, p  = klf_rightinf(M, N, fast = fast, ut = true, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [2, 1, 0] && μr == [4, 2, 1]  && νi == [1, 2] && n == 6 && p == 2
      

M = copy(M2'); N = copy(N2'); 
@time M1, N1, Q1, Z1, ν, μ, nf, νl, μl = klf_left(M, N, fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2'*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2'*Z1-N1) < sqrt(eps(1.)) &&
      ν == [3, 2, 1, 0] && μ == [4, 2, 1, 1] && νl == [1, 2, 4] && μl == [0, 1, 2] && nf == 3


M = copy(M2'); N = copy(N2'); 
@time M1, N1, Q1, Z1, νr, μr, nf, ν, μ  = klf_right(M, N, fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2'*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2'*Z1-N1) < sqrt(eps(1.)) &&
      νr == [1, 1, 1, 0] && μr == [2, 1, 1, 1] && ν == [1, 3, 6] && μ == [0, 2, 4] && nf == 3

      
# intermediary computation
M2 = [
    1.0  -1.14463e-18  -1.47479e-15  4.87132e-15  6.96416e-31  -2.32652e-31  -1.50216e-16   1.69785e-16   1.42103e-16   7.77705e-17
    0.0   1.0           0.0          0.0          0.0           0.0          -2.34988e-16  -3.55922e-16  -6.84622e-17  -6.11787e-17
    0.0   0.0           1.0          0.0          0.0           0.0           4.90878e-16   3.80428e-16   3.62292e-16   3.74223e-16
    0.0   0.0           0.0          1.0          0.0           0.0           8.12392e-16   5.94559e-16   1.17495e-16   6.02872e-16
    0.0   0.0           0.0          0.0          0.0           0.0           1.0           5.10024e-16  -2.95823e-31  -2.69854e-16
    0.0   0.0           0.0          0.0          0.0           0.0           0.0           1.0           0.0           5.41115e-16 
]
N2 = [
    0.0  0.0  0.0  0.0  0.0  0.0  -1.29895e-15   4.68167e-15   1.51642e-15  -2.88014e-17
    0.0  0.0  0.0  0.0  0.0  0.0   0.689011     -0.0434045     0.72345      -3.72429e-16
    0.0  0.0  0.0  0.0  0.0  0.0   0.723189      0.10666      -0.682364     -7.51122e-18
    0.0  0.0  0.0  0.0  0.0  0.0  -0.0475455     0.993348      0.10488       1.43459e-17
    0.0  0.0  0.0  0.0  0.0  0.0   0.0           0.0           0.0          -5.10024e-16
    0.0  0.0  0.0  0.0  0.0  0.0   0.0           0.0           0.0          -1.0   
]

M = copy(M2); N = copy(N2); 
@time M1, N1, Q1, Z1, νr, μr, nf, νl, μl  = klf_right(M, N, fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [2, 1, 0] && μr == [4, 2, 1] &&  νl == [1, 2] && μl == [1, 2] && nf == 0

# intermediary result
M2 = [
1.0  -1.51959e-15   4.9582e-15   -7.05588e-17   6.96416e-31  -2.32652e-31   2.38551e-16  -2.02364e-16   5.79165e-17   7.77705e-17
0.0   1.0           2.77556e-17  -1.73472e-18  -2.76763e-47  -1.35446e-47  -1.43825e-16  -3.50328e-16  -5.85653e-16   3.74223e-16
0.0   0.0          -1.0           2.77556e-17   4.26419e-48   9.29779e-48   5.70351e-16   5.60542e-16   6.19106e-16  -6.02872e-16
0.0   0.0           0.0          -1.0           7.98872e-48   1.74576e-47  -1.64948e-16  -3.84582e-16  -1.94459e-16   6.11787e-17
0.0   0.0           0.0           0.0           0.0           0.0           0.731013      0.0979        0.675304      2.52908e-16
0.0   0.0           0.0           0.0           0.0           0.0           0.0          -0.989654      0.143472      5.77433e-16
]
N2 = [
    0.0  0.0  0.0  0.0  0.0  0.0   1.40857e-15  -4.84718e-15  -4.04657e-17  -4.16334e-17
    0.0  0.0  0.0  0.0  0.0  0.0  -1.0           2.21239e-7    5.83691e-7    2.19061e-17
    0.0  0.0  0.0  0.0  0.0  0.0   0.0           1.0           2.90233e-7   -1.24119e-17
    0.0  0.0  0.0  0.0  0.0  0.0   0.0           0.0           1.0           3.44402e-16
    0.0  0.0  0.0  0.0  0.0  0.0   0.0           0.0           0.0           0.145907
    0.0  0.0  0.0  0.0  0.0  0.0   0.0           0.0           0.0          -0.989298
]   
M = copy(M2); N = copy(N2); 
@time M1, N1, Q1, Z1, νr, μr, nf, νl, μl  = klf_right(M, N, fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [2, 1, 0] && μr == [4, 2, 1] &&  νl == [1, 2] && μl == [1, 2] && nf == 0 

# Example  Karathanasi and Karampetakis IMA J. Math. Contr Inform. (2019)
N2 = [
1. 1 0 0 1 0 -1 0
1 1 1 1 0 0 0 0
-1 1 0 0 0 0 0 0
1 0 0 0 0 -1 0 0
0 -2 0 0 0 1 1 0
1 0 0 0 0 -1 0 0
]
M2 = [
2. 1 0 0 0 -2 0 0
0 2 1 0 0 0 -2 0
-2 1 0 0 0 1 -1 1
3 -1 1 1 0 -2 2 0
0 -2 0 0 0 1 2 -1
2 -1 0 0 0 -2 1 0
]


M = copy(M2); N = copy(N2); 
@time M1, N1, Q1, Z1, ν, μ, nf, νl, μl = klf_left(M, N, fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      ν == [3, 1 ] && μ == [3, 3] && νl == [] && μl == [] && nf == 2


M = copy(M2); N = copy(N2); 
@time M1, N1, Q1, Z1, νr, μr, nf, ν, μ  = klf_right(M, N, fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [2, 0 ] && μr == [2, 2] && ν == [1, 1] && μ == [1, 1] && nf == 2

for Ty in (Float64, Complex{Float64}, BigFloat, Complex{BigFloat})

abstol = sqrt(eps(one(real(Ty))))

# generic cases 
m = 1; n = 2;
M2 = rand(Ty,m,n)
N2 = rand(Ty,m,n)
M = copy(M2); N = copy(N2);
@time M1, N1, Q, Z, ν, μ, nf, νl, μl = klf_left(M, N, fast = fast)
@test norm(Q'*M2*Z-M1) < abstol &&
      norm(Q'*N2*Z-N1) < abstol &&
      ν == [1, 0 ] && μ == [1, 1] && νl == [] && μl == [] && nf == 0

M = copy(M2); N = copy(N2);
@time M1, N1, Q, Z, νr, μr, nf, ν, μ = klf_right(M, N, fast = fast)
@test norm(Q'*M2*Z-M1) < abstol &&
      norm(Q'*N2*Z-N1) < abstol &&
      νr == [1, 0 ] && μr == [1, 1] && ν == [] && μ == [] && nf == 0

m = 3; n = 5;
M2 = rand(Ty,m,n)
N2 = rand(Ty,m,n)
M = copy(M2); N = copy(N2);
@time M1, N1, Q, Z, ν, μ, nf, νl, μl = klf_left(M, N, fast = fast)
@test norm(Q'*M2*Z-M1) < abstol &&
      norm(Q'*N2*Z-N1) < abstol &&
      ν == [2, 1, 0] && μ == [2, 2, 1] && νl == [] && μl == [] && nf == 0

m = 3; n = 5;
M2 = rand(Ty,m,n)
N2 = rand(Ty,m,n)
M = copy(M2); N = copy(N2);
@time M1, N1, Q, Z, νr, μr, nf, ν, μ = klf_right(M, N, fast = fast)
@test norm(Q'*M2*Z-M1) < abstol &&
      norm(Q'*N2*Z-N1) < abstol &&
      νr == [2, 1, 0] && μr == [2, 2, 1] && ν == [] && μ == [] && nf == 0

      
m = 5; n = 3; 
M2 = rand(Ty,m,n)
N2 = rand(Ty,m,n)
M = copy(M2); N = copy(N2);
@time M1, N1, Q, Z, ν, μ, nf, νl, μl = klf_left(M, N, fast = fast);
@test norm(Q'*M2*Z-M1) < abstol &&
      norm(Q'*N2*Z-N1) < abstol &&
      ν == [] && μ == [] && νl == [1, 2, 2] && μl == [0, 1, 2] && nf == 0

m = 5; n = 3; 
M2 = rand(Ty,m,n)
N2 = rand(Ty,m,n)
M = copy(M2); N = copy(N2);
@time M1, N1, Q, Z, νr, μr, nf, ν, μ = klf_right(M, N, fast = fast);
@test norm(Q'*M2*Z-M1) < abstol &&
      norm(Q'*N2*Z-N1) < abstol &&
      νr == [] && μr == [] && ν == [1, 2, 2] && μ == [0, 1, 2] && nf == 0

m = 3; n = 5; r = 2;
M2 = rand(Ty,m,n)
N2 = rand(Ty,m,m)*[rand(Ty,r,r) zeros(Ty,r,n-r); zeros(Ty,m-r,n)]*rand(Ty,n,n)
M = copy(M2); N = copy(N2);
@time M1, N1, Q, Z, ν, μ, nf, νl, μl = klf_left(M, N; fast = fast);
@test norm(Q'*M2*Z-M1) < abstol &&
      norm(Q'*N2*Z-N1) < abstol &&
      ν == [3, 0] && μ == [3, 2] && νl == [] && μl == [] && nf == 0

m = 3; n = 5; r = 2;
M2 = rand(Ty,m,n)
N2 = rand(Ty,m,m)*[rand(Ty,r,r) zeros(Ty,r,n-r); zeros(Ty,m-r,n)]*rand(Ty,n,n)
M = copy(M2); N = copy(N2);
@time M1, N1, Q, Z, νr, μr, nf, ν, μ = klf_right(M, N; fast = fast)
@test norm(Q'*M2*Z-M1) < abstol &&
      norm(Q'*N2*Z-N1) < abstol &&
      νr == [2, 0] && μr == [2, 2] && ν == [1] && μ == [1] && nf == 0

m = 3; n = 5; r = 2;
M2 = rand(Ty,m,n)
N2 = rand(Ty,m,m)*[rand(Ty,r,r) zeros(Ty,r,n-r); zeros(Ty,m-r,n)]*rand(Ty,n,n)
M = copy(M2); N = copy(N2);
@time M1, N1, Q, Z, νr, μr, nf, ν, μ = klf_right(M, N; fast = fast)
@test norm(Q'*M2*Z-M1) < abstol &&
      norm(Q'*N2*Z-N1) < abstol &&
      νr == [2, 0] && μr == [2, 2] && ν == [1] && μ == [1] && nf == 0

m = 3; n = 5; r = 2;
M2 = rand(Ty,m,n)
N2 = rand(Ty,m,m)*[rand(Ty,r,r) zeros(Ty,r,n-r); zeros(Ty,m-r,n)]*rand(Ty,n,n)
M = copy(M2); N = copy(N2);
@time M1, N1, Q, Z, ν, μ, nf, νl, μl = klf_left(M, N; fast = fast)
@test norm(Q'*M2*Z-M1) < abstol &&
      norm(Q'*N2*Z-N1) < abstol &&
      ν == [3, 0] && μ == [3, 2] && νl == [] && μl == [] && nf == 0


M2 = rand(Ty,7,7)
N2 = [zeros(Ty,4,3) rand(Ty,4,4); zeros(Ty,3,7)]
M = copy(M2); N = copy(N2);
@time M1, N1, Q, Z, ν, μ, nf, νl, μl = klf_left(M, N, fast = fast);
@test norm(Q'*M2*Z-M1) < abstol &&
      norm(Q'*N2*Z-N1) < abstol &&
      ν == [3] && μ == [3] && νl == [] && μl == [] && nf == 4


M2 = rand(Ty,7,7)
N2 = [zeros(Ty,4,3) rand(Ty,4,4); zeros(Ty,3,7)]
M = copy(M2); N = copy(N2);
@time M1, N1, Q, Z, νr, μr, nf, ν, μ = klf_right(M, N, fast = fast);
@test norm(Q'*M2*Z-M1) < abstol &&
      norm(Q'*N2*Z-N1) < abstol &&
      νr == [] && μr == [] && ν == [3] && μ == [3] && nf == 4

end
end

@testset "klf" begin
println("klf")
fast = true
for fast in (false,true)


M2 = zeros(0,0); N2 = zeros(0,0);
M = copy(M2); N = copy(N2);

@time M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl  = klf(M, N, fast = fast)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [] && μr == [] && νi == [] && νl == [] && μl == []  && nf == 0      

M = copy(M2); N = copy(N2);

@time M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl  = klf(M, N, fast = fast, finite_infinite = true)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [] && μr == [] && νi == [] && νl == [] && μl == []  && nf == 0      

M2 = zeros(3,0); N2 = zeros(3,0);
M = copy(M2); N = copy(N2);

@time M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl  = klf(M, N, fast = fast)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [] && μr == [] && νi == [] && νl == [3] && μl == [0]  && nf == 0           

M2 = zeros(0,3); N2 = zeros(0,3); 
M = copy(M2); N = copy(N2);

@time M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl  = klf(M, N, fast = fast)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [0] && μr == [3] && νi == [] && νl == [] && μl == []  && nf == 0           

M2 = ones(1,1); N2 = zeros(1,1); 
M = copy(M2); N = copy(N2);

@time M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl  = klf(M, N, fast = fast)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [] && μr == [] && νi == [1] && νl == [] && μl == []  && nf == 0           


M2 = zeros(1,1); N2 = ones(1,1);  
M = copy(M2); N = copy(N2);

@time M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl  = klf(M, N, fast = fast)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [] && μr == [] && νi == [] && νl == [] && μl == []  && nf == 1


N2 = zeros(2,1); M2 = [zeros(1,1); ones(1,1)];   
M = copy(M2); N = copy(N2);

@time M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl  = klf(M, N, fast = fast)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [] && μr == [] && νi == [1] && νl == [1] && μl == [0]  && nf == 0

M2 = zeros(1,1); N2 = zeros(1,1);   
M = copy(M2); N = copy(N2);

@time M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl  = klf(M, N, fast = fast)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [0] && μr == [1] && νi == [] && νl == [1] && μl == [0]  && nf == 0


# Test Suite 1 (Kagstrom)
M2 = [  22  34  31   31  17
        45  45  42   19  29
        39  47  49   26  34
        27  31  26   21  15
        38  44  44   24  30 ];
    
N2 = [   13  26  25  17  24 
        31  46  40  26  37 
        26  40  19  25  25 
        16  25  27  14  23 
        24  35  18  21  22  ];

  
M = copy(M2); N = copy(N2);

@time M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl  = klf(M, N, fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [0] && μr == [1] && νi == [1] && νl == [1] && μl == [0]  && nf == 3


M = copy(M2); N = copy(N2);

@time M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl  = klf(M, N, fast = fast, finite_infinite = true)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [0] && μr == [1] && νi == [1] && νl == [1] && μl == [0]  && nf == 3

M = copy(M2); N = copy(N2);

@time N1, M1, Q1, Z1, νr, μr, νi, nf, νl, μl  = klf(N, M, fast = fast)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [0] && μr == [1] && νi == [1, 1] && νl == [1] && μl == [0]  && nf == 2      


M = copy(M2); N = copy(N2);

@time N1, M1, Q1, Z1, νr, μr, νi, nf, νl, μl  = klf(N, M, fast = fast, finite_infinite = true)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [0] && μr == [1] && νi == [1, 1] && νl == [1] && μl == [0]  && nf == 2      


# Test data for MB04UD
M2 = [
   2.0  0.0  2.0 -2.0
   0.0 -2.0  0.0  2.0
   2.0  0.0 -2.0  0.0
   2.0 -2.0  0.0  2.0];
N2 = [
   1.0  0.0  1.0 -1.0
   0.0 -1.0  0.0  1.0
   1.0  0.0 -1.0  0.0
   1.0 -1.0  0.0  1.0];


M = copy(M2); N = copy(N2);

@time M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl  = klf(M, N, fast = fast)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [] && μr == [] && νi == [] && νl == [] && μl == []  && nf == 4

# Test data for MB04VD
M2 = [
   1.0  0.0 -1.0  0.0
   1.0  1.0  0.0 -1.0];
N2 = [
   0.0 -1.0  0.0  0.0
   0.0 -1.0  0.0  0.0];

M = copy(M2); N = copy(N2);

@time M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl  = klf(M, N, fast = fast)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [1, 0] && μr == [2, 1] && νi == [1] && νl == [] && μl == []  && nf == 0

# Test Suite 2 (Example 2.2.1, Beelen)
M = zeros(14,16); 
i1 = [1 2 3 6 7 8 9 10 11 12 13 13 14]; 
j1 = [4 6 7 8 9 10 11 12 13 14 15 16 16];
aval = [1 1 1 1 1 1 1 1 1 2 3 1 3];
for k = 1:length(i1)
    M[i1[k],j1[k]] = aval[k]
end
N = zeros(14,16); 
i1 = [1 2 3 5 6 7 10 12 13 14]; j1 = [3 5 6 8 9 10 13 14 15 16];
for k = 1:length(i1)
    N[i1[k],j1[k]] = 1
end
 
Q = qr(rand(14,14)).Q; Z = qr(rand(16,16)).Q;
M2 = Q*M*Z; N2 = Q*N*Z;


M = copy(M2); N = copy(N2); 
@time M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl = klf(M, N, fast = fast, ut = false, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [2, 1, 0] && μr == [4, 2, 1] && νi == [1, 2] && νl == [1, 1, 1, 2] && μl == [0, 1, 1, 1] && nf == 3

M = copy(M2); N = copy(N2); 
@time M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl = klf(M, N, fast = fast, ut = true, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [2, 1, 0] && μr == [4, 2, 1] && νi == [1, 2] && νl == [1, 1, 1, 2] && μl == [0, 1, 1, 1] && nf == 3


M = copy(M2); N = copy(N2);
@time M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl  = klf(M, N, fast = fast, finite_infinite = true, ut = false, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [2, 1, 0] && μr == [4, 2, 1] && νi == [2, 1] && νl == [1, 1, 1, 2] && μl == [0, 1, 1, 1] && nf == 3

M = copy(M2); N = copy(N2);
@time M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl  = klf(M, N, fast = fast, finite_infinite = true, ut = true, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [2, 1, 0] && μr == [4, 2, 1] && νi == [2, 1] && νl == [1, 1, 1, 2] && μl == [0, 1, 1, 1] && nf == 3


M = copy(M2'); N = copy(N2'); 
@time M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl = klf(M, N, fast = fast, ut = false, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2'*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2'*Z1-N1) < sqrt(eps(1.)) &&
      νr == [1, 1, 1, 0] && μr == [2, 1, 1, 1] && νi == [1, 2] && νl == [1, 2, 4] && μl == [0, 1, 2] && nf == 3

M = copy(M2'); N = copy(N2'); 
@time M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl = klf(M, N, fast = fast, ut = true, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2'*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2'*Z1-N1) < sqrt(eps(1.)) &&
      νr == [1, 1, 1, 0] && μr == [2, 1, 1, 1] && νi == [1, 2] && νl == [1, 2, 4] && μl == [0, 1, 2] && nf == 3


M = copy(M2'); N = copy(N2'); 
@time M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl = klf(M, N, fast = fast, finite_infinite = true, ut = false, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2'*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2'*Z1-N1) < sqrt(eps(1.)) &&
      νr == [1, 1, 1, 0] && μr == [2, 1, 1, 1] && νi == [2, 1] && νl == [1, 2, 4] && μl == [0, 1, 2] && nf == 3

M = copy(M2'); N = copy(N2'); 
@time M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl = klf(M, N, fast = fast, finite_infinite = true, ut = true, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2'*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2'*Z1-N1) < sqrt(eps(1.)) &&
      νr == [1, 1, 1, 0] && μr == [2, 1, 1, 1] && νi == [2, 1] && νl == [1, 2, 4] && μl == [0, 1, 2] && nf == 3


# intermediary computation
M2 = [
    1.0  -1.14463e-18  -1.47479e-15  4.87132e-15  6.96416e-31  -2.32652e-31  -1.50216e-16   1.69785e-16   1.42103e-16   7.77705e-17
    0.0   1.0           0.0          0.0          0.0           0.0          -2.34988e-16  -3.55922e-16  -6.84622e-17  -6.11787e-17
    0.0   0.0           1.0          0.0          0.0           0.0           4.90878e-16   3.80428e-16   3.62292e-16   3.74223e-16
    0.0   0.0           0.0          1.0          0.0           0.0           8.12392e-16   5.94559e-16   1.17495e-16   6.02872e-16
    0.0   0.0           0.0          0.0          0.0           0.0           1.0           5.10024e-16  -2.95823e-31  -2.69854e-16
    0.0   0.0           0.0          0.0          0.0           0.0           0.0           1.0           0.0           5.41115e-16 
]
N2 = [
    0.0  0.0  0.0  0.0  0.0  0.0  -1.29895e-15   4.68167e-15   1.51642e-15  -2.88014e-17
    0.0  0.0  0.0  0.0  0.0  0.0   0.689011     -0.0434045     0.72345      -3.72429e-16
    0.0  0.0  0.0  0.0  0.0  0.0   0.723189      0.10666      -0.682364     -7.51122e-18
    0.0  0.0  0.0  0.0  0.0  0.0  -0.0475455     0.993348      0.10488       1.43459e-17
    0.0  0.0  0.0  0.0  0.0  0.0   0.0           0.0           0.0          -5.10024e-16
    0.0  0.0  0.0  0.0  0.0  0.0   0.0           0.0           0.0          -1.0   
]

M = copy(M2); N = copy(N2); 
@time M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl = klf(M, N, fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [2, 1, 0] && μr == [4, 2, 1] && νi == [1, 2] && νl == [] && μl == [] && nf == 0      
     

M = copy(M2); N = copy(N2); 
@time M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl = klf(M', N', fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2'*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2'*Z1-N1) < sqrt(eps(1.)) &&
      μl == [0, 1, 2] && νl == [1, 2, 4] && νi == [1, 2] && νr == [] && μr == [] && nf == 0   
    
      
# Example  Karathanasi and Karampetakis IMA J. Math. Contr Inform. (2019)
N2 = [
1. 1 0 0 1 0 -1 0
1 1 1 1 0 0 0 0
-1 1 0 0 0 0 0 0
1 0 0 0 0 -1 0 0
0 -2 0 0 0 1 1 0
1 0 0 0 0 -1 0 0
]
M2 = [
2. 1 0 0 0 -2 0 0
0 2 1 0 0 0 -2 0
-2 1 0 0 0 1 -1 1
3 -1 1 1 0 -2 2 0
0 -2 0 0 0 1 2 -1
2 -1 0 0 0 -2 1 0
]

M = copy(M2); N = copy(N2); 
@time M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl = klf(M, N, fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [2, 0] && μr == [2, 2] && νi == [1, 1] && νl == [] && μl == [] && nf == 2

M = copy(M2); N = copy(N2); 
@time M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl = klf(M', N', fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2'*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2'*Z1-N1) < sqrt(eps(1.)) &&
      νl == [2, 2] && μl == [0, 2] && νi == [1, 1] && νr == [] && μr == [] && nf == 2


Ty = Float64; 
#for Ty in (Float64, Float32, Complex{Float64},  Complex{Float32})
for Ty in (Float64, Complex{Float64}, BigFloat, Complex{BigFloat})

abstol = sqrt(eps(one(real(Ty))))

# generic cases 
m = 1; n = 2;
M2 = rand(Ty,m,n)
N2 = rand(Ty,m,n)
M = copy(M2); N = copy(N2);
@time M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl = klf(M, N, fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [1, 0] && μr == [1, 1] && νi == [] && νl == [] && μl == [] && nf == 0

m = 3; n = 5;
M2 = rand(Ty,m,n)
N2 = rand(Ty,m,n)
M = copy(M2); N = copy(N2);
@time M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl = klf(M, N, fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [2, 1, 0] && μr == [2, 2, 1] && νi == [] && νl == [] && μl == [] && nf == 0
          
m = 5; n = 3; 
M2 = rand(Ty,m,n)
N2 = rand(Ty,m,n)
M = copy(M2); N = copy(N2);
@time M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl = klf(M, N, fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [] && μr == [] && νi == [] && νl == [1, 2, 2] && μl == [0, 1, 2] && nf == 0

m = 3; n = 5; r = 2;
M2 = rand(Ty,m,n)
N2 = rand(Ty,m,m)*[rand(Ty,r,r) zeros(Ty,r,n-r); zeros(Ty,m-r,n)]*rand(Ty,n,n)
M = copy(M2); N = copy(N2);
@time M1, N1, Q, Z, νr, μr, νi, nf, νl, μl = klf(M, N, fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q'*M2*Z-M1) < abstol &&
      norm(Q'*N2*Z-N1) < abstol &&
      νr == [2, 0] && μr == [2, 2] && νi == [1] && νl == [] && μl == [] && nf == 0

mr = 2; nr = mr+1; ni = 4; nf = 10; nl = 4; ml = nl+1; 
mM = mr+ni+nf+ml;
nM = nr+ni+nf+nl;
M2 = [rand(Ty,mr,nM); 
     zeros(Ty,ni,nr) Matrix{Ty}(I,ni,ni) rand(Ty,ni,nf+nl); 
     zeros(Ty,nf,nr+ni) rand(Ty,nf,nf+nl);
     zeros(Ty,ml,nr+ni+nf) rand(Ty,ml,nl)]
N2 = [rand(Ty,mr,nM); 
     zeros(Ty,ni,nr) diagm(1 => ones(Ty,ni-1)) rand(Ty,ni,nf+nl); 
     zeros(Ty,nf,nr+ni) rand(Ty,nf,nf+nl);
     zeros(Ty,ml,nr+ni+nf) rand(Ty,ml,nl)]
# Q = qr(rand(Ty,mM,mM)).Q;
# Z = qr(rand(Ty,nM,nM)).Q; 
# M2 = Q*M2*Z;
# N2 = Q*N2*Z;

M = copy(M2); N = copy(N2);
atol1 = 1.e-6; atol2 = 1.e-7; rtol = 1.e-7;
@time M1, N1, Q1, Z1, νr, μr, νi, nfe, νl, μl = klf(M, N, finite_infinite=true,fast = fast, atol1 = atol1, atol2 = 0*atol2, rtol = rtol)
@test norm(Q1'*M2*Z1-M1) < atol1 &&
      norm(Q1'*N2*Z1-N1) < atol2 &&
      νr == [ones(Int,mr); [0]] && μr == ones(Int,nr) && 
      νi == ones(Int,ni) && νl == ones(Int,ml) && μl == [[0]; ones(Int,nl)] && nfe == nf

M = copy(M2); N = copy(N2);
atol1 = 1.e-6; atol2 = 1.e-7; rtol = 1.e-7;
@time M1, N1, Q1, Z1, νr, μr, νi, nfe, νl, μl = klf(M, N, finite_infinite=false,fast = fast, atol1 = atol1, atol2 = 0*atol2, rtol = rtol)
@test norm(Q1'*M2*Z1-M1) < atol1 &&
      norm(Q1'*N2*Z1-N1) < atol2 &&
      νr == [ones(Int,mr); [0]] && μr == ones(Int,nr) && 
      νi == ones(Int,ni) && νl == ones(Int,ml) && μl == [[0]; ones(Int,nl)] && nfe == nf

end
end

end

end
end
end
   

