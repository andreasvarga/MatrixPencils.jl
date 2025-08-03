module Test_regular

using LinearAlgebra
using SparseArrays
using MatrixPencils
using Test
using GenericLinearAlgebra
using GenericSchur


@testset "Regular Matrix Pencils Utilities" begin

@testset "regbalance!" begin

n = 50;  k = 11; 
for k in 11:5:31

lambda = exp.(im.*rand(n));
la = real(lambda); li = imag(lambda) 
D = sort(la./li)

La = Diagonal(la); Li = Diagonal(li); 
Tl = randn(n,n).^k; Tr = randn(n,n).^k;
A = Tl*La*Tr; E = Tl*Li*Tr;
ev = sort(eigvals(A,E),by=real)
if any(isinf.(ev))
   corig = Inf
else
   corig = norm(abs.(ev-D)./sqrt.(1. .+ ev .^2)./sqrt.(1. .+ D .^2))
end

qsorig = qS1(abs.(A)+abs.(E))
AA = copy(A); EE = copy(E); 
@time D1, D2 = regbalance!(AA,EE)
@test AA == D1*A*D2 && EE == D1*E*D2 
qsfin = qS1(abs.(AA)+abs.(EE))
@test qsfin < qsorig 
evs = sort(eigvals(AA,EE),by=real)
#@test evs ≈ D
# compute the chordal distance between exact and computed eigenvalues
cofin = norm(abs.(evs-D)./sqrt.(1. .+ evs .^2)./sqrt.(1. .+ D .^2))
@test ev ≈ D ? cofin/10 < corig :  cofin < corig 
end



n = 50;  k = 10; 
for k in (10,20)

kappa = 10^k
lambda = exp.(im.*rand(n));
la = real(lambda); li = imag(lambda) 
D = sort(la./li)
La = Diagonal(la); Li = Diagonal(li); 



factor = kappa^(-1/(n-1)); 
sigma = factor.^(0:n-1)
Dl = Diagonal(sigma); Dr = Diagonal(sigma);
Tl = Dl*qr(rand(n,n)).Q
Tr = qr(rand(n,n)).Q*Dr

A = Tl*La*Tr; E = Tl*Li*Tr;
ev = sort(eigvals(A,E),by=real)
if any(isinf.(ev))
   corig = Inf
else
   corig = norm(abs.(ev-D)./sqrt.(1. .+ ev .^2)./sqrt.(1. .+ D .^2))
end
qsorig = qS1(abs.(A)+abs.(E))
AA = copy(A); EE = copy(E); 
@time D1, D2 = regbalance!(AA,EE)
@test AA == D1*A*D2 && EE == D1*E*D2 
qsfin = qS1(abs.(AA)+abs.(EE))
@test qsfin < qsorig 
evs = sort(eigvals(AA,EE),by=real)
# compute the chordal distance between exact and computed eigenvalues
cofin = norm(abs.(evs-D)./sqrt.(1. .+ evs .^2)./sqrt.(1. .+ D .^2))
#println("k = $k cofin = $cofin")
@test cofin < corig
end 


# examples from Dopico et al. SIMAX, 43:1213-1237, 2022.    
n = 50; k = 10; 
for k in (-16 : 5 : 15)

T = rand(n,n);
T[1, 2 : n] = 10. ^(k)*T[1, 2 : n]
T[4:n, 3] = 10. ^(k)*T[4:n, 3]
D = sort(round.(Int,1. ./ rand(n)))
A = T*Diagonal(D); E = T;
ev = sort(eigvals(A,E),by=real)
#println("norm(evs-ev) = $(norm(ev-D))")
# compute the chordal distance between exact and computed eigenvalues
corig = norm(abs.(ev-D)./sqrt.(1. .+ ev .^2)./sqrt.(1. .+ D .^2))
#println("k = $k corig = $corig")

# maxiter = 100; tol = 0.01;
# dleft,dright = MatrixPencils.rowcolsums(abs.(A).+abs.(E)) 

# evs = eigvals(Diagonal(dleft)*A*Diagonal(dright),Diagonal(dleft)*E*Diagonal(dright))
# @test !(sort(ev) ≈ sort(D)) &&  sort(evs) ≈ sort(D)

# M = abs.(A)+abs.(E); M[2,:] .= 0; M[4,:] .= 0; M[:,4] .= 0;
# MM = copy(M);
# dleft, dright = rcsumsbal!(MM)
# @test dleft*M*dright ≈ MM
# @test 100000*qS1(MM) < qS1(M)


qsorig = qS1(abs.(A)+abs.(E))
AA = copy(A); EE = copy(E); 
@time D1, D2 = regbalance!(AA,EE)
@test AA == D1*A*D2 && EE == D1*E*D2 
qsfin = qS1(abs.(AA)+abs.(EE))
@test qsfin < qsorig 
#println("qsorig/qsfin = $(qsorig/qsfin)")
@test 2. .^round.(Int,log2.(D1.diag)) == D1.diag && 2. .^round.(Int,log2.(D2.diag)) == D2.diag
evs = sort(eigvals(AA,EE),by=real)
#@test evs ≈ D
# compute the chordal distance between exact and computed eigenvalues
cofin = norm(abs.(evs-D)./sqrt.(1. .+ evs .^2)./sqrt.(1. .+ D .^2))
#println("k = $k cofin = $cofin")
#@test cofin < corig 
@test ev ≈ D ? cofin/10 < corig :  cofin < corig 


AA = copy(A); EE = copy(E); 
@time D1, D2 = regbalance!(AA,EE; pow2 = false)
@test AA ≈ D1*A*D2 && EE ≈ D1*E*D2 
qsfin = max(qS1(AA),qS1(EE))
@test qsfin < qsorig 
#println("qsorig/qsfin = $(qsorig/qsfin)")
evs = sort(eigvals(AA,EE),by=real)
@test evs ≈ D
cofin = norm(abs.(evs-D)./sqrt.(1. .+ evs .^2)./sqrt.(1. .+ D .^2))
#println("k = $k cofin = $cofin")
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
@time D1, D2 = regbalance!(AA,EE; pow2 = false)
@test AA ≈ D1*A*D2 && EE ≈ D1*E*D2 
qsfin = qS1(abs.(AA)+abs.(EE))
ev = sort(eigvals(A,E),by=real)
corig = norm(abs.(ev-D)./sqrt.(1. .+ ev .^2)./sqrt.(1. .+ D .^2))
#println("corig = $corig")
evs = sort(eigvals(AA,EE),by=real)
cofin = norm(abs.(evs-D)./sqrt.(1. .+ evs .^2)./sqrt.(1. .+ D .^2))
#println("cofin = $cofin")
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
@time D1, D2 = regbalance!(AA,EE; pow2 = false)
@test AA ≈ D1*A*D2 && EE ≈ D1*E*D2 
qsfin = qS1(abs.(AA)+abs.(EE))
@test 100000*qsfin < qsorig 


As = sparse(A); Es = sparse(E)
qsorig = qS1(abs.(As)+abs.(Es))
AA = copy(As); EE = copy(Es); 
@time D1, D2 = regbalance!(AA,EE; pow2 = false)
@test AA ≈ D1*As*D2 && EE ≈ D1*Es*D2 
qsfin = qS1(abs.(AA)+abs.(EE))
@test 100000*qsfin < qsorig 

end


@testset "_svdlikeAE" begin

for fast in (true, false)

A2 = zeros(0,0); E2 = zeros(0,0); C2 = zeros(0,0); B2 = missing;
A = copy(A2); E = copy(E2); C = copy(C2); B = missing;
Q = nothing
Z = nothing


@time rE, rA22  = _svdlikeAE!(A, E, Q, Z, B, C, fast = fast)
@test rE == 0 && rA22 == 0

A2 = zeros(0,0); E2 = zeros(0,0); B2 = zeros(0,2); C2 = missing;
A = copy(A2); E = copy(E2); B = copy(B2); C = missing;

Q = nothing
Z = nothing


@time rE, rA22  = _svdlikeAE!(A, E, Q, Z, B, C, fast = fast)
@test rE == 0 && rA22 == 0


n = 9;
A2 = [
    1.     0     0     0     0     0     0     0     0
    0     1     0     0     0     0     0     0     0
    0     0     1     0     0     0     0     0     0
    0     0     0     1     0     0     0     0     0
    0     0     0     0     1     0     0     0     0
    0     0     0     0     0     1     0     0     0
    0     0     0     0     0     0     1     0     0
    0     0     0     0     0     0     0     1     0
    0     0     0     0     0     0     0     0     1];
E2 = [
    0.     0     0     0     0     0     0     0     0
    1     0     0     0     0     0     0     0     0
    0     1     0     0     0     0     0     0     0
    0     0     1     0     0     0     0     0     0
    0     0     0     1     0     0     0     0     0
    0     0     0     0     1     0     0     0     0
    0     0     0     0     0     0     0     0     0
    0     0     0     0     0     0     1     0     0
    0     0     0     0     0     0     0     1     0];
B2 = [
    -1.     0     0
     0     0     0
     0     0     0
     0    -1     0
     0     0     0
     0     0     0
     0     0    -1
     0     0     0
     0     0     0];
C2 = [
    0.     1     1     0     3     4     0     0     2
    0     1     0     0     4     0     0     2     0
    0     0     1     0    -1     4     0    -2     2];

A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2); 
T = Float64;
Q = Matrix{T}(I,n,n)
Z = Matrix{T}(I,n,n)

@time rE, rA22  = _svdlikeAE!(A, E, Q, Z, B, C, fast = fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      (ismissing(C) || norm(C2*Z-C)  < sqrt(eps(1.))) && 
      rE == 7 && rA22 == 0

n = 9;

A2 = [
    1.     0     0     0     0     0     0     0     0
    0     1     0     0     0     0     0     0     0
    0     0     1     0     0     0     0     0     0
    0     0     0     1     0     0     0     0     0
    0     0     0     0     1     0     0     0     0
    0     0     0     0     0     1     0     0     0
    0     0     0     0     0     0     1     0     0
    0     0     0     0     0     0     0     1     0
    0     0     0     0     0     0     0     0     1];
E2 = [
    0.     0     0     0     0     0     0     0     0
    1     0     0     0     0     0     0     0     0
    0     1     0     0     0     0     0     0     0
    0     0     1     0     0     0     0     0     0
    0     0     0     1     0     0     0     0     0
    0     0     0     0     1     0     0     0     0
    0     0     0     0     0     0     0     0     0
    0     0     0     0     0     0     1     0     0
    0     0     0     0     0     0     0     1     0];
B2 = [
    -1.     0     0
     0     0     0
     0     0     0
     0    -1     0
     0     0     0
     0     0     0
     0     0    -1
     0     0     0
     0     0     0];
C2 = [
    0.     1     1     0     3     4     0     0     2
    0     1     0     0     4     0     0     2     0
    0     0     1     0    -1     4     0    -2     2]; 

T = Float64;
A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); 

Q = Matrix{T}(I,n,n)
Z = Matrix{T}(I,n,n)

@time rE, rA22  = _svdlikeAE!(A, E, Q, Z, B, C, fast = fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      (ismissing(C) || norm(C2*Z-C)  < sqrt(eps(1.))) && 
      rE == 7 && rA22 == 0

# TG01ED EXAMPLE PROGRAM DATA
n = 4;
A2 = [
    -1.     0     0     3
     0     0     1     2
     1     1     0     4
     0     0     0     0];
 E2 = [
     1.     2     0     0
     0     1     0     1
     3     9     6     3
     0     0     2     0];
 B2 = [
     1.     0
     0     0
     0     1
     1     1];
 C2 = [
    -1.     0     1     0
     0     1    -1     1];

T = Float64;
A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); 

Q = Matrix{T}(I,n,n)
Z = Matrix{T}(I,n,n)

@time rE, rA22  = _svdlikeAE!(A, E, Q, Z, B, C, fast = fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      (ismissing(C) || norm(C2*Z-C)  < sqrt(eps(1.))) && 
      rE == 3 && rA22 == 1

A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); 

Q = Matrix{T}(I,n,n)
Z = Matrix{T}(I,n,n)

@time rE, rA22  = _svdlikeAE!(A, E, Q, Z, B, C, fast = fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      (ismissing(C) || norm(C2*Z-C)  < sqrt(eps(1.))) && 
      rE == 3 && rA22 == 1

n = 4;
A2 = [
    -3.330669073875470e-16                         0                         0                         0
    -5.336424998587595e-17     4.999999999999997e-01                         0                         0
     7.882634225314346e-01     4.926646390821473e-02     1.017643702629568e+00                         0
    -7.804970910378840e-02    -5.073231091746246e-01    -3.602232247129060e-01     1.906644727694980e+00
     ];
 E2 = [
     9.708552809960075e-34    -4.715554847257422e-19     7.905694150420953e-01                         0
    -1.002546820843656e-17    -9.625586024907403e-20     9.682458365518573e-02     4.743416490252569e-01
    -9.878387353077869e-19     2.181405497860925e-19    -3.720759787739367e-01     4.673827146373172e-02
     1.017230340487078e-17     7.512741891975197e-20    -6.045704470706158e-02    -4.812889603890242e-01
     ];
 B2 = [
    -1.171404622872256e-16    -6.123724356957946e-01    -2.056086568456411e-17     2.775557561562891e-16    -2.844390262617513e-16
    -5.000000000000002e-01     1.250000000000000e-01    -5.000000000000003e-01     4.139921271247625e-17     4.999999999999999e-01
    -4.926646390821468e-02    -4.803480231050927e-01    -4.926646390821469e-02     7.882634225314342e-01     4.926646390821479e-02
    -4.975668955366508e-01    -7.804970910378836e-02    -4.975668955366509e-01    -7.804970910378847e-02    -5.073231091746250e-01
     ];
 C2 = [
     9.999999999999998e-01                         0                         0                         0
    -9.799563198300907e-17     9.999999999999998e-01                         0                         0
     ];

T = Float64;
# A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); 

# Q = Matrix{T}(I,n,n)
# Z = Matrix{T}(I,n,n)

# @time rE, rA22  = _svdlikeAE!(A, E, Q, Z, B, C)
# @test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
#       norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
#       (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
#       (ismissing(C) || norm(C2*Z-C)  < sqrt(eps(1.))) && 
#       rE == 2 && rA22 == 1

A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); 

Q = Matrix{T}(I,n,n)
Z = Matrix{T}(I,n,n)

@time rE, rA22  = _svdlikeAE!(A, E, Q, Z, B, C, fast = fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      (ismissing(C) || norm(C2*Z-C)  < sqrt(eps(1.))) && 
      rE == 2 && rA22 == 1

Ty = BigFloat
for Ty in (Float64, Complex{Float64}, BigFloat)

n = 10; m = 3; p = 2;
rE = 5;
rA22 = min(2,n-rE)
n2 = rE+rA22
n3 = n-n2
A2 = [rand(Ty,rE,n);
rand(Ty,rA22,n2) zeros(Ty,rA22,n3);
rand(Ty,n3,rE) zeros(Ty,n3,n-rE) ]; 
E2 = [rand(Ty,rE,rE) zeros(Ty,rE,n-rE);
zeros(Ty,n-rE,n)]
Q = qr(rand(Ty,n,n)).Q
Z = qr(rand(Ty,n,n)).Q
A2 = Q*A2*Z; E2 = Q*E2*Z;
B2 = rand(Ty,n,m); C2 = rand(Ty,p,n);

A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); 
Q = Matrix{Ty}(I,n,n)
Z = Matrix{Ty}(I,n,n)

@time rE1, rA221  = _svdlikeAE!(A, E, Q, Z, B, C, fast = fast, atol1=1.e-7, atol2 = 1.e-7)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      (ismissing(C) || norm(C2*Z-C)  < sqrt(eps(1.))) && 
      rE1 == rE && rA221 == rA22

end
end

end

# test to enforce the call with lanv2
@testset "ordeigvals" begin
a = rand(2,2);
ev =  eigvals(a)
evord = ordeigvals(a)
@test sort(real(ev)) ≈ sort(real(evord)) && sort(imag(ev)) ≈ sort(imag(evord)) 

a = rand(2,2); e = triu(rand(2,2))
ev =  eigvals(a,e)
evord = ordeigvals(a,e)[1]
@test sort(real(ev)) ≈ sort(real(evord)) && sort(imag(ev)) ≈ sort(imag(evord)) 

end # ordeigvals



@testset "isregular" begin

M = zeros(0,0); N = zeros(0,0);
@test isregular(M, N) 


M = zeros(3,0); N = zeros(3,0);
@test !isregular(M, N) 

M = zeros(0,3); N = zeros(0,3);
@test !isregular(M, N) 

M = zeros(1,1); N = ones(1,1);
@test isregular(M, N) 

N = zeros(1,1); M = ones(1,1);
@test isregular(M, N) 

M = [1 0;0 1]; N = [0 1; 0 0]; 
@test isregular(M, N) 

@test isregular(BigFloat.(M), BigFloat.(N)) 

M = [  22  34  31   31  17
        45  45  42   19  29
        39  47  49   26  34
        27  31  26   21  15
        38  44  44   24  30 ];
    
N = [   13  26  25  17  24 
        31  46  40  26  37 
        26  40  19  25  25 
        16  25  27  14  23 
        24  35  18  21  22  ];
@test !isregular(M, N) 
@test !isregular(BigFloat.(M), BigFloat.(N)) 

Ty = Float64
for Ty in (Float64, Complex{Float64}, BigFloat)

    abstol = sqrt(eps(one(real(Ty))))
    
    # given structure 
    mr = 2; nr = 3; ni = 2; nf = 10; ml = 5; nl = 4;
    mM = mr+ni+nf+ml;
    nM = nr+ni+nf+nl;
    M = [rand(Ty,mr,nM); 
         zeros(Ty,ni,nr) Matrix{Ty}(I,ni,ni) rand(Ty,ni,nf+nl); 
         zeros(Ty,nf,nr+ni) rand(Ty,nf,nf+nl);
         zeros(ml,nr+ni+nf) rand(Ty,ml,nl)]
    N = [rand(Ty,mr,nM); 
         zeros(Ty,ni,nr) diagm(1 => ones(Ty,ni-1)) rand(Ty,ni,nf+nl); 
         zeros(Ty,nf,nr+ni) rand(Ty,nf,nf+nl);
         zeros(ml,nr+ni+nf) rand(Ty,ml,nl)]
    Q = qr(rand(Ty,mM,mM)).Q;
    Z = qr(rand(Ty,nM,nM)).Q; 
    M = Q*M*Z;
    N = Q*N*Z;
    
    @test !isregular(M, N, atol1 = abstol, atol2 = abstol)
    
   # given structure 
   mr = 0; nr = 0; ni = 4; nf = 10; ml = 0; nl = 0;
   mM = mr+ni+nf+ml;
   nM = nr+ni+nf+nl;
   M = [rand(Ty,mr,nM); 
        zeros(Ty,ni,nr) Matrix{Ty}(I,ni,ni) rand(Ty,ni,nf+nl); 
        zeros(Ty,nf,nr+ni) rand(Ty,nf,nf+nl);
        zeros(ml,nr+ni+nf) rand(Ty,ml,nl)]
   N = [rand(Ty,mr,nM); 
        zeros(Ty,ni,nr) diagm(1 => ones(Ty,ni-1)) rand(Ty,ni,nf+nl); 
        zeros(Ty,nf,nr+ni) rand(Ty,nf,nf+nl);
        zeros(ml,nr+ni+nf) rand(Ty,ml,nl)]
   Q = qr(rand(Ty,mM,mM)).Q;
   Z = qr(rand(Ty,nM,nM)).Q; 
   M = Q*M*Z;
   N = Q*N*Z;
   
   @test isregular(M, N, atol1 = abstol, atol2 = abstol)
   γ = peigvals(M,N)[1][1]
   @test !isregular(M-γ*N, 0*N, atol1 = abstol, atol2 = abstol)
   @test !isregular(N, 0*N, atol1 = abstol, atol2 = abstol)
   @test !isregular(M, N, γ, atol = abstol)
   @test !isregular(M, N, Inf, atol = abstol)
    
end
    
end # isregular

@testset "isunimodular" begin

M = zeros(0,0); N = zeros(0,0);
@test isunimodular(M, N) 


M = zeros(3,0); N = zeros(3,0);
@test !isunimodular(M, N) 

M = zeros(0,3); N = zeros(0,3);
@test !isunimodular(M, N) 

M = zeros(1,1); N = ones(1,1);
@test !isunimodular(M, N) 

N = zeros(1,1); M = ones(1,1);
@test isunimodular(M, N) 

M = [1 0;0 1]; N = [0 1; 0 0]; 
@test isregular(M, N) 

M = [  22  34  31   31  17
        45  45  42   19  29
        39  47  49   26  34
        27  31  26   21  15
        38  44  44   24  30 ];
    
N = [   13  26  25  17  24 
        31  46  40  26  37 
        26  40  19  25  25 
        16  25  27  14  23 
        24  35  18  21  22  ];
@test !isunimodular(M, N) 
@test !isunimodular(BigFloat.(M), BigFloat.(N)) 

Ty = Float64
for Ty in (Float64, Complex{Float64}, BigFloat)

    abstol = sqrt(eps(one(real(Ty))))
    
    # given structure 
    mr = 2; nr = 3; ni = 2; nf = 10; ml = 5; nl = 4;
    mM = mr+ni+nf+ml;
    nM = nr+ni+nf+nl;
    M = [rand(Ty,mr,nM); 
         zeros(Ty,ni,nr) Matrix{Ty}(I,ni,ni) rand(Ty,ni,nf+nl); 
         zeros(Ty,nf,nr+ni) rand(Ty,nf,nf+nl);
         zeros(ml,nr+ni+nf) rand(Ty,ml,nl)]
    N = [rand(Ty,mr,nM); 
         zeros(Ty,ni,nr) diagm(1 => ones(Ty,ni-1)) rand(Ty,ni,nf+nl); 
         zeros(Ty,nf,nr+ni) rand(Ty,nf,nf+nl);
         zeros(ml,nr+ni+nf) rand(Ty,ml,nl)]
    Q = qr(rand(Ty,mM,mM)).Q;
    Z = qr(rand(Ty,nM,nM)).Q; 
    M = Q*M*Z;
    N = Q*N*Z;
    
    @test !isunimodular(M, N, atol1 = abstol, atol2 = abstol)
    
   # given structure 
   mr = 0; nr = 0; ni = 10; nf = 0; ml = 0; nl = 0;
   mM = mr+ni+nf+ml;
   nM = nr+ni+nf+nl;
   M = [rand(Ty,mr,nM); 
        zeros(Ty,ni,nr) Matrix{Ty}(I,ni,ni) rand(Ty,ni,nf+nl); 
        zeros(Ty,nf,nr+ni) rand(Ty,nf,nf+nl);
        zeros(ml,nr+ni+nf) rand(Ty,ml,nl)]
   N = [zeros(Ty,mr,nM); 
        zeros(Ty,ni,nr) diagm(1 => ones(Ty,ni-1)) rand(Ty,ni,nf+nl); 
        zeros(Ty,nf,nr+ni) rand(Ty,nf,nf+nl);
        zeros(ml,nr+ni+nf) zeros(Ty,ml,nl)]
   Q = qr(rand(Ty,mM,mM)).Q;
   Z = qr(rand(Ty,nM,nM)).Q; 
   M = Q*M*Z;
   N = Q*N*Z;
   
   @test isunimodular(M, N, atol1 = abstol, atol2 = abstol)   
end
    
end # isunimodular


@testset "fisplit" begin
A2 = zeros(0,0); E2 = zeros(0,0); C2 = zeros(0,0); B2 = missing;
A = copy(A2); E = copy(E2); C = copy(C2); B = missing;

@time A1, E1, B1, C1, Q, Z, ν, blkdims  = fisplit(A,E,B,C)
@test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B1) < sqrt(eps(1.))) && 
      (ismissing(C) || norm(C2*Z-C1)  < sqrt(eps(1.))) && 
      ν == [] && blkdims == (0,0)

A2 = zeros(0,0); E2 = zeros(0,0); B2 = zeros(0,2); C2 = missing;
A = copy(A2); E = copy(E2); B = copy(B2); C = missing;

@time A1, E1, B1, C1, Q, Z, ν, blkdims  = fisplit(A,E,B,C)
@test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B1) < sqrt(eps(1.))) && 
      (ismissing(C) || norm(C2*Z-C1)  < sqrt(eps(1.))) && 
      ν == [] && blkdims == (0,0)


A2 = [
    1     0     0     0     0     0     0     0     0
    0     1     0     0     0     0     0     0     0
    0     0     1     0     0     0     0     0     0
    0     0     0     1     0     0     0     0     0
    0     0     0     0     1     0     0     0     0
    0     0     0     0     0     1     0     0     0
    0     0     0     0     0     0     1     0     0
    0     0     0     0     0     0     0     1     0
    0     0     0     0     0     0     0     0     1];
E2 = [
    0     0     0     0     0     0     0     0     0
    1     0     0     0     0     0     0     0     0
    0     1     0     0     0     0     0     0     0
    0     0     1     0     0     0     0     0     0
    0     0     0     1     0     0     0     0     0
    0     0     0     0     1     0     0     0     0
    0     0     0     0     0     0     0     0     0
    0     0     0     0     0     0     1     0     0
    0     0     0     0     0     0     0     1     0];
B2 = [
    -1     0     0
     0     0     0
     0     0     0
     0    -1     0
     0     0     0
     0     0     0
     0     0    -1
     0     0     0
     0     0     0];
C2 = [
    0     1     1     0     3     4     0     0     2
    0     1     0     0     4     0     0     2     0
    0     0     1     0    -1     4     0    -2     2];

A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2); 
@time A1, E1, B1, C1, Q, Z, ν, blkdims  = fisplit(A,E,B,C, finite_infinite=false, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && istriu(A1) &&
      (ismissing(B) || norm(Q'*B2-B1) < sqrt(eps(1.))) && 
      (ismissing(C) || norm(C2*Z-C1)  < sqrt(eps(1.))) && 
      ν == [2, 2, 2, 1, 1, 1] && blkdims == (9,0)  

A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2); 
@time A1, E1, B1, C1, Q, Z, ν, blkdims  = fisplit(A,E,B,C, finite_infinite=true, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && istriu(A1) &&
      (ismissing(B) || norm(Q'*B2-B1) < sqrt(eps(1.))) && 
      (ismissing(C) || norm(C2*Z-C1)  < sqrt(eps(1.))) && 
      ν == [1, 1, 1, 2, 2, 2] && blkdims == (0,9)  


fast = true; finite_infinite = true; Ty = Float64      

for fast in (true, false)

for Ty in (Float64, Complex{Float64}, BigFloat)

A2 = rand(Ty,3,3); E2 = zeros(Ty,3,3); B2 = zeros(Ty,3,2); C2 = zeros(4,3);
A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2); 

@time A1, E1, B1, C1, Q, Z, ν, blkdims  = fisplit(A,E,B,C,fast = fast, finite_infinite = true)
@test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && istriu(A1) &&
      (ismissing(B) || norm(Q'*B2-B1) < sqrt(eps(1.))) && 
      (ismissing(C) || norm(C2*Z-C1)  < sqrt(eps(1.))) && 
      ν == [3] && blkdims == (0,3)   
      
@time A1, E1, B1, C1, Q, Z, ν, blkdims  = fisplit(A,E,B,C,fast = fast, finite_infinite = false)
@test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && istriu(A1) &&
      (ismissing(B) || norm(Q'*B2-B1) < sqrt(eps(1.))) && 
      (ismissing(C) || norm(C2*Z-C1)  < sqrt(eps(1.))) && 
      ν == [3] && blkdims == (3,0)   
      
A2 = rand(3,3); E2 = triu(rand(Ty,3,3),1); B2 = zeros(3,2); C2 = zeros(4,3);
A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2); 

@time A1, E1, B1, C1, Q, Z, ν, blkdims  = fisplit(A,E,B,C,fast = fast, finite_infinite = true)
@test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B1) < sqrt(eps(1.))) && 
      (ismissing(C) || norm(C2*Z-C1)  < sqrt(eps(1.))) && 
      ν == [1] && blkdims == (2,1)    
      
@time A1, E1, B1, C1, Q, Z, ν, blkdims  = fisplit(A,E,B,C,fast = fast, finite_infinite = false)
@test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B1) < sqrt(eps(1.))) && 
      (ismissing(C) || norm(C2*Z-C1)  < sqrt(eps(1.))) && 
      ν == [1] && blkdims == (1,2)    
      

A2 = zeros(3,3); E2 = rand(3,3); B2 = zeros(Ty,3,2); C2 = zeros(4,3);
A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2); 

@time A1, E1, B1, C1, Q, Z, ν, blkdims  = fisplit(A,E,B,C,fast = fast, finite_infinite = true)
@test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B1) < sqrt(eps(1.))) && 
      (ismissing(C) || norm(C2*Z-C1)  < sqrt(eps(1.))) && 
      ν == [] && blkdims == (3,0)   
      
@time A1, E1, B1, C1, Q, Z, ν, blkdims  = fisplit(A,E,B,C,fast = fast, finite_infinite = false)
@test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B1) < sqrt(eps(1.))) && 
      (ismissing(C) || norm(C2*Z-C1)  < sqrt(eps(1.))) && 
      ν == [] && blkdims == (0,3)    
      
A2 = rand(3,3); E2 = rand(3,3); B2 = zeros(Ty,3,2); C2 = zeros(4,3);
A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2); 

@time A1, E1, C1, B1, Q, Z, ν, blkdims  = fisplit(A',E',C',B',fast = fast, finite_infinite = true)
@test norm(Q'*A2'*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2'*Z-E1) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(B2'*Z-B1) < sqrt(eps(1.))) && 
      (ismissing(C) || norm(Q'*C2'-C1)   < sqrt(eps(1.))) && 
      ν == [] && blkdims == (3,0)      
      
@time A1, E1, C1, B1, Q, Z, ν, blkdims  = fisplit(A',E',C',B',fast = fast, finite_infinite = false)
@test norm(Q'*A2'*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2'*Z-E1) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(B2'*Z-B1) < sqrt(eps(1.))) && 
      (ismissing(C) || norm(Q'*C2'-C1)   < sqrt(eps(1.))) && 
      ν == [] && blkdims == (0,3)     
      
end
end
end # fisplit

fast = true; finite_infinite = true; Ty = Float64      

for fast in (true, false)

for Ty in (Float64, Complex{Float64}, BigFloat)

A2 = rand(Ty,3,3); E2 = zeros(Ty,3,3); B2 = zeros(Ty,3,2); C2 = zeros(4,3);
A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2); 

@time A1, E1, B1, C1, Q, Z, ν, blkdims  = sfisplit(A,E,B,C,fast = fast, finite_infinite = true)
@test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && istriu(A1) &&
      (ismissing(B) || norm(Q'*B2-B1) < sqrt(eps(1.))) && 
      (ismissing(C) || norm(C2*Z-C1)  < sqrt(eps(1.))) && 
      ν == Int64[] && blkdims == (3, 0, 0)   
      
@time A1, E1, B1, C1, Q, Z, ν, blkdims  = sfisplit(A,E,B,C,fast = fast, finite_infinite = false)
@test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && istriu(A1) &&
      (ismissing(B) || norm(Q'*B2-B1) < sqrt(eps(1.))) && 
      (ismissing(C) || norm(C2*Z-C1)  < sqrt(eps(1.))) && 
      ν == Int64[] && blkdims == (0,0,3)   
      
A2 = rand(3,3); E2 = triu(rand(Ty,3,3),1); B2 = zeros(3,2); C2 = zeros(4,3);
A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2); 

@time A1, E1, B1, C1, Q, Z, ν, blkdims  = sfisplit(A,E,B,C,fast = fast, finite_infinite = true)
@test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B1) < sqrt(eps(1.))) && 
      (ismissing(C) || norm(C2*Z-C1)  < sqrt(eps(1.))) && 
      ν == Int64[] && blkdims == (1,2,0)    
      
@time A1, E1, B1, C1, Q, Z, ν, blkdims  = sfisplit(A,E,B,C,fast = fast, finite_infinite = false)
@test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B1) < sqrt(eps(1.))) && 
      (ismissing(C) || norm(C2*Z-C1)  < sqrt(eps(1.))) && 
      ν == Int64[] && blkdims == (0, 2, 1)    
      

A2 = zeros(3,3); E2 = rand(3,3); B2 = zeros(Ty,3,2); C2 = zeros(4,3);
A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2); 

@time A1, E1, B1, C1, Q, Z, ν, blkdims  = sfisplit(A,E,B,C,fast = fast, finite_infinite = true)
@test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B1) < sqrt(eps(1.))) && 
      (ismissing(C) || norm(C2*Z-C1)  < sqrt(eps(1.))) && 
      ν == [] && blkdims == (0,3,0)   
      
@time A1, E1, B1, C1, Q, Z, ν, blkdims  = sfisplit(A,E,B,C,fast = fast, finite_infinite = false)
@test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B1) < sqrt(eps(1.))) && 
      (ismissing(C) || norm(C2*Z-C1)  < sqrt(eps(1.))) && 
      ν == [] && blkdims == (0,3,0)    
      
A2 = rand(3,3); E2 = rand(3,3); B2 = zeros(Ty,3,2); C2 = zeros(4,3);
A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2); 

@time A1, E1, C1, B1, Q, Z, ν, blkdims  = sfisplit(A',E',C',B',fast = fast, finite_infinite = true)
@test norm(Q'*A2'*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2'*Z-E1) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(B2'*Z-B1) < sqrt(eps(1.))) && 
      (ismissing(C) || norm(Q'*C2'-C1)   < sqrt(eps(1.))) && 
      ν == [] && blkdims == (0,3,0)      
      
@time A1, E1, C1, B1, Q, Z, ν, blkdims  = sfisplit(A',E',C',B',fast = fast, finite_infinite = false)
@test norm(Q'*A2'*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2'*Z-E1) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(B2'*Z-B1) < sqrt(eps(1.))) && 
      (ismissing(C) || norm(Q'*C2'-C1)   < sqrt(eps(1.))) && 
      ν == [] && blkdims == (0,3,0)     
      
end

end # sfisplit

end # regular testset
end # module