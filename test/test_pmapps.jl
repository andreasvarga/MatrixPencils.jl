module Test_pmapps

using Random
using LinearAlgebra
using MatrixPencils
using Polynomials
using Test


@testset "Polynomial Matrix Applications" begin


for fast in (false,true)

P = zeros(0,0,0); 
val, kinfo  = pmeigvals(P, fast = fast)
@test val == [] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf) == (Int64[], Int64[], Int64[], 0)

P = zeros(0,0,3); 
val, kinfo  = pmeigvals(P, fast = fast)
@test val == [] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf) == (Int64[], Int64[], Int64[], 0)

P = zeros(3,0,3); 
val, kinfo  = pmeigvals(P, fast = fast)
@test val == [] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf) == (Int64[], [0, 0, 0], Int64[], 0)

P = zeros(0,3,3); 
val, kinfo  = pmeigvals(P, fast = fast)
@test val == [] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf) == ([0, 0, 0], Int64[], Int64[], 0)


P = zeros(1,1,5)
val, kinfo  = pmeigvals(P, grade = pmdeg(P), fast = fast)
@test val == [] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf) == (Int64[0], Int64[0], Int64[], 0)

P = zeros(1,1,5)
val, kinfo  = pmeigvals(P, grade = pmdeg(P)+4, fast = fast)
@test val == [] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf) == (Int64[0], Int64[0], Int64[], 0)

P = zeros(1,1,5)
val, kinfo  = pmeigvals(P, CF1 = false, grade = pmdeg(P)+3, fast = fast)
@test val == [] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf) == (Int64[0], Int64[0], Int64[], 0)

P = zeros(1,1,5)
P[1,1,1] = 1. 

val, kinfo  = pmeigvals(P, fast = fast)
@test val == Float64[] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf) == (Int64[], Int64[], Int64[], 0)

val, kinfo  = pmeigvals(P, grade = 1, fast = fast)
@test val == [Inf] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf) == (Int64[], Int64[], Int64[1], 0)


val, kinfo  = pmeigvals(P, grade = pmdeg(P)+4, fast = fast)
@test val == [Inf, Inf, Inf, Inf] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf) == (Int64[], Int64[], Int64[4], 0)

val, kinfo  = pmeigvals(P, CF1 = false, grade = pmdeg(P)+3, fast = fast)
@test val == [Inf, Inf, Inf] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf) == (Int64[], Int64[], Int64[3], 0)

P = zeros(1,1,5)
P[1,1,3] = 1. 
P[1,1,2] = -3. 
P[1,1,1] = 2. 
val, kinfo  = pmeigvals(P, grade = pmdeg(P), fast = fast)
@test (val ≈ [2, 1] || val ≈ [1, 2]) && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf) == (Int64[], Int64[], Int64[], 2) &&
kinfo.nf+sum(kinfo.id)+sum(kinfo.rki)+sum(kinfo.lki) == pmdeg(P)

val, kinfo  = pmeigvals(P, grade = 4, fast = fast)
@test sort(val) ≈ [1, 2, Inf, Inf] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf) == (Int64[], Int64[], Int64[2], 2) &&
      kinfo.nf+sum(kinfo.id)+sum(kinfo.rki)+sum(kinfo.lki) == 4

val, kinfo  = pmeigvals(P, CF1 = false, grade = 3, fast = fast)
@test sort(val) ≈ [1, 2, Inf] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf) == (Int64[], Int64[], Int64[1], 2) &&
kinfo.nf+sum(kinfo.id)+sum(kinfo.rki)+sum(kinfo.lki) == 3

# P = 0

info = pmkstruct(0, fast = fast) 
@test (info.rki, info.lki,info.id, info.nf) == ([0], [0], [], 0)

info = pmkstruct(0,grade = 3, fast = fast) 
@test (info.rki, info.lki,info.id, info.nf) == ([0], [0], [], 0)

val, info = pmeigvals(0, fast = fast) 
@test val ≈ Float64[] && (info.rki, info.lki,info.id, info.nf) == ([0], [0], [], 0)

val, info = pmeigvals(0,grade=1, fast = fast)
@test val ≈ Float64[] && (info.rki, info.lki,info.id, info.nf) == ([0], [0], [], 0)

val, info = pmeigvals(0,grade=3, fast = fast)
@test val ≈ Float64[] && (info.rki, info.lki,info.id, info.nf) == ([0], [0], [], 0)

val, iz, info = pmzeros(0, fast = fast)
@test val ≈ Float64[] && iz == [] && (info.rki, info.lki,info.id, info.nf) == ([0], [0], [], 0)

val, ip, info = pmpoles(0, fast = fast)
@test val ≈ Float64[] && ip == [] && (info.rki, info.lki,info.id, info.nf) == ([], [], [], 0)

@test pmrank(0) == 0 && pmrank(0,fastrank=false) == 0

@test !ispmregular(0) && !ispmregular(0,fastrank=false)

@test !ispmunimodular(0)

# P = 1

info = pmkstruct(1, fast = fast)
@test (info.rki, info.lki,info.id, info.nf) == ([], [], [], 0)

info = pmkstruct(1,grade=3, fast = fast)
@test (info.rki, info.lki,info.id, info.nf) == ([], [], [3], 0)

val, info = pmeigvals(1, fast = fast)
@test val ≈ Float64[] && (info.rki, info.lki,info.id, info.nf) == ([], [], [], 0)

val, info = pmeigvals(1,grade=1, fast = fast)
@test val ≈ [Inf] && (info.rki, info.lki,info.id, info.nf) == ([], [], [1], 0)

val, info = pmeigvals(1,grade=3, fast = fast)
@test val ≈ [Inf, Inf, Inf] && (info.rki, info.lki,info.id, info.nf) == ([], [], [3], 0)

val, iz, info = pmzeros(1, fast = fast)
@test val ≈ Float64[] && iz == [] && (info.rki, info.lki,info.id, info.nf) == ([], [], [], 0)

val, ip, info = pmpoles(1, fast = fast)
@test val ≈ Float64[] && ip == [] && (info.rki, info.lki,info.id, info.nf) == ([], [], [], 0)

@test pmrank(1) == 1 && pmrank(1,fastrank=false) == 1

@test ispmregular(1) && ispmregular(1,fastrank=false)

@test ispmunimodular(1)



# P = λ 

λ = Polynomial([0,1],:λ)

info = pmkstruct(λ, fast = fast)
@test (info.rki, info.lki,info.id, info.nf) == ([], [], [], 1)

info = pmkstruct(λ,grade=3, fast = fast)
@test (info.rki, info.lki,info.id, info.nf) == ([], [], [2], 1)

val, info = pmeigvals(λ, fast = fast)
@test val ≈ [0.] && (info.rki, info.lki,info.id, info.nf) == ([], [], [], 1)

val, info = pmeigvals([λ 1;1 0], fast = fast)
@test val ≈ [Inf, Inf] && (info.rki, info.lki,info.id, info.nf) == ([], [], [2], 0)

val, info = pmeigvals(λ,grade=3, fast = fast)
@test val ≈ [0., Inf, Inf] && (info.rki, info.lki,info.id, info.nf) == ([], [], [2], 1)

val, info = pmeigvals([λ 1;1 0],grade=3, fast = fast)
@test val ≈ [Inf, Inf, Inf, Inf, Inf, Inf] && (info.rki, info.lki,info.id, info.nf) == ([], [], [2, 4], 0)

val, iz, info = pmzeros(λ, fast = fast)
@test val ≈ [0.] && iz == [] && (info.rki, info.lki,info.id, info.nf) == ([], [], [], 1)

val = pmroots(λ, fast = fast)
@test val ≈ [0.] 

val, ip, info = pmpoles(λ, fast = fast)
@test val == [Inf] && ip == [1] && (info.rki, info.lki,info.id, info.nf) == ([], [], [2], 0)

@test pmrank(λ) == 1 && pmrank(λ,fastrank=false) == 1

@test ispmregular(λ) && ispmregular(λ,fastrank=false)

@test !ispmunimodular(λ)

# P = [λ 1] 
λ = Polynomial([0,1],:λ)
P = [λ one(λ)]

info = pmkstruct(P, fast = fast)
@test (info.rki, info.lki,info.id, info.nf) == ([1], [], [], 0)

info = pmkstruct(P,CF1=true, fast = fast)
@test (info.rki, info.lki,info.id, info.nf) == ([1], [], [], 0)

info = pmkstruct(P,grade=3, fast = fast)
@test (info.rki, info.lki,info.id, info.nf) == ([1], [], [2], 0)

nfo = pmkstruct(P,grade=3,CF1=true, fast = fast)
@test (info.rki, info.lki,info.id, info.nf) == ([1], [], [2], 0)

val, info = pmeigvals(P, fast = fast)
@test val == Float64[] && (info.rki, info.lki,info.id, info.nf) == ([1], [], [], 0)

val, info = pmeigvals(P,grade=3, fast = fast)
@test val == [Inf, Inf] && (info.rki, info.lki,info.id, info.nf) == ([1], [], [2], 0)

val, iz, info = pmzeros(P, fast = fast)
@test val == Float64[] && iz == [] && (info.rki, info.lki,info.id, info.nf) == ([1], [], [], 0)

val, ip, info = pmpoles(P, fast = fast)
@test val == [Inf] && ip == [1] && (info.rki, info.lki,info.id, info.nf) == ([], [], [1, 2], 0)

@test pmrank(P) == 1 && pmrank(P,fastrank=false) == 1

@test !ispmregular(P) && !ispmregular(P,fastrank=false)

@test !ispmunimodular(P)


# Example 3: P = [λ^2 λ; λ 1] DeTeran, Dopico, Mackey, ELA 2009
λ = Polynomial([0,1],:λ)
P = [λ^2 λ; λ 1]

info = pmkstruct(P, fast = fast)
@test (info.rki, info.lki,info.id, info.nf) == ([1], [1], [], 0)

info = pmkstruct(P,CF1=true, fast = fast)
@test (info.rki, info.lki,info.id, info.nf) == ([1], [1], [], 0)

info = pmkstruct(P,grade=3, fast = fast)
@test (info.rki, info.lki,info.id, info.nf) == ([1], [1], [1], 0)

nfo = pmkstruct(P,grade=3,CF1=true, fast = fast)
@test (info.rki, info.lki,info.id, info.nf) == ([1], [1], [1], 0)

val, info = pmeigvals(P, fast = fast)
@test val == Float64[] && (info.rki, info.lki,info.id, info.nf) == ([1], [1], [], 0)

val, info = pmeigvals(P,grade=3, fast = fast)
@test val == [Inf] && (info.rki, info.lki,info.id, info.nf) == ([1], [1], [1], 0)

val, iz, info = pmzeros(P, fast = fast)
@test val == Float64[] && iz == [] && (info.rki, info.lki,info.id, info.nf) == ([1], [1], [], 0)

val, ip, info = pmpoles(P, fast = fast)
@test val == [Inf, Inf] && ip == [2] && (info.rki, info.lki,info.id, info.nf) == ([], [], [2, 2, 4], 0)

@test pmrank(P) == 1 && pmrank(P,fastrank=false) == 1

@test !ispmregular(P) && !ispmregular(P,fastrank=false)

@test !ispmunimodular(P)

# Example 4: Van Dooren, Dewilde, LAA, 1983 
P = zeros(2,2,3);
P[:,:,1] = [0 0; 0 1.];
P[:,:,2] = [0 1.; 1. 0];
P[:,:,3] = [1. 0; 0 0]; 
val, info = pmeigvals(P)
@test val == Float64[] && (info.rki, info.lki,info.id, info.nf) == ([1], [1], [], 0)

# Example 4: Van Dooren, Dewilde, LAA, 1983 
P = zeros(Int,3,3,3)
P[:,:,1] = [1 2 -2; 0 -1 -2; 0 0 0]
P[:,:,2] = [1 3 0; 1 4 2; 0 -1 -2]
P[:,:,3] = [1 4 2; 0 0 0; 1 4 2]

val, info = pmeigvals(P, fast = fast)
@test val ≈ [1, Inf, Inf] && (info.rki, info.lki,info.id, info.nf) == ([0], [1], [2], 1)

val, info = pmeigvals(P,grade = 3, fast = fast)
@test val ≈ [1, Inf, Inf, Inf, Inf] && (info.rki, info.lki,info.id, info.nf) == ([0], [1], [1, 3], 1)

val, iz, info = pmzeros(P, fast = fast)
@test val ≈ [1] && iz == [] && (info.rki, info.lki,info.id, info.nf) == ([0], [1], [2], 1)

val, ip, info = pmpoles(P, fast = fast)
@test val ≈ [Inf, Inf] && ip == [2] && (info.rki, info.lki, info.id, info.nf) == ([], [], [2, 2, 2, 2, 4], 0)

val, iz, info = pmzeros1(P, fast = fast)
@test val ≈ [1] && iz == [] && (info.rki, info.lki,info.id, info.nf) == ([0], [1], [1], 1)

val, ip, info = pmpoles1(P, fast = fast)
@test val ≈ [Inf, Inf] && ip == [2] && (info.rki, info.lki, info.id, info.nf) == ([], [], [1, 1, 1, 1, 3], 0)

val, iz, info = pmzeros2(P, fast = fast)
@test val ≈ [1] && iz == [] && (info.rki, info.lki,info.id, info.nf) == ([0], [1], [1, 1, 1, 1], 1)

val, ip, info = pmpoles2(P, fast = fast)
@test val ≈ [Inf, Inf] && ip == [2] && (info.rki, info.lki, info.id, info.nf) == ([], [], [1, 3], 0)

@test !ispmregular(P) && !ispmregular(P,fastrank=false)

@test !ispmunimodular(P)

λ = Polynomial([0,1],:λ);
P = [λ^2 + λ + 1 4λ^2 + 3λ + 2 2λ^2 - 2;
λ 4λ - 1 2λ - 2;
λ^2 4λ^2 - λ 2λ^2 - 2λ]

info = pmkstruct(P, fast = fast)
@test (info.rki, info.lki,info.id, info.nf) == ([0], [1], [2], 1)

info = pmkstruct(P,CF1=true, fast = fast)
@test (info.rki, info.lki,info.id, info.nf) == ([0], [1], [2], 1)

info = pmkstruct(P,grade=3, fast = fast)
@test (info.rki, info.lki,info.id, info.nf) == ([0], [1], [1, 3], 1)

nfo = pmkstruct(P,grade=3,CF1=true, fast = fast)
@test (info.rki, info.lki,info.id, info.nf) == ([0], [1], [1, 3], 1)

val, info = pmeigvals(P, fast = fast)
@test val ≈ [1, Inf, Inf] && (info.rki, info.lki,info.id, info.nf) == ([0], [1], [2], 1)

val, info = pmeigvals(P,grade = 3, fast = fast)
@test val ≈ [1, Inf, Inf, Inf, Inf] && (info.rki, info.lki,info.id, info.nf) == ([0], [1], [1, 3], 1)

val, info = pmeigvals(P,CF1=false, grade = 3, fast = fast)
@test val ≈ [1, Inf, Inf, Inf, Inf] && (info.rki, info.lki,info.id, info.nf) == ([0], [1], [1, 3], 1)

val, iz, info = pmzeros(P, fast = fast)
@test val ≈ [1] && iz == [] && (info.rki, info.lki,info.id, info.nf) == ([0], [1], [2], 1)

val, iz, info = pmpoles(P, fast = fast)
@test val ≈ [Inf, Inf] && iz == [2] && (info.rki, info.lki, info.id, info.nf) == ([], [], [2, 2, 2, 2, 4], 0)

val, iz, info = pmpoles(P,CF1=false, fast = fast)
@test val ≈ [Inf, Inf] && iz == [2] && (info.rki, info.lki, info.id, info.nf) == ([], [], [2, 2, 2, 2, 4], 0)

val, iz, info = pmzeros1(P, fast = fast)
@test val ≈ [1] && iz == [] && (info.rki, info.lki,info.id, info.nf) == ([0], [1], [1], 1)

val, ip, info = pmpoles1(P, fast = fast)
@test val ≈ [Inf, Inf] && ip == [2] && (info.rki, info.lki, info.id, info.nf) == ([], [], [1, 1, 1, 1, 3], 0)

val, iz, info = pmzeros2(P, fast = fast)
@test val ≈ [1] && iz == [] && (info.rki, info.lki,info.id, info.nf) == ([0], [1], [1, 1, 1, 1], 1)

val, ip, info = pmpoles2(P, fast = fast)
@test val ≈ [Inf, Inf] && ip == [2] && (info.rki, info.lki, info.id, info.nf) == ([], [], [1, 3], 0)


@test pmrank(P) == 2 && pmrank(P,fastrank=false) == 2

@test pmrank(rand(2,3,5)) == 2 && pmrank(rand(2,3,5),fastrank=false) == 2

@test pmrank(rand(3,2,5)) == 2 && pmrank(rand(3,2,5),fastrank=false) == 2

@test !ispmregular(P) && !ispmregular(P,fastrank=false)

@test !ispmunimodular(P)


s = Polynomial([0, 1],:x)
P = [ s^2+s+1  4*s^2+3*s+2 2*s^2-2
      s  4*s-1  2*s-2
      s^2 4*s^2-s  2*s^2-2*s ]
U = [-s  0   s^2 + 2*s + 1
s^2 - s + 1  0 -s^3 - s^2 - 1
0 -1 s];
V = [3*s + 1 3*s^3 + s^2 - 3*s - 4  6
-s -s^3 + s + 1  -2
0  0  1];
@test ispmunimodular(U)
@test ispmunimodular(V)





# unimodular matrix example
P = zeros(2,2,2)
P[:,:,1] = [1 0;0 1]; P[:,:,2] = [0. 0; 1 0];
@test ispmunimodular(P)
val, kinfo  = pmeigvals(P, CF1 = true, fast = fast)
@test val == [Inf, Inf] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf) == (Int64[], Int64[], Int64[2], 0)
val, kinfo  = pmeigvals(P, CF1 = false, fast = fast)
@test val == [Inf, Inf] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf) == (Int64[], Int64[], Int64[2], 0)
val, iz, kinfo  = pmzeros(P, CF1 = true, fast = fast)
@test val == [Inf] && iz == [1] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf) == (Int64[], Int64[], Int64[2], 0)
val, iz, kinfo  = pmzeros(P, CF1 = false, fast = fast)
@test val == [Inf] && iz == [1] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf) == (Int64[], Int64[], Int64[2], 0)
val, iz, kinfo  = pmzeros2(P, fast = fast)
@test val == [Inf] && iz == [1] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf) == (Int64[], Int64[], Int64[1, 1, 2], 0)

N = zeros(1,1); M = ones(1,1);
P = zeros(1,1,3)
P[:,:,1] = M; P[:,:,2] = -N;
val, iz, kinfo  = pmzeros2(P, fast = fast)
@test val == [] && iz == [] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf) == (Int64[], Int64[], Int64[1], 0)

M = [1 0;0 1]; N = [0 1; 0 0]; 
P = zeros(2,2,3)
P[:,:,1] = M; P[:,:,2] = -N;
val, kinfo = pmeigvals(P, fast = fast)
@test val == [Inf;Inf] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf) == (Int64[], Int64[], [2], 0)


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

P = zeros(5,5,3)
P[:,:,1] = M; P[:,:,2] = -N;
val, kinfo = pmeigvals(P, fast = fast)
@test (isapprox(val,[2,0,0,Inf],atol = 1.e-7) || isapprox(val,[0,0,2,Inf],atol = 1.e-7)) &&
      (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf) == ([0], [0], [1], 3)



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
P = zeros(size(M)...,3)
P[:,:,1] = M; P[:,:,2] = -N;

val, kinfo = pmeigvals(P, fast = fast, atol = 1.e-7)
@test length(val) == 6 &&
      length(filter(y-> y == true,isinf.(val))) == 3 &&
      length(filter(y-> y == true,isfinite.(val))) == 3 &&
      (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf) == ([0, 0, 1, 2], [0, 3], [1, 2], 3)

for Ty in (Float64, Complex{Float64})

P = rand(Ty,2,3,5);
P1 = rand(Ty,3,2,5);
abstol = sqrt(eps(one(real(Ty))))

info = pmkstruct(P,CF1=false, fast = fast)
@test (info.rki, info.lki,info.id, info.nf) == ([8], [], [], 0)

info = pmkstruct(P,CF1=true, fast = fast)
@test (info.rki, info.lki,info.id, info.nf) == ([8], [], [], 0)

info = pmkstruct(P,grade=5, CF1=false, fast = fast)
@test (info.rki, info.lki,info.id, info.nf) == ([8], [], [1, 1], 0)

info = pmkstruct(P,grade=5,CF1=true, fast = fast)
@test (info.rki, info.lki,info.id, info.nf) == ([8], [], [1, 1], 0)

val, info = pmeigvals(P, fast = fast)
@test val == Ty[] && (info.rki, info.lki,info.id, info.nf) == ([8], [], [], 0)

val, info = pmeigvals(P,CF1=true,grade = 5, fast = fast)
@test val == [Inf, Inf] && (info.rki, info.lki,info.id, info.nf) == ([8], [], [1, 1], 0)

val, info = pmeigvals(P,CF1=false, grade = 5, fast = fast)
@test val == [Inf, Inf] && (info.rki, info.lki,info.id, info.nf) == ([8], [], [1, 1], 0)

val, iz, info = pmzeros(P, fast = fast)
@test val == Ty[] && iz == [] && (info.rki, info.lki,info.id, info.nf) == ([8], [], [], 0)

val, iz, info = pmpoles(P,CF1=false, fast = fast)
@test val == [Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf] && iz == [4, 4] && (info.rki, info.lki, info.id, info.nf) == ([], [], [4, 8, 8], 0)

val, ip, info = pmpoles(P,CF1=true, fast = fast)
@test val == [Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf] && ip == [4, 4] && (info.rki, info.lki, info.id, info.nf) == ([], [], [4, 8, 8], 0)

val, iz, info = pmzeros1(P, fast = fast)
@test val == Ty[] && iz == [] && (info.rki, info.lki,info.id, info.nf) == ([8], [], [], 0)

val, iz, info = pmzeros1(P1, fast = fast)
@test val == Ty[] && iz == [] && (info.rki, info.lki,info.id, info.nf) == ([], [8], [], 0)

val, ip, info = pmpoles1(P, fast = fast, atol = abstol)
@test val == [Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf] && ip == [4, 4] && (info.rki, info.lki, info.id, info.nf) == ([], [], [1, 5, 5], 0)

val, ip, info = pmpoles1(P1, fast = fast, atol = abstol)
@test val == [Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf] && ip == [4, 4] && (info.rki, info.lki, info.id, info.nf) == ([], [], [1, 5, 5], 0)

val, iz, info = pmzeros2(P, fast = fast)
@test val == Ty[] && iz == [] && (info.rki, info.lki,info.id, info.nf) == ([8], [], [1, 1, 1, 1], 0)

val, ip, info = pmpoles2(P, fast = fast)
@test val == [Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf] && ip == [4, 4] && (info.rki, info.lki, info.id, info.nf) == ([], [], [5, 5], 0)


@test pmrank(P) == 2 && pmrank(P,fastrank=false) == 2

@test !ispmregular(P) && !ispmregular(P,fastrank=false)

@test !ispmunimodular(P)

@test ispmregular(rand(Ty,2,2,5)) && ispmregular(rand(Ty,2,2,5),fastrank=false)

@test !ispmunimodular(rand(Ty,2,2,5)) 

@test ispmunimodular(rand(Ty,2,2,1))


end
end
#end

end  # applications testset

end  # module