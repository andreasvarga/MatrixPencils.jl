module Test_pmapps

using Random
using LinearAlgebra
using MatrixPencils
using Polynomials
using Test
using GenericLinearAlgebra
using GenericSchur

println("Test_pmapps")

@testset "Polynomial Matrix Applications" begin


for fast in (false,true)

P = zeros(0,0,0); 
@time @time val, kinfo  = pmeigvals(P, fast = fast)
@test val == [] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == (Int64[], Int64[], Int64[], 0, 0)

P = zeros(0,0,3); 
@time val, kinfo  = pmeigvals(P, fast = fast)
@test val == [] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == (Int64[], Int64[], Int64[], 0, 0)

P = zeros(3,0,3); 
@time val, kinfo  = pmeigvals(P, fast = fast)
@test val == [] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == (Int64[], [0, 0, 0], Int64[], 0, 0)

P = zeros(0,3,3); 
@time val, kinfo  = pmeigvals(P, fast = fast)
@test val == [] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == ([0, 0, 0], Int64[], Int64[], 0, 0)


P = zeros(1,1,5)
@time val, kinfo  = pmeigvals(P, grade = pmdeg(P), fast = fast)
@test val == [] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == (Int64[0], Int64[0], Int64[], 0, 0)

P = zeros(1,1,5)
@time val, kinfo  = pmeigvals(P, grade = pmdeg(P)+4, fast = fast)
@test val == [] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == (Int64[0], Int64[0], Int64[], 0, 0)

P = zeros(1,1,5)
@time val, kinfo  = pmeigvals(P, CF1 = false, grade = pmdeg(P)+3, fast = fast)
@test val == [] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == (Int64[0], Int64[0], Int64[], 0, 0)

P = zeros(1,1,5)
P[1,1,1] = 1. 

@time val, kinfo  = pmeigvals(P, fast = fast)
@test val == Float64[] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == (Int64[], Int64[], Int64[], 0, 1)

@time val, kinfo  = pmeigvals(P, grade = 1, fast = fast)
@test val == [Inf] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == (Int64[], Int64[], Int64[1], 0, 1)


@time val, kinfo  = pmeigvals(P, grade = pmdeg(P)+4, fast = fast)
@test val == [Inf, Inf, Inf, Inf] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == (Int64[], Int64[], Int64[4], 0, 1)

@time val, kinfo  = pmeigvals(P, CF1 = false, grade = pmdeg(P)+3, fast = fast)
@test val == [Inf, Inf, Inf] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == (Int64[], Int64[], Int64[3], 0, 1)

P = zeros(1,1,5)
P[1,1,3] = 1. 
P[1,1,2] = -3. 
P[1,1,1] = 2. 
@time val, kinfo  = pmeigvals(P, grade = pmdeg(P), fast = fast)
@test (val ≈ [2, 1] || val ≈ [1, 2]) && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == (Int64[], Int64[], Int64[], 2, 1) &&
kinfo.nf+sum(kinfo.id)+sum(kinfo.rki)+sum(kinfo.lki) == kinfo.nrank*pmdeg(P)

@time val, kinfo  = pmeigvals(P, grade = 4, fast = fast)
@test sort(val) ≈ [1, 2, Inf, Inf] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == (Int64[], Int64[], Int64[2], 2, 1) &&
      kinfo.nf+sum(kinfo.id)+sum(kinfo.rki)+sum(kinfo.lki) == kinfo.nrank*4

@time val, kinfo  = pmeigvals(P, CF1 = false, grade = 3, fast = fast)
@test sort(val) ≈ [1, 2, Inf] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == (Int64[], Int64[], Int64[1], 2, 1) &&
kinfo.nf+sum(kinfo.id)+sum(kinfo.rki)+sum(kinfo.lki) == kinfo.nrank*3

# P = 0

@time @time info, iz, ip = pmkstruct(0, fast = fast) 
@test (info.rki, info.lki,info.id, info.nf, info.nrank) == ([0], [0], [], 0, 0) && iz == [] && ip == []

@time @time info, iz, ip = pmkstruct(0,grade = 3, fast = fast) 
@test (info.rki, info.lki,info.id, info.nf, info.nrank) == ([0], [0], [], 0, 0) && iz == [] && ip == []

@time @time val, info = pmeigvals(0, fast = fast) 
@test val ≈ Float64[] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([0], [0], [], 0, 0)

@time val, info = pmeigvals(0,grade=1, fast = fast)
@test val ≈ Float64[] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([0], [0], [], 0, 0)

@time val, info = pmeigvals(0,grade=3, fast = fast)
@test val ≈ Float64[] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([0], [0], [], 0, 0)

@time val, iz, info = pmzeros(0, fast = fast)
@test val ≈ Float64[] && iz == [] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([0], [0], [], 0, 0)

@time val, ip, id = pmpoles(0, fast = fast)
@test val ≈ Float64[] && ip == [] && id == []

@test pmrank(0) == 0 && pmrank(0,fastrank=false) == 0

@test !ispmregular(0) && !ispmregular(0,fastrank=false)

@test !ispmunimodular(0)

# P = 1

@time @time info, iz, ip = pmkstruct(1, fast = fast)
@test (info.rki, info.lki,info.id, info.nf, info.nrank) == ([], [], [], 0, 1) && iz == [] && ip == []

@time @time info, iz, ip = pmkstruct(1,grade=3, fast = fast)
@test (info.rki, info.lki,info.id, info.nf, info.nrank) == ([], [], [3], 0, 1) && iz == [] && ip == []

@time val, info = pmeigvals(1, fast = fast)
@test val ≈ Float64[] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([], [], [], 0, 1)

@time val, info = pmeigvals(1,grade=1, fast = fast)
@test val ≈ [Inf] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([], [], [1], 0, 1)

@time val, info = pmeigvals(1,grade=3, fast = fast)
@test val ≈ [Inf, Inf, Inf] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([], [], [3], 0, 1)

@time val, iz, info = pmzeros(1, fast = fast)
@test val ≈ Float64[] && iz == [] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([], [], [], 0, 1)

@time val, ip, id = pmpoles(1, fast = fast)
@test val ≈ Float64[] && ip == [] && id == []

@test pmrank(1) == 1 && pmrank(1,fastrank=false) == 1

@test ispmregular(1) && ispmregular(1,fastrank=false)

@test ispmunimodular(1)



# P = λ 

λ = Polynomial([0,1],:λ)

@time @time info, iz, ip = pmkstruct(λ, fast = fast)
@test (info.rki, info.lki,info.id, info.nf, info.nrank) == ([], [], [], 1, 1) && iz == [] && ip == [1]

@time @time info, iz, ip = pmkstruct(λ,grade=3, fast = fast)
@test (info.rki, info.lki,info.id, info.nf, info.nrank) == ([], [], [2], 1, 1) && iz == [] && ip == [1]

@time val, info = pmeigvals(λ, fast = fast)
@test val ≈ [0.] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([], [], [], 1, 1)

@time val, info = pmeigvals([λ 1;1 0], fast = fast)
@test val ≈ [Inf, Inf] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([], [], [2], 0, 2)

@time val, info = pmeigvals(λ,grade=3, fast = fast)
@test val ≈ [0., Inf, Inf] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([], [], [2], 1, 1)

@time val, info = pmeigvals([λ 1;1 0],grade=3, fast = fast)
@test val ≈ [Inf, Inf, Inf, Inf, Inf, Inf] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([], [], [2, 4], 0, 2)

@time val, iz, info = pmzeros(λ, fast = fast)
@test val ≈ [0.] && iz == [] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([], [], [], 1, 1)

val = pmroots(λ, fast = fast)
@test val ≈ [0.] 

@time @time val, ip, id = pmpoles(λ, fast = fast)
@test val == [Inf] && ip == [1] && id == [2]

@test pmrank(λ) == 1 && pmrank(λ,fastrank=false) == 1

@test ispmregular(λ) && ispmregular(λ,fastrank=false)

@test !ispmunimodular(λ)

# P = [λ 1] 
λ = Polynomial([0,1],:λ)
P = [λ one(λ)]

@time @time info, iz, ip = pmkstruct(P, fast = fast)
@test (info.rki, info.lki,info.id, info.nf, info.nrank) == ([1], [], [], 0, 1) && iz == [] && ip == [1]

@time @time info, iz, ip = pmkstruct(P,CF1=true, fast = fast)
@test (info.rki, info.lki,info.id, info.nf, info.nrank) == ([1], [], [], 0, 1) && iz == [] && ip == [1]

@time @time info, iz, ip = pmkstruct(P,grade=3, fast = fast)
@test (info.rki, info.lki,info.id, info.nf, info.nrank) == ([1], [], [2], 0, 1) && iz == [] && ip == [1]

@time @time info, iz, ip = pmkstruct(P,grade=3,CF1=true, fast = fast)
@test (info.rki, info.lki,info.id, info.nf, info.nrank) == ([1], [], [2], 0, 1) && iz == [] && ip == [1]

@time val, info = pmeigvals(P, fast = fast)
@test val == Float64[] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([1], [], [], 0, 1)

@time val, info = pmeigvals(P,grade=3, fast = fast)
@test val == [Inf, Inf] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([1], [], [2], 0, 1)

@time val, iz, info = pmzeros(P, fast = fast)
@test val == Float64[] && iz == [] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([1], [], [], 0, 1)

@time val, ip, id = pmpoles(P, fast = fast)
@test val == [Inf] && ip == [1] && id == [1, 2]

@test pmrank(P) == 1 && pmrank(P,fastrank=false) == 1

@test !ispmregular(P) && !ispmregular(P,fastrank=false)

@test !ispmunimodular(P)


# Example 3: P = [λ^2 λ; λ 1] DeTeran, Dopico, Mackey, ELA 2009
λ = Polynomial([0,1],:λ)
P = [λ^2 λ; λ 1]

@time info, iz, ip = pmkstruct(P, fast = fast)
@test (info.rki, info.lki,info.id, info.nf, info.nrank) == ([1], [1], [], 0, 1) && iz == [] && ip == [2]

@time info, iz, ip = pmkstruct(P,CF1=true, fast = fast)
@test (info.rki, info.lki,info.id, info.nf, info.nrank) == ([1], [1], [], 0, 1) && iz == [] && ip == [2]

@time info, iz, ip = pmkstruct(P,grade=3, fast = fast)
@test (info.rki, info.lki,info.id, info.nf, info.nrank) == ([1], [1], [1], 0, 1) && iz == [] && ip == [2]

@time info, iz, ip = pmkstruct(P,grade=3,CF1=true, fast = fast)
@test (info.rki, info.lki,info.id, info.nf, info.nrank) == ([1], [1], [1], 0, 1) && iz == [] && ip == [2]

@time val, info = pmeigvals(P, fast = fast)
@test val == Float64[] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([1], [1], [], 0, 1)

@time val, info = pmeigvals(P,grade=3, fast = fast)
@test val == [Inf] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([1], [1], [1], 0, 1)

@time val, iz, info = pmzeros(P, fast = fast)
@test val == Float64[] && iz == [] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([1], [1], [], 0, 1)

@time val, ip, id = pmpoles(P, fast = fast)
@test val == [Inf, Inf] && ip == [2] && id == [2, 2, 4]

@test pmrank(P) == 1 && pmrank(P,fastrank=false) == 1

@test !ispmregular(P) && !ispmregular(P,fastrank=false)

@test !ispmunimodular(P)

# Example: DeTeran, Dopico, Mackey, ELA 2009
P = zeros(2,2,3);
P[:,:,1] = [0 0; 0 1.];
P[:,:,2] = [0 1.; 1. 0];
P[:,:,3] = [1. 0; 0 0]; 
@time val, info = pmeigvals(P)
@test val == Float64[] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([1], [1], [], 0, 1)

# Example 4: Van Dooren, Dewilde, LAA, 1983 
P = zeros(Int,3,3,3)
P[:,:,1] = [1 2 -2; 0 -1 -2; 0 0 0]
P[:,:,2] = [1 3 0; 1 4 2; 0 -1 -2]
P[:,:,3] = [1 4 2; 0 0 0; 1 4 2]

@time val, info = pmeigvals(P, fast = fast)
@test val ≈ [1, Inf, Inf] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([0], [1], [2], 1, 2)

@time val, info = pmeigvals(P,grade = 3, fast = fast)
@test val ≈ [1, Inf, Inf, Inf, Inf] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([0], [1], [1, 3], 1, 2)

@time val, iz, info = pmzeros(P, fast = fast)
@test val ≈ [1] && iz == [] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([0], [1], [2], 1, 2)

@time val, ip, id = pmpoles(P, fast = fast)
@test val ≈ [Inf, Inf] && ip == [2] && id == [2, 2, 2, 2, 4]

val, iz, info  = pmzeros1(P, fast = fast)
@test val ≈ [1] && iz == [] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([0], [1], [1], 1, 2) 

@time val, ip, id = pmpoles1(P, fast = fast)
@test val ≈ [Inf, Inf] && ip == [2] && id == [1, 1, 1, 1, 3]

@time val, iz, info = pmzeros2(P, fast = fast)
@test val ≈ [1] && iz == [] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([0], [1], [1, 1, 1, 1], 1, 2) 

@time val, ip, id = pmpoles2(P, fast = fast)
@test val ≈ [Inf, Inf] && ip == [2] && id == [1, 3]

@test !ispmregular(P) && !ispmregular(P,fastrank=false)

@test !ispmunimodular(P)

λ = Polynomial([0,1],:λ);
P = [λ^2 + λ + 1 4λ^2 + 3λ + 2 2λ^2 - 2;
λ 4λ - 1 2λ - 2;
λ^2 4λ^2 - λ 2λ^2 - 2λ]

@time info, iz, ip = pmkstruct(P, fast = fast)
@test (info.rki, info.lki,info.id, info.nf, info.nrank) == ([0], [1], [2], 1, 2)  && iz == [] && ip == [2]

@time info, iz, ip = pmkstruct(P,CF1=true, fast = fast)
@test (info.rki, info.lki,info.id, info.nf, info.nrank) == ([0], [1], [2], 1, 2)  && iz == [] && ip == [2]

@time info, iz, ip = pmkstruct(P,grade=3, fast = fast)
@test (info.rki, info.lki,info.id, info.nf, info.nrank) == ([0], [1], [1, 3], 1, 2)  && iz == [] && ip == [2]

@time info, iz, ip = pmkstruct(P,grade=3,CF1=true, fast = fast)
@test (info.rki, info.lki,info.id, info.nf, info.nrank) == ([0], [1], [1, 3], 1, 2)  && iz == [] && ip == [2]

@time val, info = pmeigvals(P, fast = fast)
@test val ≈ [1, Inf, Inf] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([0], [1], [2], 1, 2)

@time val, info = pmeigvals(P,grade = 3, fast = fast)
@test val ≈ [1, Inf, Inf, Inf, Inf] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([0], [1], [1, 3], 1, 2)

@time val, info = pmeigvals(P,CF1=false, grade = 3, fast = fast)
@test val ≈ [1, Inf, Inf, Inf, Inf] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([0], [1], [1, 3], 1, 2)

@time val, iz, info = pmzeros(P, fast = fast)
@test val ≈ [1] && iz == [] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([0], [1], [2], 1, 2)

@time val, iz, id = pmpoles(P, fast = fast)
@test val ≈ [Inf, Inf] && iz == [2] && id == [2, 2, 2, 2, 4]

@time val, iz, id = pmpoles(P,CF1=false, fast = fast)
@test val ≈ [Inf, Inf] && iz == [2] && id == [2, 2, 2, 2, 4]

@time val, iz, info = pmzeros1(P, fast = fast)
@test val ≈ [1] && iz == [] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([0], [1], [1], 1, 2) 

@time val, ip, id = pmpoles1(P, fast = fast)
@test val ≈ [Inf, Inf] && ip == [2] && id == [1, 1, 1, 1, 3]

@time val, iz, info = pmzeros2(P, fast = fast)
@test val ≈ [1] && iz == [] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([0], [1], [1, 1, 1, 1], 1, 2) 

@time val, ip, id = pmpoles2(P, fast = fast)
@test val ≈ [Inf, Inf] && ip == [2] && id == [1, 3]


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
@time val, kinfo  = pmeigvals(P, CF1 = true, fast = fast)
@test val == [Inf, Inf] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == (Int64[], Int64[], Int64[2], 0, 2)
@time val, kinfo  = pmeigvals(P, CF1 = false, fast = fast)
@test val == [Inf, Inf] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == (Int64[], Int64[], Int64[2], 0, 2)
val, iz, kinfo  = pmzeros(P, CF1 = true, fast = fast)
@test val == [Inf] && iz == [1] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == (Int64[], Int64[], Int64[2], 0, 2)
val, iz, kinfo  = pmzeros(P, CF1 = false, fast = fast)
@test val == [Inf] && iz == [1] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == (Int64[], Int64[], Int64[2], 0, 2)
val, iz, kinfo  = pmzeros2(P, fast = fast)
@test val == [Inf] && iz == [1] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == (Int64[], Int64[], Int64[1, 1, 2], 0, 2) 

N = zeros(1,1); M = ones(1,1);
P = zeros(1,1,3)
P[:,:,1] = M; P[:,:,2] = -N;
val, iz, kinfo  = pmzeros2(P, fast = fast)
@test val == [] && iz == [] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == (Int64[], Int64[], Int64[1], 0, 1) 

M = [1 0;0 1]; N = [0 1; 0 0]; 
P = zeros(2,2,3)
P[:,:,1] = M; P[:,:,2] = -N;
val, kinfo = pmeigvals(P, fast = fast)
@test val == [Inf;Inf] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == (Int64[], Int64[], [2], 0, 2)


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
      (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == ([0], [0], [1], 3, 4)



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
      (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == ([0, 0, 1, 2], [0, 3], [1, 2], 3, 12)

for Ty in (Float64, Complex{Float64}, BigFloat, Complex{BigFloat})

P = rand(Ty,2,3,5);
P1 = rand(Ty,3,2,5);
abstol = sqrt(eps(one(real(Ty))))

@time info, iz, ip = pmkstruct(P,CF1=false, fast = fast, atol = abstol)
@test (info.rki, info.lki,info.id, info.nf, info.nrank) == ([8], [], [], 0, 2) && iz == [] && ip ==[4, 4]

@time info, iz, ip = pmkstruct(P,CF1=true, fast = fast, atol = abstol)
@test (info.rki, info.lki,info.id, info.nf, info.nrank) == ([8], [], [], 0, 2) && iz == [] && ip ==[4, 4]

@time info, iz, ip = pmkstruct(P,grade=5, CF1=false, fast = fast, atol = abstol)
@test (info.rki, info.lki,info.id, info.nf, info.nrank) == ([8], [], [1, 1], 0, 2) && iz == [] && ip ==[4, 4]

@time info, iz, ip = pmkstruct(P,grade=5,CF1=true, fast = fast, atol = abstol)
@test (info.rki, info.lki,info.id, info.nf, info.nrank) == ([8], [], [1, 1], 0, 2) && iz == [] && ip ==[4, 4]

@time val, info = pmeigvals(P, fast = fast, atol = abstol)
@test val == Ty[] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([8], [], [], 0, 2)

@time val, info = pmeigvals(P,CF1=true,grade = 5, fast = fast, atol = abstol)
@test val == [Inf, Inf] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([8], [], [1, 1], 0, 2)

@time val, info = pmeigvals(P,CF1=false, grade = 5, fast = fast, atol = abstol)
@test val == [Inf, Inf] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([8], [], [1, 1], 0, 2)

@time val, iz, info = pmzeros(P, fast = fast, atol = abstol)
@test val == Ty[] && iz == [] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([8], [], [], 0, 2)

@time val, ip, id = pmpoles(P,CF1=false, fast = fast, atol = abstol)
@test val == [Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf] && ip == [4, 4] && id == [4, 8, 8]

@time val, ip, id = pmpoles(P,CF1=true, fast = fast, atol = abstol)
@test val == [Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf] && ip == [4, 4] && id == [4, 8, 8]

@time val, iz, info = pmzeros1(P, fast = fast, atol = abstol)
@test val == Ty[] && iz == [] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([8], [], [], 0, 2) 

@time val, iz, info = pmzeros1(P1, fast = fast, atol = abstol)
@test val == Ty[] && iz == [] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([], [8], [], 0, 2) 

@time val, ip, id = pmpoles1(P, fast = fast, atol = abstol)
@test val == [Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf] && ip == [4, 4] && id == [1, 5, 5]

@time val, ip, id = pmpoles1(P1, fast = fast, atol = abstol)
@test val == [Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf] && ip == [4, 4] && id == [1, 5, 5]

@time val, iz, info = pmzeros2(P, fast = fast, atol = abstol)
@test val == Ty[] && iz == [] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([8], [], [1, 1, 1, 1], 0, 2)  

@time val, ip, id = pmpoles2(P, fast = fast, atol = abstol)
@test val == [Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf] && ip == [4, 4] && id == [5, 5]


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