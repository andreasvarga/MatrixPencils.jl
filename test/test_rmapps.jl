module Test_rmapps

using Random
using LinearAlgebra
using MatrixPencils
using Polynomials
using Test

# λ = Polynomial([0,1],:λ)
# @time info, iz, nfp, ip = rmkstruct(λ, 1, fast = true)
Random.seed!(21235)

@testset "Rational Matrix Applications" begin

fast = true
for fast in (false,true)

# P = 0

@time info, iz, nfp, ip = rmkstruct(0, fast = fast) 
@test (info.rki, info.lki,info.id, info.nf, info.nrank) == ([0], [0], [], 0, 0) && iz == [] && ip == [] && nfp == 0

@time info, iz, nfp, ip = rmkstruct(0, 1, fast = fast) 
@test (info.rki, info.lki,info.id, info.nf, info.nrank) == ([0], [0], [], 0, 0) && iz == [] && ip == [] && nfp == 0

@time val, iz, info = rmzeros(0, fast = fast)
@test val ≈ Float64[] && iz == [] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([0], [0], [], 0, 0)

@time val, iz, info = rmzeros(0, 1, fast = fast)
@test val ≈ Float64[] && iz == [] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([0], [0], [], 0, 0)

@time val, ip, id = rmpoles(0, fast = fast)
@test val ≈ Float64[] && ip == [] && id == []

@time val, ip, id = rmpoles(0, 1, fast = fast)
@test val ≈ Float64[] && ip == [] && id == []

@test rmrank(0) == 0 && rmrank(0,fastrank=false) == 0

@test rmrank(0,1) == 0 && rmrank(0,1,fastrank=false) == 0


# P = 1
@time info, iz, nfp, ip = rmkstruct(1, fast = fast)
@test (info.rki, info.lki,info.id, info.nf, info.nrank) == ([], [], [], 0, 1) && iz == [] && ip == [] && nfp == 0

@time info, iz, nfp, ip = rmkstruct(1, 1, fast = fast)
@test (info.rki, info.lki,info.id, info.nf, info.nrank) == ([], [], [1], 0, 1) && iz == [] && ip == [] && nfp == 0

@time val, iz, info = rmzeros(1, fast = fast)
@test val ≈ Float64[] && iz == [] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([], [], [1], 0, 1)

@time val, iz, info = rmzeros(1, 1, fast = fast)
@test val ≈ Float64[] && iz == [] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([], [], [1], 0, 1)

@time val, ip, id = rmpoles(1, fast = fast)
@test val ≈ Float64[] && ip == [] && id == []

@time val, ip, id = rmpoles(1, 1, fast = fast)
@test val ≈ Float64[] && ip == [] && id == []

@test rmrank(1) == 1 && rmrank(1,fastrank=false) == 1

@test rmrank(1,1) == 1 && rmrank(1,1,fastrank=false) == 1


# P = λ 
λ = Polynomial([0,1],:λ)

@time info, iz, nfp, ip = rmkstruct(λ, fast = fast)
@test (info.rki, info.lki,info.id, info.nf, info.nrank) == ([], [], [], 1, 1) && iz == [] && ip == [1] && nfp == 0

@time info, iz, nfp, ip = rmkstruct(λ, 1, fast = fast)
@test (info.rki, info.lki,info.id, info.nf, info.nrank) == ([], [], [1, 1], 1, 1) && iz == [] && ip == [1] && nfp == 0

@time info, iz, nfp, ip = rmkstruct(1, λ, fast = fast)
@test (info.rki, info.lki,info.id, info.nf, info.nrank) == ([], [], [2], 0, 1) && iz == [1] && ip == [] && nfp == 1

@time val, iz, info = rmzeros(λ, fast = fast)
@test val ≈ [0.] && iz == [] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([], [], [1,1], 1, 1)

@time val, iz, info = rmzeros(λ, 1, fast = fast)
@test val ≈ [0.] && iz == [] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([], [], [1,1], 1, 1)

@time val, iz, info = rmzeros(1, λ, fast = fast)
@test val == [Inf] && iz == [1] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([], [], [2], 0, 1)

@time val, ip, id = rmpoles(λ, fast = fast)
@test val == [Inf] && ip == [1] && id == [2]

@time val, ip, id = rmpoles(λ, 1, fast = fast)
@test val == [Inf] && ip == [1] && id == [2]

@test rmrank(λ) == 1 && rmrank(λ,fastrank=false) == 1

@test rmrank(λ,1) == 1 && rmrank(λ,1,fastrank=false) == 1

# P = [λ 1] 
λ = Polynomial([0,1],:λ)
N = [λ one(λ)]
D = [one(λ) one(λ)]


@time  info, iz, nfp, ip = rmkstruct(N, fast = fast)
@test (info.rki, info.lki,info.id, info.nf, info.nrank) == ([1], [], [], 0, 1) && iz == [] && ip == [1] && nfp == 0

@time  info, iz, nfp, ip = rmkstruct(D, fast = fast)
@test (info.rki, info.lki,info.id, info.nf, info.nrank) == ([0], [], [], 0, 1) && iz == [] && ip == [] && nfp == 0

@time  info, iz, nfp, ip = rmkstruct(N, D, fast = fast)
@test (info.rki, info.lki,info.id, info.nf, info.nrank) == ([1], [], [1, 1], 0, 1) && iz == [] && ip == [1] && nfp == 0

@time  info, iz, nfp, ip = rmkstruct(D, N, fast = fast)
@test (info.rki, info.lki,info.id, info.nf, info.nrank) == ([1], [], [1], 0, 1) && iz == [] && ip == [] && nfp == 1

@time val, iz, info = rmzeros(N, fast = fast)
@test val == Float64[] && iz == [] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([1], [], [1, 1], 0, 1)

@time val, iz, info = rmzeros(N, D, fast = fast)
@test val == Float64[] && iz == [] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([1], [], [1, 1], 0, 1)

@time val, iz, info = rmzeros(D, N, fast = fast)
@test val == Float64[] && iz == [] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([1], [], [1], 0, 1)

@time val, ip, id = rmpoles(N, fast = fast)
@test val == [Inf] && ip == [1] && id == [2]

@time val, ip, id = rmpoles(N, D, fast = fast)
@test val == [Inf] && ip == [1] && id == [2]

@time val, ip, id = rmpoles(D, N, fast = fast)
@test val ≈ [0] && ip == [] && id == []

@test rmrank(N) == 1 && rmrank(N,fastrank=false) == 1

@test rmrank(N,D) == 1 && rmrank(N,D,fastrank=false) == 1

@test rmrank(D,N) == 1 && rmrank(D,N,fastrank=false) == 1

# Example 3: P = [λ^2 λ; λ 1] DeTeran, Dopico, Mackey, ELA 2009
λ = Polynomial([0,1],:λ)
N = [λ^2 λ; λ 1]
D = [1 1; 1 1]

@time  info, iz, nfp, ip = rmkstruct(N, fast = fast)
@test (info.rki, info.lki,info.id, info.nf, info.nrank) == ([1], [1], [], 0, 1) && iz == [] && ip == [2] && nfp == 0

@time  info, iz, nfp, ip = rmkstruct(N, D, fast = fast)
@test (info.rki, info.lki,info.id, info.nf, info.nrank) == ([1], [1], [1, 1, 1], 0, 1) && iz == [] && ip == [2] && nfp == 0

@time  info, iz, nfp, ip = rmkstruct(poly2pm(N), D, fast = fast)
@test (info.rki, info.lki,info.id, info.nf, info.nrank) == ([1], [1], [1, 1, 1], 0, 1) && iz == [] && ip == [2] && nfp == 0

@time  info, iz, nfp, ip = rmkstruct(N, poly2pm(D), fast = fast)
@test (info.rki, info.lki,info.id, info.nf, info.nrank) == ([1], [1], [1, 1, 1], 0, 1) && iz == [] && ip == [2] && nfp == 0

@time  info, iz, nfp, ip = rmkstruct(D, N, fast = fast)
@test (info.rki, info.lki,info.id, info.nf, info.nrank) == ([1], [1], [1], 0, 1) && iz == [] && ip == [] && nfp == 2

@time val, iz, info = rmzeros(N, fast = fast)
@test val == Float64[] && iz == [] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([1], [1], [1, 1, 1], 0, 1)

@time val, iz, info = rmzeros(N, D, fast = fast)
@test val == Float64[] && iz == [] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([1], [1], [1, 1, 1], 0, 1)

@time val, iz, info = rmzeros1(N, D, fast = fast)
@test val == Float64[] && iz == [] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([1], [1], [], 0, 1)

@time val, iz, info = rmzeros(N, poly2pm(D), fast = fast)
@test val == Float64[] && iz == [] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([1], [1], [1, 1, 1], 0, 1)

@time val, iz, info = rmzeros(D, N, fast = fast)
@test val == Float64[] && iz == [] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([1], [1], [1], 0, 1)

@time val, ip, id = rmpoles(N, fast = fast)
@test val == [Inf, Inf] && ip == [2] && id ==  [1, 3]

@time val, ip, id = rmpoles(N, D, fast = fast)
@test val == [Inf, Inf] && ip == [2] && id ==  [1, 3]

@time val, ip, id = rmpoles(N, poly2pm(D), fast = fast)
@test val == [Inf, Inf] && ip == [2] && id ==  [1, 3]

@time val, ip, id = rmpoles(D, N, fast = fast)
@test val ≈ [0, 0] && ip == [] && id ==  []

@test rmrank(N) == 1 && rmrank(N, D, fastrank=false) == 1 


# Example 4: Van Dooren, Dewilde, LAA, 1983 
N = zeros(Int,3,3,3);
N[:,:,1] = [1 2 -2; 0 -1 -2; 0 0 0];
N[:,:,2] = [1 3 0; 1 4 2; 0 -1 -2];
N[:,:,3] = [1 4 2; 0 0 0; 1 4 2];
D = ones(3,3);

@time val, iz, info = rmzeros(N, fast = fast)
@test val ≈ [1] && iz == [] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([0], [1], [1, 1, 1, 1], 1, 2)

@time val, iz, info = rmzeros(N, D, fast = fast)
@test val ≈ [1] && iz == [] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([0], [1], [1, 1, 1, 1], 1, 2)

@time val, ip, id = rmpoles(N, fast = fast)
@test val ≈ [Inf, Inf] && ip == [2] && id == [1, 3]

@time val, ip, id = rmpoles(N, D, fast = fast)
@test val ≈ [Inf, Inf] && ip == [2] && id == [1, 3]

@time val, iz, info = rmzeros1(N, D, fast = fast)
@test val ≈ [1] && iz == [] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([0], [1], [1], 1, 2)

@time val, iz, info = rmzeros1(N, poly2pm(D), fast = fast)
@test val ≈ [1] && iz == [] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([0], [1], [1], 1, 2)

@time val, iz, info = rmzeros1(pm2poly(N), reshape(D,3,3,1), fast = fast)
@test val ≈ [1] && iz == [] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([0], [1], [1], 1, 2)

@time val, iz, info  = rmzeros1(N, fast = fast)
@test val ≈ [1] && iz == [] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([0], [1], [1], 1, 2) 

@time val, ip, id = rmpoles1(N, fast = fast)
@test val ≈ [Inf, Inf] && ip == [2] && id == [1, 1, 1, 1, 3]

@time val, ip, id = rmpoles1(N, D, fast = fast)
@test val ≈ [Inf, Inf] && ip == [2] && id == [1, 1, 1, 1, 3]

@test rmrank(N) == 2 && rmrank(N, D, fastrank=false) == 2 




λ = Polynomial([0,1],:λ);
N = [λ^2 + λ + 1 4λ^2 + 3λ + 2 2λ^2 - 2;
λ 4λ - 1 2λ - 2;
λ^2 4λ^2 - λ 2λ^2 - 2λ];
D = ones(3,3);

@time  info, iz, nfp, ip = rmkstruct(N, fast = fast)
@test (info.rki, info.lki,info.id, info.nf, info.nrank) == ([0], [1], [2], 1, 2)  && iz == [] && ip == [2] && nfp == 0

@time  info, iz, nfp, ip = rmkstruct(N, D, fast = fast)
@test (info.rki, info.lki,info.id, info.nf, info.nrank) == ([0], [1], [1, 1, 1, 1], 1, 2)  && iz == [] && ip == [2] && nfp == 0

@time info, iz, nfp, ip = rmkstruct(D, N, fast = fast, atol = 1.e-7)
@test (info.rki, info.lki,info.id, info.nf, info.nrank) == ([3], [1], [2, 5], 0, 2)  && iz == [1, 4] && ip == [] && nfp == 9

@time val, iz, info = rmzeros(N, fast = fast, atol = 1.e-7)
@test val ≈ [1] && iz == [] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([0], [1], [1, 1, 1, 1], 1, 2)

@time val, iz, info = rmzeros(N, D, fast = fast)
@test val ≈ [1] && iz == [] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([0], [1], [1, 1, 1, 1], 1, 2)

@time val, iz, id = rmpoles(N, fast = fast)
@test val ≈ [Inf, Inf] && iz == [2] && id == [1, 3]

@time val, ip, id = rmpoles(N, D, fast = fast)
@test val ≈ [Inf, Inf] && ip == [2] && id == [1, 3]

@time val, iz, info = rmzeros1(N, fast = fast)
@test val ≈ [1] && iz == [] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([0], [1], [1], 1, 2) 

@time val, ip, id = rmpoles1(N, fast = fast)
@test val ≈ [Inf, Inf] && ip == [2] && id == [1, 1, 1, 1, 3]


@test rmrank(N) == 2 && rmrank(N,D,fastrank=false) == 2

@test rmrank(rand(2,3,5)) == 2 && rmrank(rand(2,3,5),rand(2,3,5),fastrank=false) == 2

@test rmrank(rand(3,2,5)) == 2 && rmrank(rand(3,2,5),rand(3,2,5),fastrank=false) == 2


# Example Vlad Ionescu 
s = Polynomial([0, 1],:s);
N = [s^2+3*s+3 1; -1 2*s^2+7*s+4];
D = [(s+1)^2 s+2; (s+1)^3 (s+1)*(s+2)]
@time info, iz, nfp, ip = rmkstruct(N, D, fast = fast, atol=1.e-7)
@test (info.rki, info.lki,info.id, info.nf, info.nrank) == (Int64[], Int64[], [1, 1], 4, 2)  && iz == [] && ip == [] && nfp == 4

@time val, ip, id = rmpoles(N, D, fast = fast, atol=1.e-7)
@test fromroots(val) ≈ Polynomial([2, 7, 9, 5, 1]) && ip == [] && id == []

@time val, ip, id = rmpoles1(N, D, fast = fast, atol=1.e-7)
@test fromroots(val) ≈ Polynomial([2, 7, 9, 5, 1]) && ip == [] && id == [1, 1, 1, 1]

@time val, ip, id = rmpoles1(poly2pm(N), D, fast = fast, atol=1.e-7)
@test fromroots(val) ≈ Polynomial([2, 7, 9, 5, 1]) && ip == [] && id == [1, 1, 1, 1]

@time val, ip, id = rmpoles1(N, poly2pm(D), fast = fast, atol=1.e-7)
@test fromroots(val) ≈ Polynomial([2, 7, 9, 5, 1]) && ip == [] && id == [1, 1, 1, 1]


# Example 1 (Polynomial): Dopico et al. 2020
c1 = [0.06155268911372547, 0.006545378521362721, 0.40445990039119284, 0.8829892254580274, 0.7573496766341161, 0.4094804382958148]; 
c2 = [0.4059208742974014, 0.48094667571705574]; 
e1 = Polynomial(c1)
e2 = Polynomial(c2)
P = [e1 0; 0 e2]


@time zer, iz, info = rmzeros(P,fast = fast, atol=1.e-9)
@test fromroots(zer) ≈ e1*e2/(e1[end]*e2[end]) && iz == [] && (info.rki, info.lki,info.id, info.nf, info.nrank) == ([], [], [1, 1, 1, 1], 6, 2)

# Example 2 (Rational): Dopico et al. 2020
c1 = [0.06155268911372547, 0.006545378521362721, 0.40445990039119284, 0.8829892254580274, 0.7573496766341161, 0.4094804382958148]; 
c2 = [0.4059208742974014, 0.48094667571705574]; 
e1 = Polynomial(c1)
e2 = Polynomial(c2)
N = [e1 0; 1 e2]
D = [1 1; Polynomial([0, 1]) 1]

@time zer, iz, info = rmzeros(N, D, fast = fast, atol=1.e-9)
@test fromroots(zer) ≈ e1*e2/(e1[end]*e2[end])*Polynomial([0,1]) && iz == [] && (info.rki, info.lki,info.id, info.nf, info.nrank) == ([], [], [1, 1, 1, 1], 7, 2)

@time pol, ip, id = rmpoles(N, D, fast = fast, atol=1.e-7)
@test pol ≈ [0, Inf, Inf, Inf, Inf, Inf, Inf] && ip == [1, 5] && id == [2, 6]


# Example 4.26 (Rational): Dopico et al. 2020 (submitted)
s = Polynomial([0,1]);
N = [s^2+s-1 -1; -1 -s^2+s-2];
D = [s s; 1 1];
@time zer, iz, info = rmzeros(N, D, fast = fast, atol=1.e-7)
@test fromroots(zer) ≈ Polynomial([-1, 3, 0, 0, 1]) && iz == [] && (info.rki, info.lki,info.id, info.nf, info.nrank) == ([], [], [1, 1, 1, 1], 4, 2)

@time pol, ip, id = rmpoles(N, D, fast = fast, atol=1.e-7)
@test pol ≈ [0, Inf, Inf, Inf] && ip == [1, 2] && id == [2, 3]


Ty = Float64
for Ty in (Float64, Complex{Float64})

N = rand(Ty,2,3,5);
N1 = rand(Ty,3,2,5);
D = ones(2,3);
D1 = ones(3,2);

abstol = sqrt(eps(one(real(Ty))))

@time info, iz, nfp, ip = rmkstruct(N, fast = fast)
@test (info.rki, info.lki,info.id, info.nf, info.nrank) == ([8], [], [], 0, 2) && iz == [] && ip ==[4, 4] && nfp == 0

@time info, iz, nfp, ip = rmkstruct(N1, fast = fast)
@test (info.rki, info.lki,info.id, info.nf, info.nrank) == ([], [8], [], 0, 2) && iz == [] && ip ==[4, 4] && nfp == 0

@time info, iz, nfp, ip = rmkstruct(N, N, fast = fast, atol = abstol)
@test (info.rki, info.lki,info.id, info.nf, info.nrank) == ([0, 0], [0], [1], 0, 1) && iz == [] && ip ==[] && nfp == 0

@time val, iz, info = rmzeros(N, fast = fast)
@test val == Ty[] && iz == [] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([8], [], [1, 1, 1, 1], 0, 2)

@time val, ip, id = rmpoles(N, fast = fast)
@test val == [Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf] && ip == [4, 4] && id == [5, 5]

@time val, iz, info = rmzeros1(N, fast = fast)
@test val == Ty[] && iz == [] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([8], [], [], 0, 2) 

@time val, iz, info = rmzeros1(N1, fast = fast)
@test val == Ty[] && iz == [] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([], [8], [], 0, 2) 

@time val, ip, id = rmpoles1(N, fast = fast, atol = abstol)
@test val == [Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf] && ip == [4, 4] && id == [1, 5, 5]

@time val, ip, id = rmpoles1(N1, fast = fast, atol = abstol)
@test val == [Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf] && ip == [4, 4] && id == [1, 5, 5]

@test rmrank(N) == 2 && rmrank(N,fastrank=false) == 2
@test rmrank(N,D) == 2 && rmrank(N,D,fastrank=false) == 2

@test rmrank(N1) == 2 && rmrank(N1,fastrank=false) == 2
@test rmrank(N1,D1) == 2 && rmrank(N1,D1,fastrank=false) == 2


N = rand(Ty,2,2,4);
D = rand(Ty,2,2,2);

@time info, iz, nfp, ip = rmkstruct(N, D, fast = fast)
@test (info.rki, info.lki,info.id, info.nf, info.nrank) == ([], [], [1, 1, 1, 1], 8, 2) && iz == [] && ip ==[2, 2] && nfp == 4

@time val, iz, info = rmzeros(N, D, fast = fast)
@test iz == [] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([], [], [1, 1, 1, 1], 8, 2)

@time val, ip, id = rmpoles(N, D, fast = fast)
@test real(val[isinf.(val)]) == Ty[Inf, Inf, Inf, Inf] && ip == [2, 2] && id == [3, 3]

@time info, iz, nfp, ip = rmkstruct(D, N, fast = fast)
@test (info.rki, info.lki,info.id, info.nf, info.nrank) == ([], [], [3, 3], 8, 2) && iz == [2, 2] && ip ==[] && nfp == 12

@time val, iz, info = rmzeros(D, N, fast = fast)
@test real(val[isinf.(val)]) == Ty[Inf, Inf, Inf, Inf] && iz == [2, 2] && (info.rki, info.lki,info.id, info.nf,info.nrank) == ([], [], [3, 3], 8, 2)

@time val, ip, id = rmpoles(D, N, fast = fast)
@test length(val) == 12 && ip == [] && id == []

end
end
#end

end  # applications testset

end  # module