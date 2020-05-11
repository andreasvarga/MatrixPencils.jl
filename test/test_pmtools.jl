module Test_pmtools

using LinearAlgebra
using MatrixPencils
using Polynomials
using Test


@testset "Polynomial Matrix Tools" begin

@testset "Conversions" begin

p = pm2poly(rand(2,3,3),:s);
@test p == pm2poly(poly2pm(p,grade=5),:s)

s = Polynomial([0, 1],:s)
v = [1+2*s;1+2*s+3*s^2]
@test v == pm2poly(poly2pm(v),:s)[:]

@test 2 == pmdeg(v)

@test [1;1] == pmeval(v,0)[:]

M = diagm(v).*one(s)   # without this trick the test fails
@test M == pm2poly(poly2pm(M),:s)

s = Polynomial([0, 1],:s);
p = 1+2*s+3*s^2
@test p == pm2poly(poly2pm(p),:s)[1,1]

@test ([-2 -1; 1 0], [3 0; 0 1]) == pm2lpCF1(p)
@test ([-2 1; -1 0], [3 0; 0 1]) == pm2lpCF2(p)

@test reverse(p.coeffs) == pmreverse(p)[:]

@test 3im == pm2poly(poly2pm(3im))[1,1]


a = rand(2,3); 
@test a == reshape(poly2pm(a),2,3)

a = rand(3); 
@test a == reshape(poly2pm(a),3)

a = transpose(rand(3)); 
@test a == reshape(poly2pm(a),1,3)

a = rand(Complex{Float64},3);
@test a' == reshape(poly2pm(a'),1,3)


end # conversions


@testset "Basic polynomial manipulations" begin

P = rand(0,0,4);
@test pmeval(P,2) ≈ lseval(pm2ls(P)[1:5]...,2)

P = rand(0,0,4);
@test pmeval(P,2) ≈ lpseval(pm2lps(P)[1:8]...,2)

P = rand(2,3,4);
@test pmeval(P,2) ≈ lseval(pm2ls(P)[1:5]...,2)

P = rand(2,3,4);
@test pmeval(P,2) ≈ lpseval(pm2lps(P)[1:8]...,2)

P = rand(Complex{Float64},2,3,4);
@test pmeval(P,2im) ≈ lseval(pm2ls(P)[1:5]...,2im)

P = rand(Complex{Float64},2,3,4);
@test pmeval(P,2im) ≈ lpseval(pm2lps(P)[1:8]...,2im)

P = rand(0,0,4);
@test P[:,:,1:pmdeg(P)+1] ≈ ls2pm(pm2ls(P)[1:5]...)

P = rand(0,0,4);
@test P[:,:,1:pmdeg(P)+1] ≈ lps2pm(pm2lps(P)[1:8]...)

P = rand(2,3,4);
@test P ≈ ls2pm(pm2ls(P,contr = true)[1:5]...)

P = rand(2,3,4);
@test P ≈ ls2pm(pm2ls(P,obs = true)[1:5]...)

P = rand(2,3,4);
@test P ≈ ls2pm(pm2ls(P,minimal = true)[1:5]...)

P = rand(2,3,4);
@test P ≈ lps2pm(pm2lps(P)[1:8]...)

P = rand(Complex{Float64},2,3,4);
@test P ≈ ls2pm(pm2ls(P)[1:5]...)

P = rand(Complex{Float64},2,3,4);
@test P ≈ lps2pm(pm2lps(P)[1:8]...)

P = rand(2,3,4);
@test P ≈ pmreverse(pmreverse(P,6))[:,:,1:4]

P = rand(0,0,4);
@test P[:,:,1:pmdeg(P)+1] ≈ pmreverse(pmreverse(P,2))[:,:,1:pmdeg(P)+1]


end # Basic manipulations

@testset "Structured polynomial matrix linearization tools" begin

A,E,B,C,D = spm2ls(1,1,1,1,atol = 1.e-7, minimal=true)  
@test iszero(D)

A,E,B,C,D = spm2ls(1,1,1,1)  
@test iszero(-C*inv(A)*B+D) && iszero(E)

A,E,B,F,C,G,D,H = spm2lps(1,1,1,1,atol = 1.e-7, minimal=true)  
@test iszero(D) && iszero(H)

A,E,B,F,C,G,D,H = spm2lps(1,1,1,1)  
@test iszero(-C*inv(A)*B+D) && iszero(E) && iszero(F) && iszero(G) && iszero(H)


#  simple test
D = rand(3,3);
W = zeros(3,3);   
A2,E2,B2,C2,D2 = spm2ls(D,D,D,W,atol = 1.e-7, minimal=true)  
@test D2 ≈ -D  

A2,E2,B2,C2,D2 = spm2ls(D,D,D,W,atol = 1.e-7)  
@test -C2*inv(A2)*B2+D2 ≈ -D && iszero(E2)

#  simple test
D = rand(3,3);
W = zeros(3,3);   
sys = spm2lps(D,D,D,W,atol = 1.e-7, minimal=true)  
@test sys[7] ≈ -D  && sys[8] ≈ 0*D

# 
D = rand(3,3,2); V = zeros(3,3,1); V[:,:,1] = Matrix{eltype(V)}(I,3,3); W = zeros(3,3);
A2,E2,B2,C2,D2 = spm2ls(D,D,V,W,atol = 1.e-7, minimal=true)   
@test D2 ≈ Matrix{eltype(V)}(I,3,3)

D = rand(3,3,2); V = zeros(3,3,1); V[:,:,1] = Matrix{eltype(V)}(I,3,3); W = zeros(3,3);
sys = spm2lps(D,D,V,W,atol = 1.e-7, minimal=true)   
@test sys[7] ≈ Matrix{eltype(V)}(I,3,3) && sys[8] ≈ 0*W

D = rand(3,3,4); W = zeros(3,3);
sys2 = spm2ls(D,D,D,W,atol = 1.e-7, minimal=true)  
sys1 = pm2ls(D) 
@test lsequal(sys1[1:5]..., sys2...,atol1 = 1.e-7,atol2 = 1.e-7) 

D = rand(3,3,4); W = zeros(3,3);
sys2 = spm2lps(D,D,D,W,atol = 1.e-7, minimal=true)  
sys1 = pm2lps(D) 
@test lpsequal(sys1..., sys2...,atol1 = 1.e-7,atol2 = 1.e-7) 

T = rand(3,3,4); U = rand(3,3,2); V = rand(3,3,4); W = rand(3,3,3);
sys = spm2ls(T,U,V,W);
@test lseval(sys...,1) ≈ pmeval(V,1)*(pmeval(T,1)\pmeval(U,1))+pmeval(W,1)

T = rand(3,3,4); U = rand(3,3,2); V = rand(3,3,4); W = rand(3,3,3);
sys = spm2lps(T,U,V,W);
@test lpseval(sys...,1) ≈ pmeval(V,1)*(pmeval(T,1)\pmeval(U,1))+pmeval(W,1)

T = rand(Complex{Float64},3,3,4); U = rand(3,3,2); V = rand(3,3,4); W = rand(3,3,3);
sys = spm2ls(T,U,V,W);
@test lseval(sys...,1) ≈ pmeval(V,1)*(pmeval(T,1)\pmeval(U,1))+pmeval(W,1)

T = rand(Complex{Float64},3,3,4); U = rand(3,3,2); V = rand(3,3,4); W = rand(3,3,3);
sys = spm2lps(T,U,V,W);
@test lpseval(sys...,1) ≈ pmeval(V,1)*(pmeval(T,1)\pmeval(U,1))+pmeval(W,1)

T = rand(0,0,4); U = rand(0,3,2); V = rand(3,0,4); W = rand(3,3,3);
sys = spm2ls(T,U,V,W);
@test lseval(sys...,1) ≈ pmeval(V,1)*(pmeval(T,1)\pmeval(U,1))+pmeval(W,1)

T = rand(0,0,4); U = rand(0,3,2); V = rand(3,0,4); W = rand(3,3,3);
sys = spm2lps(T,U,V,W);
@test lpseval(sys...,1) ≈ pmeval(V,1)*(pmeval(T,1)\pmeval(U,1))+pmeval(W,1)



#  simple transfer function realization
D = reshape([-2,-1,2,1],1,1,4);
N = reshape([-1,1,0,1],1,1,4);
V = reshape([1],1,1);
W = reshape([0.],1,1);
sys = spm2ls(D,N,V,W,atol = 1.e-7, minimal=true); 
sys_poles = eigvals(sys[1],sys[2])
@test sort(sys_poles) ≈ [-2., -1., 1.]
sys_zeros = spzeros(sys...)[1] 
@test coeffs(fromroots(sys_zeros)) ≈ [-1,1,0,1]

D = reshape([-2,-1,2,1],1,1,4);
N = reshape([-1,1,0,1],1,1,4);
V = reshape([1],1,1);
W = reshape([0.],1,1);
sys = spm2lps(D,N,V,W,atol = 1.e-7, minimal=true); 
sys_poles = eigvals(sys[1],sys[2])
@test sort(sys_poles) ≈ [-2., -1., 1.]
sys_zeros = pzeros([sys[1] sys[3]; sys[5] sys[7]],[sys[2] sys[4]; sys[6] sys[8]])[1] 
@test coeffs(fromroots(sys_zeros)) ≈ [-1,1,0,1]

D1 = Polynomial([-2,-1,2,1]);
N1 = Polynomial([-1,1,0,1]);
V1 = 1;
W1 = 0.;
sys = spm2ls(D1,N1,V1,W1,atol = 1.e-7, minimal=true); 
sys_poles = eigvals(sys[1],sys[2])
@test sort(sys_poles) ≈ [-2., -1., 1.]
sys_zeros = spzeros(sys...)[1] 
@test coeffs(fromroots(sys_zeros)) ≈ [-1,1,0,1]

end # structured linearization


@testset "Linearization tools" begin

# Various simple examples 

# P = empty

P = rand(0,0,3);
sys = pm2ls(P);
@test P[:,:,1:pmdeg(P)+1] ≈ ls2pm(sys...)

P = rand(0,0,3);
sys = pm2lps(P);
@test P[:,:,1:pmdeg(P)+1] ≈ lps2pm(sys...)

# P = 0

sys = pm2ls(0)
info = spkstruct(sys...)
@test (info.rki, info.lki,info.id, info.nf) == ([0], [0], [], 0)

sys=pm2lps(0)
info = pkstruct(sys[7],sys[8])
@test (info.rki, info.lki,info.id, info.nf) == ([0], [0], [], 0)

# P = 1

sys = pm2ls(1)
info = spkstruct(sys...)
@test (info.rki, info.lki,info.id, info.nf) == (Int64[], Int64[], [1], 0)

sys=pm2lps(1)
info = pkstruct(sys[7],sys[8])
@test (info.rki, info.lki,info.id, info.nf) == ([], [], [1], 0)

# P = λ 

λ = Polynomial([0,1],:λ)

sys = pm2ls(λ)
info = spkstruct(sys...)
@test (info.rki, info.lki,info.id, info.nf) == (Int64[], Int64[], [1, 1], 1)

sys=pm2lps(λ)
info = pkstruct(sys[7],sys[8])
@test (info.rki, info.lki,info.id, info.nf) == ([], [], [], 1)

# P = [λ 1] 
λ = Polynomial([0,1],:λ)
P = [λ one(λ)]

sys = pm2ls(P)
info = spkstruct(sys...)
@test (info.rki, info.lki,info.id, info.nf) == ([1], [], [1, 1], 0)

sys=pm2lps(P)
info = pkstruct(sys[7],sys[8])
@test (info.rki, info.lki,info.id, info.nf) == ([1], [], [], 0)

# Example 3: P = [λ^2 λ; λ 1] DeTeran, Dopico, Mackey, ELA 2009
λ = Polynomial([0,1],:λ)
P = [λ^2 λ; λ 1]
@test all(P.*one(λ) .≈ pm2poly(ls2pm(pm2ls(P)...),:λ))    # use the one(λ) trick 
@test all(P.*one(λ) .≈ pm2poly(lps2pm(pm2lps(P)...),:λ))  # use the one(λ) trick 

A, E, B, C, D = pm2ls(P,minimal = true)
M1 = [A B; C D]
N1 = [E zeros(size(B)...); zeros(size(C)...) zeros(size(D)...)]
val1, info1 = peigvals(M1,N1)
@test val1 ≈ [Inf, Inf] && (info1.rki, info1.lki, info1.id, info1.nf) == ([1], [1], [1, 1], 0) 
val1, iz, info1 = pzeros(M1,N1)
@test val1 ≈ Float64[] && iz == [] && (info1.rki, info1.lki, info1.id, info1.nf) == ([1], [1], [1, 1], 0) 

P = zeros(2,2,3);
P[:,:,1] = [0 0; 0 1.];
P[:,:,2] = [0 1.; 1. 0];
P[:,:,3] = [1. 0; 0 0]; 
@test P ≈ ls2pm(pm2ls(P)...)
@test P ≈ lps2pm(pm2lps(P)...)

val, info = pmeigvals(P)
@test val == Float64[] && (info.rki, info.lki,info.id, info.nf) == ([1], [1], [], 0)
val, iz, info = pmzeros(P)
@test val == Float64[] && iz == [] && (info.rki, info.lki,info.id, info.nf) == ([1], [1], [], 0)


# build a strong (least order) structured linearization which preserves the finite eigenvalues, 
# infinite zeros (with multiplicities), the left and right Kronecker structure
A2,E2,B2,C2,D2 = pm2ls(P)
A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal(A,E,B,C,D)
@test lsequal(A,E,B,C,D,A1,E1,B1,C1,D1) &&
      nuc == 2 && nuo == 0 && nse == 1
@time val, iz, info = spzeros(A1,E1,B1,C1,D1) 
@test val ≈ Float64[] && iz == [] &&  (info.rki, info.lki, info.id, info.nf) == ([1], [1], [1, 1], 0)
@time val, info = speigvals(A1,E1,B1,C1,D1) 
@test val ≈ [Inf, Inf] &&  (info.rki, info.lki, info.id, info.nf) == ([1], [1], [1, 1], 0)

# build linearizations which preserve the finite eigenvalues and right structure;
# the infinite zeros are not preserved because P is singular and the left Kronecker indices are in excess with 1
#M2,N2 = pm2lp(P,left=false)
M2,N2 = pm2lpCF2(P)
M = copy(M2); N = copy(N2); 
@time val, iz, info = pzeros(M,N) 
@test val[1:info.nf] ≈ Float64[] && iz == [] &&  (info.rki, info.lki, info.nf) == ([1], [2], 0)
@time val, info = peigvals(M,N) 
@test val ≈ Float64[] &&  (info.rki, info.lki, info.nf) == ([1], [2], 0)

# build a linearization which preserves the finite eigenvalues and left structure;
# the infinite zeros are not preserved and the right Kronecker indices are in excess with 1
# M2,N2 = pm2lp(P,left=true)
M2,N2 = pm2lpCF1(P)
M = copy(M2); N = copy(N2); 
@time val, iz, info = pzeros(M,N) 
@test val[1:info.nf] ≈ Float64[] && iz == [] &&  (info.rki, info.lki, info.nf) == ([2], [1], 0)
@time val, info = peigvals(M,N) 
@test val ≈ Float64[] &&  (info.rki, info.lki, info.nf) == ([2], [1], 0)

# Strong reductions

# Example 4: Van Dooren, Dewilde, LAA, 1983 
P = zeros(Int,3,3,3)
P[:,:,1] = [1 2 -2; 0 -1 -2; 0 0 0]
P[:,:,2] = [1 3 0; 1 4 2; 0 -1 -2]
P[:,:,3] = [1 4 2; 0 0 0; 1 4 2]
@test P ≈ ls2pm(pm2ls(P)...)
@test P ≈ lps2pm(pm2lps(P)...;atol1=1.e-7,atol2=1.e-7)


# computations as in paper
M, N = pm2lpCF1(P)
kinfo = pkstruct(M,N)
@test (kinfo.rki, kinfo.lki, kinfo.id, kinfo.nf) == ([1], [1], [2], 1)

val, kinfo = peigvals(M,N)
@test val ≈ [1, Inf, Inf] && (kinfo.rki, kinfo.lki, kinfo.id, kinfo.nf) == ([1], [1], [2], 1)


# this produces the correct structure at infinity, and shifted singular structure information
A, E, B, F, C, G, D, H = lpsminreal(pm2lps(P)...,contr=false);
M1 = [A B; C D];
N1 = [E F; G H];
val1, info1 = peigvals(M1,N1)
@test val1 ≈ [1, Inf, Inf] && (info1.rki, info1.lki,info1.id, info1.nf) == ([0], [2], [2], 1) 

# this produces the wrong structure at infinity, and shifted singular structure information
A, E, B, F, C, G, D, H = lpsminreal(pm2lps(P)...,obs=false)
M1 = [A B; C D];
N1 = [E F; G H];
val1, info1 = peigvals(M1,N1)
@test val1 ≈ [1, Inf] && (info1.rki, info1.lki,info1.id, info1.nf) == ([0], [1], [1], 1) 



# build controllable and observable pencil realizations
sys1  = pm2lps(P,contr=true)
sys2  = pm2lps(P,obs=true)
@test lpsequal(sys1..., sys2...)
@test lpseval(sys1...,2) ≈ lpseval(sys2...,2)

# 
@test lps2pm(sys1...,atol1=1.e-7,atol2=1.e-7,val=2) ≈ lps2pm(sys2...,atol1=1.e-7,atol2=1.e-7)
@test ls2pm(pm2ls(P,contr=true)...,atol1=1.e-7,atol2=1.e-7,val=2) ≈ P[:,:,1:3]

(A, E, B, F, C, G, D, H) = copy.(sys2)
A1, E1, B1, F1, C1, G1, D1, H1, V, W, nuc,nuo = lpsminreal(A, E, B, F, C, G, D, H,atol1=1.e-7,atol2=1.e-7) 
@test lpsequal(A1, E1, B1/W, F1/W, V'\C1, V'\G1, V'\D1/W, V'\H1/W, sys2...,atol1=1.e-7,atol2=1.e-7)

# compute zeros structure
M = [A1 B1; C1 D1]
N = [E1 F1; G1 H1]
val, info = peigvals(M,N)
@test val ≈ [1, Inf] && (info.rki, info.lki,info.id, info.nf) == ([0], [1], [1], 1)

# compute poles structure
p, m = size(D1)
n = size(B1,1)
M = [A1 B1 zeros(n,p); C1 D1 -V'; zeros(m,n) W zeros(m,p)]
N = [E1 F1 zeros(n,p); G1 H1 zeros(p,p); zeros(m,n+m+p)]
val, info = peigvals(M,N)
@test val ≈ [Inf, Inf, Inf, Inf, Inf, Inf, Inf] && (info.rki, info.lki,info.id, info.nf) == ([], [], [1, 1, 1, 1, 3], 0)
val, iz, info = pzeros(M,N)
@test val ≈ [Inf, Inf] && (info.rki, info.lki,info.id, info.nf) == ([], [], [1, 1, 1, 1, 3], 0)

(A, E, B, F, C, G, D, H) = copy.(sys1)
A1, E1, B1, F1, C1, G1, D1, H1, V, W, nuc,nuo = lpsminreal(A, E, B, F, C, G, D, H,atol1=1.e-7,atol2=1.e-7,contr=false) 
@test lpsequal(A1, E1, B1/W, F1/W, V'\C1, V'\G1, V'\D1/W, V'\H1/W, sys1...,atol1=1.e-7,atol2=1.e-7)

M = [A1 B1; C1 D1]
N = [E1 F1; G1 H1]
val, info = peigvals(M,N)
@test val ≈ [1, Inf] && (info.rki, info.lki,info.id, info.nf) == ([0], [1], [1], 1)


# ensuring strong observability
# Example 4: Van Dooren, Dewilde, LAA 1983
P = zeros(3,3,3)
P[:,:,1] = [1 2 -2; 0 -1 -2; 0 0 0]
P[:,:,2] = [1 3 0; 1 4 2; 0 -1 -2]
P[:,:,3] = [1 4 2; 0 0 0; 1 4 2]

# build controllable and observable pencil realizations
sys1  = pm2lps(P,contr=true)
sys2  = pm2lps(P,obs=true)
@test lpsequal(sys1..., sys2...)
@test lpseval(sys1...,1) ≈ lpseval(sys2...,1)

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
    0     0     0     0     0     0     0     0     0
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

D2 = zeros(Int,3,3);

sys = (A2, E2, B2, C2, D2)
@time P = ls2pm(sys...)
@test pmeval(P,1) ≈ lseval(sys...,1)


A2 = [
    -2    -3     0     0     0     0     0     0     0
     1     0     0     0     0     0     0     0     0
     0     0    -2    -3     0     0     0     0     0
     0     0     1     0     0     0     0     0     0
     0     0     0     0     1     0     0     0     0
     0     0     0     0     0     1     0     0     0
     0     0     0     0     0     0     1     0     0
     0     0     0     0     0     0     0     1     0
     0     0     0     0     0     0     0     0     1];
 E2 = [
     1     0     0     0     0     0     0     0     0
     0     1     0     0     0     0     0     0     0
     0     0     1     0     0     0     0     0     0
     0     0     0     1     0     0     0     0     0
     0     0     0     0     0     0     0     0     0
     0     0     0     0     1     0     0     0     0
     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     1     0     0
     0     0     0     0     0     0     0     1     0];
 B2 = [
     1     0
     0     0
     0     1
     0     0
    -1     0
     0     0
     0    -1
     0     0
     0     0];
 C2 = [
     1     0     1    -3     0     1     0     2     0
     0     1     1     3     0     1     0     0     1];
D2 = zeros(Int,2,2);  

try 
   ls2pm(A2,E2,B2,C2,D2,atol1 = 1.e-7,atol2=1.e-7)
   @test false
catch
   @test true
end

# Example 3 - (Varga, Kybernetika, 1990) 
A2 = [
1 0 0 0 -1 0 0 0
0 1 0 0 0 -1 0 0
0 0 1 0 0 0 0 0      
0 0 0 1 0 0 0 0
0 0 0 0 -1 0 0 0
0 0 0 0 0 -1 0 0
0 0 0 0 3 0 1 0
0 0 0 0 0 2 0 1
]
E2 = [
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0
1 0 0 0 0 0 0 0
0 1 0 0 0 0 0 0
0 0 1 0 0 0 0 0
0 0 0 1 0 0 0 0
]
B2 = [
      -1 1
      0 0
      0 0
      0 0
      1 -2
      -2 3
      0 0
      3 -3
]
C2 = [
      0 0 0 0 0 0 -1 0
      0 0 0 0 0 0 0 -1      
]
D2 = zeros(Int,2,2);  

# case of a non-minimal realization of a polynomial matrix with finite eigenvalues 
sys = ([A2 B2; C2 D2], [E2 zeros(8,2); zeros(2,10)], [zeros(8,2);I], [zeros(2,8) -I], zeros(2,2))

@time P = ls2pm(sys...,atol1 = 1.e-7,atol2=1.e-7);
@test pmeval(P,1) ≈ lseval(sys...,1)

end # linearization tools

end # testset
end # module
