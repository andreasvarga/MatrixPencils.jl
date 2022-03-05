module Test_linsystools

using LinearAlgebra
using MatrixPencils
using Test
using Polynomials

@testset "Linear System Tools" begin

@testset "lseval, lpseval & lps2ls" begin

# g = [z]
A = [0.0 1.0; -1.0 0.0];
E = [0.0 0.0; 0.0 -1.0];
B = [-1.0; 0.0];
C = [1.0 0.0];
D = [0.0];

@time  G = lseval(A,E,B,C,D,1)
@test G ≈ ones(1,1)

@time  G = lseval(A,E,B,C,D,Inf)
@test G ≈ Inf*ones(1,1)


# g = [z 1/(z-1)]
A = [ 1     0     0; 0     1     0; 0     0     1];
E = [ 0     1     0; 0     0     0; 0     0     1];
B = [ 0     0; -1     0; 0     1]; 
C = [1     0     1];
D = [0 0]; 
@time  G = lseval(A,E,B,C,D,1)
@test G ≈ [1 Inf]

@time  G = lseval(A,E,B,C,D,Inf)
@test G ≈ [Inf 0]

@time  G = lseval(A,E,B,C,D,0)
@test G ≈ [0 -1]

# g = [1/z (z-1); z z/(1-z)]
A =  [    0     0     0     0     0     0
0     1     0     0     0     0
0     0     1     0     0     0
0     0     0     1     0     0
0     0     0     0     1     0
0     0     0     0     0     1];
E = [1     0     0     0     0     0
0     0     1     0     0     0
0     0     0     0     0     0
0     0     0     0     1     0
0     0     0     0     0     0
0     0     0     0     0     1];
B = [ 1     0
0     0
-1     0
0     0
0    -2
0     1];
C = [    1.0000         0         0    0.5000   -0.5000         0
0    1.0000         0         0         0   -1.0000];
D = [     0     0
0    -1]; 

@time  G = lseval(A,E,B,C,D,0)
@test ≈(G,[Inf -1; 0 0],atol=1.e-7)

@time  G = lseval(A,E,B,C,D,1)
@test ≈(G,[1 0; 1 Inf],atol=1.e-7)

@time  G = lseval(A,E,B,C,D,2)
@test ≈(G,[0.5   1.0; 2.0  -2.0],atol=1.e-7)

@time  G = lseval(A,E,B,C,D,Inf)
@test ≈(G,[  0.0  Inf; Inf   -1.0],atol=1.e-7)

@time  G = lseval(A,E,B,C,D,im)
@test ≈(G,[ -1.0im  -1.0+1.0im; 1.0im  -0.5+0.5im],atol=1.e-7)

@time  G = lseval(A,E,B,C,D,exp(im))
@test G ≈ [5.403023058681398e-01 - 8.414709848078965e-01im -4.596976941318602e-01 + 8.414709848078965e-01im;
           5.403023058681398e-01 + 8.414709848078965e-01im -5.000000000000000e-01 + 9.152438608562260e-01im]

A = rand(2,2); E = rand(2,2); B = rand(2,3); C = rand(1,2); D = rand(1,3);  
@time  G = lseval(A,E,B,C,D,Inf)
@test G ≈ D

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
F2 = ones(size(B2)...);
G2 = ones(size(C2)...);
H2 = ones(size(D2)...);
sys = (A2,E2,B2,F2,C2,G2,D2,H2);

sys1 = lps2ls(A2,I,B2,missing,C2,G2,D2,H2);
sys2 = lps2ls(A2,I,B2,missing,C2,G2,D2,H2,compacted = true);
@test lsequal(sys1...,sys2...,atol1=1.e-7,atol2=1.e-7)

sys1 = lps2ls(sys...);
sys2 = lps2ls(sys...,compacted = true);
@test lsequal(sys1...,sys2...,atol1=1.e-7,atol2=1.e-7)

@test lseval(sys2...,1) ≈ (C2-G2)*((E2-A2)\(B2-F2)) + D2-H2

g1 = lseval(sys1...,1)
@test g1 ≈ (C2-G2)*((E2-A2)\(B2-F2)) + D2-H2
g2 = lseval(sys2...,1)
@test g2 ≈ (C2-G2)*((E2-A2)\(B2-F2)) + D2-H2

sys = (A2,E2,B2,F2,C2,G2,D2,H2);
@time  G = lpseval(sys...,0)
@test G ≈ lseval(A2,E2,B2,C2,D2,0)
@test G ≈ lseval(lps2ls(sys...)...,0)


@time  G = lpseval(A2,E2,view(B2,:,1),view(F2,:,1),C2,G2,view(D2,:,1),view(H2,:,1),0)
@test G ≈ lseval(A2,E2,view(B2,:,1),C2,view(D2,:,1),0)

@time  G = lpseval(A2,E2,B2,F2,view(C2,1:1,:),view(G2,1:1,:),view(D2,1:1,:),view(H2,1:1,:),0)
@test G ≈ lseval(A2,E2,B2,view(C2,1:1,:),view(D2,1:1,:),0)

@time  G = lpseval(sys...,Inf)
@test G ≈ [Inf Inf; Inf Inf]

@time  G = lpseval(A2,E2,view(B2,:,1),view(F2,:,1),C2,G2,view(D2,:,1),view(H2,:,1),Inf)
@test G ≈ [Inf; Inf]


# Example Van Dooren & Dewilde, LAA 1983.
# strongly controllable realization

sys = ([1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0], 
[0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0], 
[0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0], 
[1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0], 
[1.0 3.0 0.0; 1.0 4.0 2.0; 0.0 -1.0 -2.0], 
[-1.0 -4.0 -2.0; -0.0 -0.0 -0.0; -1.0 -4.0 -2.0], 
[1.0 2.0 -2.0; 0.0 -1.0 -2.0; 0.0 0.0 0.0], 
[0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0]);
@time  G = lpseval(sys...,0)
@test G ≈ [1.0 2.0 -2.0; 0.0 -1.0 -2.0; 0.0 0.0 0.0]

@time  G = lpseval(sys...,Inf)
@test G ≈ [Inf  Inf  Inf;
Inf  Inf  Inf;
Inf  Inf  Inf]

end


@testset "lsequal and lpsequal" begin

A = zeros(0,0); E = zeros(0,0); C = zeros(2,0); B = zeros(0,3); D = zeros(2,3);
@test lsequal(A,E,B,C,D,A,E,B,C,D) 

@test !lsequal(A,E,B,C,D,A,E,C,B,D) 

A = zeros(0,0); E = zeros(0,0); C = zeros(2,0); G = zeros(2,0);  B = zeros(0,3); F = zeros(0,3); D = zeros(2,3); H = zeros(2,3);
@test lpsequal(A,E,B,F,C,G,D,H,A,E,B,F,C,G,D,H) 

@test !lpsequal(A,E,B,F,C,G,D,H,A,E,C,F,B,G,D,H) 


end


@testset "lsminreal and lsminreal2" begin

for fast in (true, false)

A2 = zeros(0,0); E2 = zeros(0,0); C2 = zeros(0,0); B2 = zeros(0,0); D2 = zeros(0,0);
sys = (A2,E2,B2,C2,D2);

sys1 = lsminreal(sys..., fast = fast)
@test lsequal(sys...,sys1[1:5]...) && sys1[6:8] == (0,0,0)

sys1 = lsminreal2(sys..., fast = fast)
@test lsequal(sys...,sys1[1:5]...) && sys1[6:8] == (0,0,0)

A2 = rand(3,3); E2 = zeros(3,3); C2 = zeros(0,3); B2 = zeros(3,0); D2 = zeros(0,0);
sys = (A2,E2,B2,C2,D2);

sys1 = lsminreal(sys..., fast = fast)
@test lsequal(sys...,sys1[1:5]...) && sys1[6:8] == (3,0,0)

sys1 = lsminreal(sys..., fast = fast, contr = false)
@test lsequal(sys...,sys1[1:5]...) && sys1[6:8] == (0,3,0)

sys1 = lsminreal2(sys..., fast = fast)
@test lsequal(sys...,sys1[1:5]...) && sys1[6:8] == (3,0,0)

sys1 = lsminreal2(sys..., fast = fast, contr = false)
@test lsequal(sys...,sys1[1:5]...) && sys1[6:8] == (0,3,0)

# B and D vectors
A2 = rand(3,3); E2 = zeros(3,3); C2 = zeros(1,3); B2 = zeros(3,1); D2 = zeros(1);
sys = (A2,E2,B2,C2,D2);

sys1 = lsminreal(sys..., fast = fast)
@test lsequal(sys...,sys1[1:5]...) && sys1[6:8] == (3,0,0)

sys1 = lsminreal(sys..., fast = fast, contr = false)
@test lsequal(sys...,sys1[1:5]...) && sys1[6:8] == (0,3,0)

sys1 = lsminreal2(sys..., fast = fast)
@test lsequal(sys...,sys1[1:5]...) && sys1[6:8] == (3,0,0)

sys1 = lsminreal2(sys..., fast = fast, contr = false)
@test lsequal(sys...,sys1[1:5]...) && sys1[6:8] == (0,3,0)

# Example 1: DeTeran, Dopico, Mackey, ELA 2009

A2 = [1.0  0.0  0.0  0.0  0.0  0.0
0.0  1.0  0.0  0.0  0.0  0.0
0.0  0.0  1.0  0.0  0.0  0.0
0.0  0.0  0.0  1.0  0.0  0.0
0.0  0.0  0.0  0.0  1.0  0.0
0.0  0.0  0.0  0.0  0.0  1.0];

E2 = [0.0  0.0  1.0  0.0  0.0  0.0
0.0  0.0  0.0  1.0  0.0  0.0
0.0  0.0  0.0  0.0  1.0  0.0
0.0  0.0  0.0  0.0  0.0  1.0
0.0  0.0  0.0  0.0  0.0  0.0
0.0  0.0  0.0  0.0  0.0  0.0];

B2 = [0.0  0.0
0.0  0.0
0.0  1.0
1.0  0.0
1.0  0.0
0.0  0.0];

C2 = [-1.0   0.0  0.0  0.0  0.0  0.0
0.0  -1.0  0.0  0.0  0.0  0.0];

D2 = [0.0  0.0
0.0  1.0];

sys = (A2,E2,B2,C2,D2);

# compute minimal realization 
sys1 = lsminreal(sys..., fast = fast)
@test lsequal(sys...,sys1[1:5]...) && sys1[6:8] == (2,0,1)  # nuc == 2 && nuo == 0 && nse == 1
# an order reduction without enforcing controllability and observability may not be possible
sys1 = lsminreal(sys...,contr=false,obs=false, fast = fast)
@test lsequal(sys...,sys1[1:5]...) && sys1[6:8] == (0,0,0)  # nuc == 0 && nuo == 0 && nse == 0 !!
# compute an irreducible realization which still contains a non-dynamic mode
sys1 = lsminreal(sys...,noseig=false, fast = fast)
@test lsequal(sys...,sys1[1:5]...) && sys1[6:8] == (2,0,0)  # nuc == 2 && nuo == 0 && nse == 0

sys = (E2,A2,B2,C2,D2); 
# compute minimal realization for a standard system (i.e., irreducible realization)
sys1 = lsminreal(sys..., fast = fast)
@test lsequal(sys...,sys1[1:5]...) && sys1[6:8] == (2,0,0)  # nuc == 2 && nuo == 0 && nse == 0
# an order reduction without enforcing controllability and observability may not be possible
sys1 = lsminreal(sys...,contr=false,obs=false, fast = fast)
@test lsequal(sys...,sys1[1:5]...) && sys1[6:8] == (0,0,0)  # nuc == 0 && nuo == 0 && nse == 0 
# compute an irreducible realization which still contains a non-dynamic mode
sys1 = lsminreal(sys...,noseig=false, fast = fast)
@test lsequal(sys...,sys1[1:5]...) && sys1[6:8] == (2,0,0)  # nuc == 2 && nuo == 0 && nse == 0


sys = (A2,E2,B2,C2,D2);
# compute minimal realization 
sys1 = lsminreal2(sys..., fast = fast)
@test lsequal(sys...,sys1[1:5]...) && sys1[6:8] == (2,0,1)  # nuc == 2 && nuo == 0 && nse == 1
# minimal realization is possible only applying the infinite controllability/observability algorithm
sys1 = lsminreal2(sys...,finite = false, fast = fast)
@test lsequal(sys...,sys1[1:5]...) && sys1[6:8] == (2,0,1)  # nuc == 2 && nuo == 0 && nse == 0
# order reduction may results even when applying the finite controllability/observability algorithm
sys1 = lsminreal2(sys...,infinite = false, fast = fast)
@test lsequal(sys...,sys1[1:5]...) && sys1[6:8] == (1,0,0)  # nuc == 1 && nuo == 0 && nse == 0
# an order reduction without enforcing controllability and observability may not be possible
sys1 = lsminreal2(sys...,contr=false,obs=false, fast = fast)
@test lsequal(sys...,sys1[1:5]...) && sys1[6:8] == (0,0,0)  # nuc == 0 && nuo == 0 && nse == 0 !!
# compute an irreducible realization which still contains a non-dynamic mode
sys1 = lsminreal2(sys...,noseig=false, fast = fast)
@test lsequal(sys...,sys1[1:5]...) && sys1[6:8] == (2,0,0)  # nuc == 2 && nuo == 0 && nse == 0



sys = (E2,A2,B2,C2,D2); 
# compute minimal realization for a standard system (i.e., irreducible realization)
sys1 = lsminreal2(sys..., fast = fast)
@test lsequal(sys...,sys1[1:5]...) && sys1[6:8] == (2,0,0)  # nuc == 2 && nuo == 0 && nse == 0
# an order reduction without enforcing controllability and observability may not be possible
sys1 = lsminreal2(sys...,contr=false,obs=false, fast = fast)
@test lsequal(sys...,sys1[1:5]...) && sys1[6:8] == (0,0,0)  # nuc == 0 && nuo == 0 && nse == 0 
# compute an irreducible realization which still contains a non-dynamic mode
sys1 = lsminreal2(sys...,noseig=false, fast = fast)
@test lsequal(sys...,sys1[1:5]...) && sys1[6:8] == (2,0,0)  # nuc == 2 && nuo == 0 && nse == 0

# Example Van Dooren & Dewilde, LAA 1983.
# P = zeros(3,3,3)
# P[:,:,1] = [1 2 -2; 0 -1 -2; 0 0 0]
# P[:,:,2] = [1 3 0; 1 4 2; 0 -1 -2]
# P[:,:,3] = [1 4 2; 0 0 0; 1 4 2]

# observable realization with A2 = I
A2 = [ 1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0
0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0
0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0
0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0
0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0
0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0
0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0];

E2 = [ 0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0
0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0
0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0
0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0
0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0
0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0
0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0];

B2 = [ 0.0   0.0   0.0
0.0   0.0   0.0
0.0   0.0   0.0
1.0   3.0   0.0
1.0   4.0   2.0
0.0  -1.0  -2.0
1.0   4.0   2.0
0.0   0.0   0.0
1.0   4.0   2.0];

C2 = [ -1.0   0.0   0.0  0.0  0.0  0.0  0.0  0.0  0.0
0.0  -1.0   0.0  0.0  0.0  0.0  0.0  0.0  0.0
0.0   0.0  -1.0  0.0  0.0  0.0  0.0  0.0  0.0];

D2 = [ 1.0   2.0  -2.0
0.0  -1.0  -2.0
0.0   0.0   0.0];

# build a strong (least order) structured linearization which preserves the finite eigenvalues, 
# infinite zeros (with multiplicities), the left and right Kronecker structure
#A2,E2,B2,C2,D2 = pol2lp(P)
A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal(A,E,B,C,D, atol1 = 1.e-7, atol2 = 1.e-7, fast = fast);
@test lsequal(A2,E2,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 5 && nuo == 0 && nse == 1
val, iz, info = spzeros(A1,E1,B1,C1,D1; atol1 = 1.e-7, atol2 = 1.e-7) 
println("val, iz, info = $((val, iz, info))")
@test val ≈ [1.0] && iz == [] &&  (info.rki, info.lki, info.id, info.nf) == ([0], [1], [1, 1, 1], 1)
val, info = speigvals(A1,E1,B1,C1,D1) 
@test val ≈ [1., Inf, Inf, Inf] &&  (info.rki, info.lki, info.id, info.nf) == ([0], [1], [1, 1, 1], 1)

A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal(A,E,B,C,D,obs=false, fast = fast, atol1 = 1.e-7, atol2 = 1.e-7);
@test lsequal(A2,E2,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 5 && nuo == 0 && nse == 1
val, iz, info = spzeros(A1,E1,B1,C1,D1; atol1 = 1.e-7, atol2 = 1.e-7) 
@test val ≈ [1.0] && iz == [] &&  (info.rki, info.lki, info.id, info.nf) == ([0], [1], [1, 1, 1], 1)
val, info = speigvals(A1,E1,B1,C1,D1) 
@test val ≈ [1., Inf, Inf, Inf] &&  (info.rki, info.lki, info.id, info.nf) == ([0], [1], [1, 1, 1], 1)


A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal2(A,E,B,C,D, fast = fast);
@test lsequal(A2,E2,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 5 && nuo == 0 && nse == 1
val, iz, info = spzeros(A1,E1,B1,C1,D1) 
@test val ≈ [1.0] && iz == [] &&  (info.rki, info.lki, info.id, info.nf) == ([0], [1], [1, 1, 1], 1)
val, info = speigvals(A1,E1,B1,C1,D1) 
@test val ≈ [1., Inf, Inf, Inf] &&  (info.rki, info.lki, info.id, info.nf) == ([0], [1], [1, 1, 1], 1)

A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal2(A,E,B,C,D,obs=false,finite=false, fast = fast);
@test lsequal(A2,E2,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 5 && nuo == 0 && nse == 1
val, iz, info = spzeros(A1,E1,B1,C1,D1) 
@test val ≈ [1.0] && iz == [] &&  (info.rki, info.lki, info.id, info.nf) == ([0], [1], [1, 1, 1], 1)
val, info = speigvals(A1,E1,B1,C1,D1) 
@test val ≈ [1., Inf, Inf, Inf] &&  (info.rki, info.lki, info.id, info.nf) == ([0], [1], [1, 1, 1], 1)

# perform lsminreal with A and E interchanged
A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
E1, A1, B1, C1, D1, nuc, nuo, nse  = lsminreal(E,A,B,C,D,obs = false, fast = fast)
@test lsequal(A2,E2,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 5 && nuo == 0 && nse == 0

# perform lsminreal2 with A and E interchanged
A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
E1, A1, B1, C1, D1, nuc, nuo, nse  = lsminreal2(E,A,B,C,D,obs = false, fast = fast)
@test lsequal(A2,E2,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 5 && nuo == 0 && nse == 0


# controllable realization with A2 = I
A2 = [ 1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0  0.0
0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0
0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0
0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0
0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0
0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0
0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0];

E2 = [ 0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0  0.0
0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0
0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0
0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0  0.0
0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0  0.0
0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  1.0
0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0];

B2 = [   0.0   0.0   0.0
0.0   0.0   0.0
0.0   0.0   0.0
0.0   0.0   0.0
0.0   0.0   0.0
0.0   0.0   0.0
-1.0   0.0   0.0
0.0  -1.0   0.0
0.0   0.0  -1.0];

C2 = [  1.0  4.0  2.0  1.0   3.0   0.0  0.0  0.0  0.0
0.0  0.0  0.0  1.0   4.0   2.0  0.0  0.0  0.0
1.0  4.0  2.0  0.0  -1.0  -2.0  0.0  0.0  0.0
];

D2 = [ 1.0   2.0  -2.0
0.0  -1.0  -2.0
0.0   0.0   0.0];


A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal(A,E,B,C,D, fast = fast)
@test lsequal(A2,E2,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 0 && nuo == 5 && nse == 1

A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal(A,E,B,C,D,contr = false, fast = fast);
@test lsequal(A2,E2,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 0 && nuo == 5 && nse == 1

A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal2(A,E,B,C,D, fast = fast)
@test lsequal(A2,E2,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 0 && nuo == 5 && nse == 1

A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal2(A,E,B,C,D,contr = false,finite=false, fast = fast);
@test lsequal(A2,E2,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 0 && nuo == 5 && nse == 1


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
    0     0     0     0     0     0     0     0     0
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

D2 = zeros(Float64,3,3);

A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = 
     lsminreal(A,E,B,C,D,atol1 = 1.e-7, atol2 = 1.e-7, fast = fast)
@test lsequal(A2,E2,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 0 && nuo == 5 && nse == 1
      
A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = 
     lsminreal(A,E,B,C,D,atol1 = 1.e-7, atol2 = 1.e-7,contr=false, fast = fast);
@test lsequal(A2,E2,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 0 && nuo == 5 && nse == 1

A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal(A,I,B,C,D,atol1 = 1.e-7, atol2 = 1.e-7, fast = fast)
@test lsequal(A2,I,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 6 && nuo == 3 && nse == 0

A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal2(A,E,B,C,D,atol1 = 1.e-7, atol2 = 1.e-7, fast = fast)
@test lsequal(A2,E2,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 0 && nuo == 5 && nse == 1

A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = 
      lsminreal2(A,E,B,C,D,contr=false,finite=false,atol1 = 1.e-7, atol2 = 1.e-7, fast = fast);
@test lsequal(A2,E2,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 0 && nuo == 5 && nse == 1

A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal2(A,I,B,C,D,atol1 = 1.e-7, atol2 = 1.e-7, fast = fast)
@test lsequal(A2,I,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 6 && nuo == 3 && nse == 0


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

A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal(A,E,B,C,D,atol1 = 1.e-7, atol2 = 1.e-7,noseig=true, fast = fast)
@test lsequal(A2,E2,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 0 && nuo == 2 && nse == 0

A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal(A,E,B,C,D,obs=false,atol1 = 1.e-7, atol2 = 1.e-7, fast = fast)
@test lsequal(A2,E2,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 0 && nuo == 0 && nse == 0

A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal(A,E,B,C,D,contr=false,atol1 = 1.e-7, atol2 = 1.e-7, fast = fast)
@test lsequal(A2,E2,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 0 && nuo == 2 && nse == 0


A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal2(A,E,B,C,D,atol1 = 1.e-7, atol2 = 1.e-7,noseig=true, fast = fast)
@test lsequal(A2,E2,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 0 && nuo == 2 && nse == 0

A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal2(A,E,B,C,D,obs=false,atol1 = 1.e-7, atol2 = 1.e-7, fast = fast)
@test lsequal(A2,E2,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 0 && nuo == 0 && nse == 0

A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal2(A,E,B,C,D,contr=false,finite=false,atol1 = 1.e-7, atol2 = 1.e-7, fast = fast)
@test lsequal(A2,E2,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 0 && nuo == 2 && nse == 0

A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal(A',E',C',B',D',atol1 = 1.e-7, atol2 = 1.e-7, fast = fast)
@test lsequal(A2,E2,B2,C2,D2,A1',E1',C1',B1',D1') &&
      nuc == 2 && nuo == 0 && nse == 0

A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal2(A',E',C',B',D',atol1 = 1.e-7, atol2 = 1.e-7, fast = fast)
@test lsequal(A2,E2,B2,C2,D2,A1',E1',C1',B1',D1') &&
      nuc == 2 && nuo == 0 && nse == 0

# Example 1 - (Varga, Kybernetika, 1990) 
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

A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal(A,E,B,C,D,atol1 = 1.e-7, atol2 = 1.e-7, fast = fast)
@test lsequal(A2,E2,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 6 && nuo == 0 && nse == 1

A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time @time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal2(A,E,B,C,D,atol1 = 1.e-7, atol2 = 1.e-7, fast = fast)
@test lsequal(A2,E2,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 6 && nuo == 0 && nse == 1


A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal(A',E',C',B',D',atol1 = 1.e-7, atol2 = 1.e-7, fast = fast)
@test lsequal(A2,E2,B2,C2,D2,A1',E1',C1',B1',D1') &&
      nuc == 2 && nuo == 4 && nse == 1 

A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal2(A',E',C',B',D',atol1 = 1.e-7, atol2 = 1.e-7, fast = fast)
@test lsequal(A2,E2,B2,C2,D2,A1',E1',C1',B1',D1') &&
      nuc == 2 && nuo == 4 && nse == 1 

A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal(A',E',C',B',D',contr=false, atol1 = 1.e-7, atol2 = 1.e-7, fast = fast)
@test lsequal(A2,E2,B2,C2,D2,A1',E1',C1',B1',D1') &&
      nuc == 0 && nuo == 6 && nse == 1 

A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal2(A',E',C',B',D',contr=false, atol1 = 1.e-7, atol2 = 1.e-7, fast = fast)
@test lsequal(A2,E2,B2,C2,D2,A1',E1',C1',B1',D1') &&
      nuc == 0 && nuo == 6 && nse == 1 

# B and D vectors
A2 = [
      0 0 0 -24 0 0 0 0 0 0 0
      1 0 0 -50 0 0 0 0 0 0 0
      0 1 0 -35 0 0 0 0 0 0 0
      0 0 1 -10 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 -30 0 0 0
      0 0 0 0 1 0 0 -61 0 0 0
      0 0 0 0 0 1 0 -41 0 0 0
      0 0 0 0 0 0 1 -11 0 0 0
      0 0 0 0 0 0 0 0 0 0 -15
      0 0 0 0 0 0 0 0 1 0 -23
      0 0 0 0 0 0 0 0 0 1 -9
]
E2 = I;
#B2 = reshape([18; 42; 30; 6; 10;17;8;1;0;-10;-2;],11,1)
B2 = [18; 42; 30; 6; 10;17;8;1;0;-10;-2;]
C2 = [0 0 0 0 0 0 0 1 0 0 0]
D2 = [0]

A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal(A,E,B,C,D,atol1 = 1.e-7, atol2 = 1.e-7, fast = fast)
@test lsequal(A2,E2,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 7 && nuo == 3 && nse == 0

@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal(A1,E1,B1,C1,D1,atol1 = 1.e-7, atol2 = 1.e-7, fast = fast)
@test lsequal(A2,E2,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 0 && nuo == 0 && nse == 0

A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal2(A,E,B,C,D,atol1 = 1.e-7, atol2 = 1.e-7, fast = fast)
@test lsequal(A2,E2,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 7 && nuo == 3 && nse == 0

@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal2(A1,E1,B1,C1,D1,atol1 = 1.e-7, atol2 = 1.e-7, fast = fast)
@test lsequal(A2,E2,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 0 && nuo == 0 && nse == 0


for Ty in (Float64, Complex{Float64})

#fast = true; Ty = Complex{Float64}    
n = 10; m = 5; p = 6;
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
D2 = zeros(Ty,p,m)


A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal(A,E,B,C,D,atol1 = 1.e-7, atol2 = 1.e-7,fast=fast)
@test lsequal(A2,E2,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 0 && nuo == 0 && nse == rA22

A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal2(A,E,B,C,D,atol1 = 1.e-7, atol2 = 1.e-7,fast=fast)
@test lsequal(A2,E2,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 0 && nuo == 0 && nse == rA22

end

end
end

@testset "lpsminreal" begin

A2 = zeros(0,0); E2 = zeros(0,0); B2 = zeros(0,0); F2 = zeros(0,0); C2 = zeros(0,0); G2 = zeros(0,0); D2 = zeros(0,0); H2 = zeros(0,0);
sys = (A2,E2,B2,F2,C2,G2,D2,H2);

@time sys1 = lpsminreal(sys...)
@test lpsequal(sys...,sys1[1:8]...) && sys1[11:12] == (0,0)

A2 = zeros(0,0); E2 = zeros(0,0); B2 = zeros(0,3); F2 = zeros(0,3); C2 = zeros(2,0); G2 = zeros(2,0); D2 = rand(2,3); H2 = rand(2,3);
sys = (A2,E2,B2,F2,C2,G2,D2,H2);

@time sys1 = lpsminreal(sys...)
@test lpsequal(sys...,sys1[1:8]...) && sys1[11:12] == (0,0)

# Example Van Dooren & Dewilde, LAA 1983.
# strongly controllable realization

sys = ([1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0], 
[0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0], 
[0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0], 
[1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0], 
[1.0 3.0 0.0; 1.0 4.0 2.0; 0.0 -1.0 -2.0], 
[-1.0 -4.0 -2.0; -0.0 -0.0 -0.0; -1.0 -4.0 -2.0], 
[1.0 2.0 -2.0; 0.0 -1.0 -2.0; 0.0 0.0 0.0], 
[0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0]);
@time A1, E1, B1, F1, C1, G1, D1, H1, V, W, nuc, nuo  = lpsminreal(sys...);
@test lpsequal(sys..., A1, E1, B1/W, F1/W, V'\C1, V'\G1, V'\D1/W, V'\H1/W) && (nuc, nuo) == (0,2)

# Example Van Dooren & Dewilde, LAA 1983.
# strongly observable realization
sys = ([1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0], 
[0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0], 
[1.0 3.0 0.0; 1.0 4.0 2.0; 0.0 -1.0 -2.0], 
[-1.0 -4.0 -2.0; -0.0 -0.0 -0.0; -1.0 -4.0 -2.0], 
[0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0], 
[1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0], 
[1.0 2.0 -2.0; 0.0 -1.0 -2.0; 0.0 0.0 0.0], 
[0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0]);
@time A1, E1, B1, F1, C1, G1, D1, H1, V, W, nuc, nuo  = lpsminreal(sys...);
@test lpsequal(sys..., A1, E1, B1/W, F1/W, V'\C1, V'\G1, V'\D1/W, V'\H1/W) && (nuc, nuo) == (2,0)


# Example 1: DeTeran, Dopico, Mackey, ELA 2009
# P = zeros(2,2,3);
# P[:,:,1] = [0 0; 0 1.];
# P[:,:,2] = [0 1.; 1. 0];
# P[:,:,3] = [1. 0; 0 0];

# strongly controllable realization
sys = ([1.0 0.0; 0.0 1.0], [0.0 0.0; 0.0 0.0], [0.0 0.0; 0.0 0.0], [1.0 0.0; 0.0 1.0], 
[0.0 1.0; 1.0 0.0], [-1.0 -0.0; -0.0 -0.0], [0.0 0.0; 0.0 1.0], [0.0 0.0; 0.0 0.0])
@time A1, E1, B1, F1, C1, G1, D1, H1, V, W, nuc, nuo  = lpsminreal(sys...);
@test lpsequal(sys..., A1, E1, B1/W, F1/W, V'\C1, V'\G1, V'\D1/W, V'\H1/W) && (nuc, nuo) == (0, 1)

# strongly observabable realization
sys = ([1.0 0.0; 0.0 1.0], [0.0 0.0; 0.0 0.0], [0.0 1.0; 1.0 0.0], [-1.0 -0.0; -0.0 -0.0], 
[0.0 0.0; 0.0 0.0], [1.0 0.0; 0.0 1.0], [0.0 0.0; 0.0 1.0], [0.0 0.0; 0.0 0.0])
@time A1, E1, B1, F1, C1, G1, D1, H1, V, W, nuc, nuo  = lpsminreal(sys...);
@test lpsequal(sys..., A1, E1, B1/W, F1/W, V'\C1, V'\G1, V'\D1/W, V'\H1/W) && (nuc, nuo) == (1,0)


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
F2 = zeros(size(B2)...);
G2 = zeros(size(C2)...);
H2 = zeros(size(D2)...);

sys = (A2,E2,B2,F2,C2,G2,D2,H2);
@time A1, E1, B1, F1, C1, G1, D1, H1, V, W, nuc, nuo  = lpsminreal(sys...);
@test lpsequal(sys..., A1, E1, B1/W, F1/W, V'\C1, V'\G1, V'\D1/W, V'\H1/W) && (nuc, nuo) == (2,4)

sys = (A2,E2,B2,F2,C2,G2,D2,H2);
@time A1, E1, B1, F1, C1, G1, D1, H1, V, W, nuc, nuo  = lpsminreal(sys...,contr=false);
@test lpsequal(sys..., A1, E1, B1/W, F1/W, V'\C1, V'\G1, V'\D1/W, V'\H1/W) && (nuc, nuo) == (0,4)

sys = (A2,E2,B2,F2,C2,G2,D2,H2);
@time A1, E1, B1, F1, C1, G1, D1, H1, V, W, nuc, nuo  = lpsminreal(sys...,obs=false);
@test lpsequal(sys..., A1, E1, B1/W, F1/W, V'\C1, V'\G1, V'\D1/W, V'\H1/W) && (nuc, nuo) == (2,0)

A2 = [
      0 0 0 -24 0 0 0 0 0 0 0
      1 0 0 -50 0 0 0 0 0 0 0
      0 1 0 -35 0 0 0 0 0 0 0
      0 0 1 -10 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 -30 0 0 0
      0 0 0 0 1 0 0 -61 0 0 0
      0 0 0 0 0 1 0 -41 0 0 0
      0 0 0 0 0 0 1 -11 0 0 0
      0 0 0 0 0 0 0 0 0 0 -15
      0 0 0 0 0 0 0 0 1 0 -23
      0 0 0 0 0 0 0 0 0 1 -9
]
E2 = Matrix{Float64}(I,11,11);
B2 = reshape([18; 42; 30; 6; 10;17;8;1;0;-10;-2;],11,1)
C2 = reshape([0 0 0 0 0 0 0 1 0 0 0],1,11)
D2 = zeros(Int,1,1)
F2 = zeros(size(B2)...);
G2 = zeros(size(C2)...);
H2 = zeros(size(D2)...);
sys = (A2,E2,B2,F2,C2,G2,D2,H2);
@time A1, E1, B1, F1, C1, G1, D1, H1, V, W, nuc, nuo  = lpsminreal(sys...);
@test lpsequal(sys..., A1, E1, B1/W, F1/W, V'\C1, V'\G1, V'\D1/W, V'\H1/W) && (nuc, nuo) == (7, 3)

fast = true; Ty = Float64      
for fast in (true, false)

for Ty in (Float64, Complex{Float64})

#fast = true; Ty = Complex{Float64}    
n = 10; m = 5; p = 6;
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
D2 = zeros(Ty,p,m)

F2 = zeros(Ty,size(B2)...);
G2 = zeros(Ty,size(C2)...);
H2 = rand(Ty,size(D2)...);
sys = (A2,E2,B2,F2,C2,G2,D2,H2);
@time A1, E1, B1, F1, C1, G1, D1, H1, V, W, nuc, nuo  = lpsminreal(sys...,atol1 = 1.e-7, atol2 = 1.e-7,fast=fast);
@test lpsequal(sys..., A1, E1, B1/W, F1/W, V'\C1, V'\G1, V'\D1/W, V'\H1/W) && (nuc, nuo) == (5, 3)


end

end




end


end # regular testset
end # module
