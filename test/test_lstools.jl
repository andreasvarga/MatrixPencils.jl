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

@testset "lsbalance! & lsbalqual" begin

# example in https://github.com/andreasvarga/MatrixPencils.jl/issues/12

A = [-6.537773175952662 0.0 0.0 0.0 -9.892378564622923e-9 0.0; 
     0.0 -6.537773175952662 0.0 0.0 0.0 -9.892378564622923e-9; 
     2.0163803998106024e8 2.0163803998106024e8 -0.006223894167415392 -1.551620418759878e8 0.002358202548321148 0.002358202548321148;
     0.0 0.0 5.063545034365582e-9 -0.4479539754649166 0.0 0.0; 
     -2.824060629317756e8 2.0198389074625736e8 -0.006234569427701143 -1.5542817673286995e8 -0.7305736722226711 0.0023622473513548576; 
     2.0198389074625736e8 -2.824060629317756e8 -0.006234569427701143 -1.5542817673286995e8 0.0023622473513548576 -0.7305736722226711];

B = [0.004019511633336128; 0.004019511633336128; 0.0; 0.0; 297809.51426114445; 297809.51426114445]

C = [0.0 0.0 0.0 1.0 0.0 0.0]

# scale A, B and C
AA = copy(A); BB = copy(B); CC = copy(C);
qsorigABC = lsbalqual(A,B,C)
qsorigSM = lsbalqual(A,B,C; SysMat = true)
D = lsbalance!(AA,BB,CC)
@test AA == D\A*D && BB == D\B && CC == C*D
qsfinABC = lsbalqual(AA,BB,CC)
qsfinSM = lsbalqual(AA,BB,CC; SysMat = true)
@test qsfinABC <= qsfinSM && 100000*qsfinABC < qsorigABC && 100000*qsfinSM < qsorigSM 

# scale only A and compare with GEBAL
AA = copy(A); BB = copy(B); CC = copy(C);
D = lsbalance!(AA,BB,CC; withB = false, withC = false)
AA1 = copy(A);
ilo, ihi, D1 = LAPACK.gebal!('S', AA1)
@test D == Diagonal(D1) && AA == AA1

# apply to upper triangular A
AA = copy(triu(A)); BB = copy(B); CC = copy(C);
qsorigABC = lsbalqual(triu(A),B,C)
D = lsbalance!(AA,BB,CC)
@test AA == D\triu(A)*D && BB == D\B && CC == C*D
qsfinABC = lsbalqual(AA,BB,CC)
@test 100000*qsfinABC <= qsorigABC  


AA = copy(A); BB = copy(B); CC = copy(C);
D1, D = lsbalance!(AA,I,BB,CC)
@test AA == D\A*D && BB == D\B && CC == C*D && D1*D == I

# TB01ID EXAMPLE PROGRAM DATA
A = [   0.0  1.0000e+000          0.0          0.0          0.0
-1.5800e+006 -1.2570e+003          0.0          0.0          0.0
3.5410e+014          0.0 -1.4340e+003          0.0 -5.3300e+011
       0.0          0.0          0.0          0.0  1.0000e+000
       0.0          0.0          0.0 -1.8630e+004 -1.4820e+000];
B = [  0.0          0.0
      1.1030e+002          0.0
       0.0          0.0
       0.0          0.0
       0.0  8.3330e-003];
C = [ 1.0000e+000          0.0          0.0          0.0          0.0
       0.0          0.0  1.0000e+000          0.0          0.0
       0.0          0.0          0.0  1.0000e+000          0.0
6.6640e-001          0.0 -6.2000e-013          0.0          0.0
       0.0          0.0 -1.0000e-003  1.8960e+006  1.5080e+002];
qsorigABC = lsbalqual(A,B,C)
qsorigSM = lsbalqual(A,B,C; SysMat = true)

AA = copy(A); BB = copy(B); CC = copy(C);
D = lsbalance!(AA,BB,CC)
@test AA == D\A*D && BB == D\B && CC == C*D
qsfinABC = lsbalqual(AA,BB,CC)
qsfinSM = lsbalqual(AA,BB,CC; SysMat = true)
@test qsfinABC <= qsfinSM && 1000*qsfinABC < qsorigABC && 1000*qsfinSM < qsorigSM 

# MATLAB result is less satisfactory
A = [0   2.0480e+03            0            0            0
    -7.7148e+02  -1.2570e+03            0            0            0
     6.4410e+02            0  -1.4340e+03            0  -1.5512e+01
          0            0            0            0   1.2800e+02
          0            0            0  -1.4555e+02  -1.4820e+00];
B = [       0            0
   2.2589e+05            0
            0            0
            0            0
            0   2.1844e+03];
C = [   2.3842e-07            0            0            0            0
0            0   1.3107e+05            0            0
0            0            0   2.9802e-08            0
1.5888e-07            0  -8.1265e-08            0            0
0            0  -1.3107e+02   5.6505e-02   5.7526e-04];
qsorigABC = lsbalqual(A,B,C)
qsorigSM = lsbalqual(A,B,C; SysMat = true)

AA = copy(A); BB = copy(B); CC = copy(C);
D = lsbalance!(AA,BB,CC)
@test AA == D\A*D && BB == D\B && CC == C*D
qsfinABC = lsbalqual(AA,BB,CC)
qsfinSM = lsbalqual(AA,BB,CC; SysMat = true)
@test qsfinABC <= qsfinSM && 10*qsfinABC < qsorigABC && 10*qsfinSM < qsorigSM 

# warnsys example from MATLAB prescale 
# load numdemo warnsys 

A1 = [-3.6111e+05            0            0            0  -2.5000e+02            0            0            0            0   1.3629e+12            0            0
1.0000e+00            0            0            0            0            0            0            0            0            0            0            0
         0            0  -3.6111e+05            0            0  -2.5000e+02            0            0            0            0            0            0
         0            0   1.0000e+00            0            0            0            0            0            0            0            0            0
1.5632e+08   4.3423e+12            0            0            0            0            0            0            0            0            0            0
         0            0   1.5632e+08   4.3423e+12            0            0            0            0            0            0            0            0
         0            0            0            0            0            0  -1.0053e+06  -4.8669e+05            0            0            0            0
         0            0            0            0            0            0   4.8669e+05            0            0            0            0            0
         0            0            0            0            0            0  -9.9274e+05  -4.8661e+05            0            0            0            0
         0            0            0            0            0            0            0            0   1.0000e+00            0            0            0
         0            0            0            0            0            0            0            0            0            0  -1.0053e+06  -4.8669e+05
         0            0            0            0            0            0            0            0            0            0   4.8669e+05            0
         0            0            0            0            0            0            0            0            0            0  -9.9274e+05  -4.8661e+05
         0            0            0            0            0            0            0            0            0            0            0            0
         0            0            0            0   1.0000e+08            0            0            0            0            0            0            0
         0            0            0            0            0   1.0000e+08            0            0            0            0            0            0
         0            0            0            0            0            0            0            0            0            0            0            0];
A2 = [            0            0            0            0            0
0            0            0            0            0
0   1.3629e+12            0            0            0
0            0            0            0            0
0            0            0            0            0
0            0            0            0            0
0            0  -1.1364e-02            0            0
0            0            0            0            0
0            0  -1.1364e-02            0            0
0            0            0            0            0
0            0            0            0            0
0            0            0            0            0
0            0            0            0            0
1.0000e+00            0            0            0            0
0            0  -1.0000e+07  -1.0000e+07   1.0000e+07
0            0  -1.0000e+07  -1.0000e+07   1.0000e+07
0            0   1.3889e+05   1.3889e+05  -1.3889e+05];
A = [A1 A2];
B = [     0.
0
0
0
0
0
0
0
0
0
1
0
1
0
0
0
0];
C1 = [            0            0            0            0            0            0            0            0            0            0            0            0
];
C2 = [            0            0            0   1.1364e-02            0]
C = [C1 C2];
qsorigABC = lsbalqual(A,B,C)
qsorigSM = lsbalqual(A,B,C; SysMat = true)

AA = copy(A); BB = copy(B); CC = copy(C);
D = lsbalance!(AA,BB,CC)
@test AA == D\A*D && BB == D\B && CC == C*D
qsfinABC = lsbalqual(AA,BB,CC)
qsfinSM = lsbalqual(AA,BB,CC; SysMat = true)
@test qsfinABC <= qsfinSM && 100000*qsfinABC < qsorigABC && 100000*qsfinSM < qsorigSM 

# anil example from MATLAB prescale 
# load numdemo anil
A1 = [-3.4698e+07  -9.9264e+11  -3.2488e+16  -7.5886e+20  -1.2408e+25  -2.3952e+29  -2.5256e+33  -4.0288e+37  -2.9886e+41  -3.8863e+45  -2.0963e+49  -2.1520e+53 -8.4938e+56  -6.4812e+60  -1.8401e+64  -9.4353e+67  -1.7730e+71  -5.2558e+74  -3.9629e+77  -8.5612e+80]
A = [A1; I zeros(19,1)];
B = [1;zeros(19)];
C = [           0            0            0   6.8179e+19   2.0483e+23   4.8230e+28   1.4477e+32   1.3324e+37   3.8535e+40   1.8274e+45   4.9592e+48   1.3074e+53 3.2383e+56   4.7158e+60   1.0155e+64   7.7931e+67   1.2745e+71   4.8525e+74   3.2088e+77   8.6210e+80]
qsorigABC = lsbalqual(A,B,C)
qsorigSM = lsbalqual(A,B,C; SysMat = true)

AA = copy(A); BB = copy(B); CC = copy(C);
D = lsbalance!(AA,BB,CC)
@test AA == D\A*D && BB == D\B && CC == C*D
qsfinABC = lsbalqual(AA,BB,CC)
qsfinSM = lsbalqual(AA,BB,CC; SysMat = true)
@test qsfinABC <= qsfinSM && 1.e50*qsfinABC < qsorigABC && 1.e50*qsfinSM < qsorigSM 

# small example for which MATLAB does nothing
A = [3.;;]; B = [1.e5;;]; C = [4.e-5;;]; 
AA = copy(A); BB = copy(B); CC = copy(C);
@time D = lsbalance!(AA,BB,CC)
@test AA == D\A*D && BB == D\B && CC == C*D


n = 10; m = 3; p = 2; k = 10
T = rand(n,n);
T[1, 2 : n] = 10^(k)*T[1, 2 : n]
T[4:n, 3] = 10^(k)*T[4:n, 3]
D = sort(round.(Int,1. ./ rand(n)))
A = T*Diagonal(D); E = T;
ev = sort(eigvals(A,E),by = real)
corig = norm(abs.(ev-D)./sqrt.(1. .+ ev .^2)./sqrt.(1. .+ D .^2))

B = rand(n,m); B[1,:] = 10^(k)*B[1,:]; 
C = rand(p,n); C[:, 3] = 10^(k)*C[:, 3];
# B = rand(n,m); B[6,:] = 10^(k)*B[6,:]; 
# C = rand(p,n); C[:, 2] = 10^(k)*C[:, 2];


qsorigABC = lsbalqual(A,E,B,C)
qsorigSM = lsbalqual(A,E,B,C; SysMat = true)
AA = copy(A); EE = copy(E); BB = copy(B); CC = copy(C);
#@time d1, d2 = lsbalance!(AA,EE,BB,CC; withB = false, withC = false)
@time D1, D2 = lsbalance!(AA,EE,BB,CC)
@test AA == D1*A*D2 && EE == D1*E*D2 && BB == D1*B && CC == C*D2

qsfinABC = lsbalqual(AA,EE,BB,CC)
qsfinSM = lsbalqual(AA,EE,BB,CC; SysMat = true)
@test qsfinABC < 100000*qsorigABC && 100000*qsfinSM < qsorigSM 
evs = sort(eigvals(AA,EE),by = real)
@test evs ≈ D
cofin = norm(abs.(evs-D)./sqrt.(1. .+ evs .^2)./sqrt.(1. .+ D .^2))
@test ev ≈ D ? cofin/10 < corig :  cofin < corig 

# eigenvalue oriented scaling of only A and E
qsorigABC = lsbalqual(A,E,B,C)
qsorigSM = lsbalqual(A,E,B,C; SysMat = true)
AA = copy(A); EE = copy(E); BB = copy(B); CC = copy(C);
@time D1, D2 = lsbalance!(AA,EE,BB,CC; withB = false, withC = false)
@test AA == D1*A*D2 && EE == D1*E*D2 && BB == D1*B && CC == C*D2

qsfinABC = lsbalqual(AA,EE,BB,CC)
qsfinSM = lsbalqual(AA,EE,BB,CC; SysMat = true)
@test qsfinABC < 100000*qsorigABC && 100000*qsfinSM < qsorigSM 

evs = sort(eigvals(AA,EE),by=real)
@test evs ≈ D


# optimal scalings
AA = copy(A); EE = copy(E); BB = copy(B); CC = copy(C);
@time D1, D2 = lsbalance!(AA,EE,BB,CC,pow2 = false)
@test AA ≈ D1*A*D2 && EE ≈ D1*E*D2 && BB ≈ D1*B && CC ≈ C*D2
qsfinABC = lsbalqual(AA,EE,BB,CC)
qsfinSM = lsbalqual(AA,EE,BB,CC; SysMat = true)
evs = sort(eigvals(AA,EE),by=real)
cofin = norm(abs.(evs-D)./sqrt.(1. .+ evs .^2)./sqrt.(1. .+ D .^2))


# small example for which MATLAB does nothing
A = [0.;;]; E = [3.;;]; B = [1.e5;;]; C = [4.e5;;]; 
AA = copy(A); EE = copy(E); BB = copy(B); CC = copy(C);
@time D1, D2 = lsbalance!(AA,EE,BB,CC,pow2 = false)
@test AA ≈ D1*A*D2 && EE ≈ D1*E*D2 && BB ≈ D1*B && CC ≈ C*D2




end


end # regular testset
end # module
