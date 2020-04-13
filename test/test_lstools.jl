module Test_linsystools

using LinearAlgebra
using MatrixPencils
using Test


@testset "Linear System Tools" begin



@testset "lsminreal and lsminreal2" begin

A2 = zeros(0,0); E2 = zeros(0,0); C2 = zeros(0,0); B2 = zeros(0,0); D2 = zeros(0,0);
sys = (A2,E2,B2,C2,D2);

@time sys1 = lsminreal(sys...)
@test lsequal(sys...,sys1[1:5]...) && sys1[6:8] == (0,0,0)

@time sys1 = lsminreal2(sys...)
@test lsequal(sys...,sys1[1:5]...) && sys1[6:8] == (0,0,0)


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

@time sys1 = lsminreal(sys...)
@test lsequal(sys...,sys1[1:5]...) && sys1[6:8] == (2,0,1)  # nuc == 2 && nuo == 0 && nse == 1
@time val, iz, info = spzeros(sys1[1:5]...) 
@test val ≈ Float64[] && iz == [] &&  (info.rki, info.lki, info.id, info.nf) == ([1], [1], [1, 1], 0)
@time val, info = speigvals(sys1[1:5]...) 
@test val ≈ [Inf, Inf] &&  (info.rki, info.lki, info.id, info.nf) == ([1], [1], [1, 1], 0)

@time sys1 = lsminreal2(sys...)
@test lsequal(sys...,sys1[1:5]...) && sys1[6:8] == (2,0,1)  # nuc == 2 && nuo == 0 && nse == 1
@time val, iz, info = spzeros(sys1[1:5]...) 
@test val ≈ Float64[] && iz == [] &&  (info.rki, info.lki, info.id, info.nf) == ([1], [1], [1, 1], 0)
@time val, info = speigvals(sys1[1:5]...) 
@test val ≈ [Inf, Inf] &&  (info.rki, info.lki, info.id, info.nf) == ([1], [1], [1, 1], 0)


#A2,E2,B2,C2,D2 = pm2ls(P)
A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal(A,E,B,C,D)
@test lsequal(A,E,B,C,D,A1,E1,B1,C1,D1) &&
      nuc == 2 && nuo == 0 && nse == 1
@time val, iz, info = spzeros(A1,E1,B1,C1,D1) 
@test val ≈ Float64[] && iz == [] &&  (info.rki, info.lki, info.id, info.nf) == ([1], [1], [1, 1], 0)
@time val, info = speigvals(A1,E1,B1,C1,D1) 
@test val ≈ [Inf, Inf] &&  (info.rki, info.lki, info.id, info.nf) == ([1], [1], [1, 1], 0)

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
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal(A,E,B,C,D);
@test lsequal(A2,E2,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 5 && nuo == 0 && nse == 1
@time val, iz, info = spzeros(A1,E1,B1,C1,D1) 
@test val ≈ [1.0] && iz == [] &&  (info.rki, info.lki, info.id, info.nf) == ([0], [1], [1, 1, 1], 1)
@time val, info = speigvals(A1,E1,B1,C1,D1) 
@test val ≈ [1., Inf, Inf, Inf] &&  (info.rki, info.lki, info.id, info.nf) == ([0], [1], [1, 1, 1], 1)

A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal(A,E,B,C,D,obs=false);
@test lsequal(A2,E2,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 5 && nuo == 0 && nse == 1
@time val, iz, info = spzeros(A1,E1,B1,C1,D1) 
@test val ≈ [1.0] && iz == [] &&  (info.rki, info.lki, info.id, info.nf) == ([0], [1], [1, 1, 1], 1)
@time val, info = speigvals(A1,E1,B1,C1,D1) 
@test val ≈ [1., Inf, Inf, Inf] &&  (info.rki, info.lki, info.id, info.nf) == ([0], [1], [1, 1, 1], 1)


A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal2(A,E,B,C,D);
@test lsequal(A2,E2,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 5 && nuo == 0 && nse == 1
@time val, iz, info = spzeros(A1,E1,B1,C1,D1) 
@test val ≈ [1.0] && iz == [] &&  (info.rki, info.lki, info.id, info.nf) == ([0], [1], [1, 1, 1], 1)
@time val, info = speigvals(A1,E1,B1,C1,D1) 
@test val ≈ [1., Inf, Inf, Inf] &&  (info.rki, info.lki, info.id, info.nf) == ([0], [1], [1, 1, 1], 1)

A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal2(A,E,B,C,D,obs=false,finite=false);
@test lsequal(A2,E2,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 5 && nuo == 0 && nse == 1
@time val, iz, info = spzeros(A1,E1,B1,C1,D1) 
@test val ≈ [1.0] && iz == [] &&  (info.rki, info.lki, info.id, info.nf) == ([0], [1], [1, 1, 1], 1)
@time val, info = speigvals(A1,E1,B1,C1,D1) 
@test val ≈ [1., Inf, Inf, Inf] &&  (info.rki, info.lki, info.id, info.nf) == ([0], [1], [1, 1, 1], 1)

# perform lsminreal with A and E interchanged
A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time E1, A1, B1, C1, D1, nuc, nuo, nse  = lsminreal(E,A,B,C,D,obs = false)
@test lsequal(A2,E2,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 5 && nuo == 0 && nse == 0

# perform lsminreal2 with A and E interchanged
A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time E1, A1, B1, C1, D1, nuc, nuo, nse  = lsminreal2(E,A,B,C,D,obs = false)
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
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal(A,E,B,C,D)
@test lsequal(A2,E2,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 0 && nuo == 5 && nse == 1

A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal(A,E,B,C,D,contr = false);
@test lsequal(A2,E2,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 0 && nuo == 5 && nse == 1

A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal2(A,E,B,C,D)
@test lsequal(A2,E2,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 0 && nuo == 5 && nse == 1

A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal2(A,E,B,C,D,contr = false,finite=false);
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
     lsminreal(A,E,B,C,D,atol1 = 1.e-7, atol2 = 1.e-7)
@test lsequal(A2,E2,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 0 && nuo == 5 && nse == 1
      
A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = 
     lsminreal(A,E,B,C,D,atol1 = 1.e-7, atol2 = 1.e-7,contr=false);
@test lsequal(A2,E2,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 0 && nuo == 5 && nse == 1

A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal(A,I,B,C,D,atol1 = 1.e-7, atol2 = 1.e-7)
@test lsequal(A2,I,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 6 && nuo == 3 && nse == 0

A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal2(A,E,B,C,D,atol1 = 1.e-7, atol2 = 1.e-7)
@test lsequal(A2,E2,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 0 && nuo == 5 && nse == 1

A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = 
      lsminreal2(A,E,B,C,D,contr=false,finite=false,atol1 = 1.e-7, atol2 = 1.e-7);
@test lsequal(A2,E2,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 0 && nuo == 5 && nse == 1

A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal2(A,I,B,C,D,atol1 = 1.e-7, atol2 = 1.e-7)
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
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal(A,E,B,C,D,atol1 = 1.e-7, atol2 = 1.e-7,noseig=true)
@test lsequal(A2,E2,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 0 && nuo == 2 && nse == 0

A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal(A,E,B,C,D,obs=false,atol1 = 1.e-7, atol2 = 1.e-7)
@test lsequal(A2,E2,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 0 && nuo == 0 && nse == 0

A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal(A,E,B,C,D,contr=false,atol1 = 1.e-7, atol2 = 1.e-7)
@test lsequal(A2,E2,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 0 && nuo == 2 && nse == 0


A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal2(A,E,B,C,D,atol1 = 1.e-7, atol2 = 1.e-7,noseig=true)
@test lsequal(A2,E2,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 0 && nuo == 2 && nse == 0

A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal2(A,E,B,C,D,obs=false,atol1 = 1.e-7, atol2 = 1.e-7)
@test lsequal(A2,E2,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 0 && nuo == 0 && nse == 0

A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal2(A,E,B,C,D,contr=false,finite=false,atol1 = 1.e-7, atol2 = 1.e-7)
@test lsequal(A2,E2,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 0 && nuo == 2 && nse == 0

A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal(A',E',C',B',D',atol1 = 1.e-7, atol2 = 1.e-7)
@test lsequal(A2,E2,B2,C2,D2,A1',E1',C1',B1',D1') &&
      nuc == 2 && nuo == 0 && nse == 0

A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal2(A',E',C',B',D',atol1 = 1.e-7, atol2 = 1.e-7)
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
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal(A,E,B,C,D,atol1 = 1.e-7, atol2 = 1.e-7)
@test lsequal(A2,E2,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 6 && nuo == 0 && nse == 1

A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal2(A,E,B,C,D,atol1 = 1.e-7, atol2 = 1.e-7)
@test lsequal(A2,E2,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 6 && nuo == 0 && nse == 1


A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal(A',E',C',B',D',atol1 = 1.e-7, atol2 = 1.e-7)
@test lsequal(A2,E2,B2,C2,D2,A1',E1',C1',B1',D1') &&
      nuc == 2 && nuo == 4 && nse == 1 

A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal2(A',E',C',B',D',atol1 = 1.e-7, atol2 = 1.e-7)
@test lsequal(A2,E2,B2,C2,D2,A1',E1',C1',B1',D1') &&
      nuc == 2 && nuo == 4 && nse == 1 

A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal(A',E',C',B',D',contr=false, atol1 = 1.e-7, atol2 = 1.e-7)
@test lsequal(A2,E2,B2,C2,D2,A1',E1',C1',B1',D1') &&
      nuc == 0 && nuo == 6 && nse == 1 

A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal2(A',E',C',B',D',contr=false, atol1 = 1.e-7, atol2 = 1.e-7)
@test lsequal(A2,E2,B2,C2,D2,A1',E1',C1',B1',D1') &&
      nuc == 0 && nuo == 6 && nse == 1 

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
B2 = reshape([18; 42; 30; 6; 10;17;8;1;0;-10;-2;],11,1)
C2 = reshape([0 0 0 0 0 0 0 1 0 0 0],1,11)
D2 = zeros(Int,1,1)

A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal(A,E,B,C,D,atol1 = 1.e-7, atol2 = 1.e-7)
@test lsequal(A2,E2,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 7 && nuo == 3 && nse == 0

A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2); D = copy(D2);
@time A1, E1, B1, C1, D1, nuc, nuo, nse  = lsminreal2(A,E,B,C,D,atol1 = 1.e-7, atol2 = 1.e-7)
@test lsequal(A2,E2,B2,C2,D2,A1,E1,B1,C1,D1) &&
      nuc == 7 && nuo == 3 && nse == 0

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
end # module
