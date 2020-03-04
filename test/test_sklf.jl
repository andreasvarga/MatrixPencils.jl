module Test_sklf

using LinearAlgebra
using MatrixPencils
using Test

   

@testset "Structured Matrix Pencils Utilities" begin

@testset "sreduceBF" begin

fast = true; Ty = Float64; Ty = Complex{Float64}     
for fast in (true, false)

for Ty in (Float64, Complex{Float64})


n2 = 3; m2 = 2; p2 = 4; 
A2 = rand(Ty,n2,n2); E2 = zeros(Ty,n2,n2); B2 = zeros(Ty,n2,m2); C2 = zeros(Ty,p2,n2); D2 = zeros(Ty,p2,m2);
A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2); D = copy(D2); 
M2 = [A2 B2; C2 D2]
N2 = [E zeros(Ty,n2,m2); zeros(Ty,p2,n2+m2)]

@time M, N, Q, Z, n, m, p = sreduceBF(A,E,B,C,D,fast = fast)
@test norm(Q'*M2*Z-M) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N) < sqrt(eps(1.)) &&
      n == rank(E) && m == n2-n+m2 && p == n2-n+p2  
      
n2 = 3; m2 = 2; p2 = 4; 
A2 = rand(Ty,n2,n2); E2 = triu(rand(Ty,n2,n2),1); B2 = rand(Ty,n2,m2); C2 = rand(Ty,p2,n2); D2 = rand(Ty,p2,m2);
A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2); D = copy(D2); 
M2 = [A2 B2; C2 D2]
N2 = [E zeros(Ty,n2,m2); zeros(Ty,p2,n2+m2)]

@time M, N, Q, Z, n, m, p = sreduceBF(A,E,B,C,D,fast = fast)
@test norm(Q'*M2*Z-M) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N) < sqrt(eps(1.)) &&
      n == rank(E) && m == n2-n+m2 && p == n2-n+p2     

n2 = 0; m2 = 2; p2 = 4; 
A2 = rand(Ty,n2,n2); E2 = triu(rand(Ty,n2,n2),1); B2 = rand(Ty,n2,m2); C2 = rand(Ty,p2,n2); D2 = rand(Ty,p2,m2);
A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2); D = copy(D2); 
M2 = [A2 B2; C2 D2]
N2 = [E zeros(Ty,n2,m2); zeros(Ty,p2,n2+m2)]

@time M, N, Q, Z, n, m, p = sreduceBF(A,E,B,C,D,fast = fast)
@test norm(Q'*M2*Z-M) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N) < sqrt(eps(1.)) &&
      n == rank(E) && m == n2-n+m2 && p == n2-n+p2 

      #
n2 = 0; m2 = 2; p2 = 4; 
A2 = rand(Ty,n2,n2); E2 = triu(rand(Ty,n2,n2),1); B2 = rand(Ty,n2,m2); C2 = rand(Ty,p2,n2); D2 = rand(Ty,p2,m2);
A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2); D = copy(D2); 
M2 = [A2 B2; C2 D2]
N2 = [E zeros(Ty,n2,m2); zeros(Ty,p2,n2+m2)]

@time M, N, Q, Z, n, m, p = sreduceBF(missing,missing,missing,missing,D,fast = fast)
@test norm(Q'*M2*Z-M) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N) < sqrt(eps(1.)) &&
      n == rank(E) && m == n2-n+m2 && p == n2-n+p2 
    
n2 = 3; m2 = 0; p2 = 4; 
A2 = rand(Ty,n2,n2); E2 = triu(rand(Ty,n2,n2),1); B2 = rand(Ty,n2,m2); C2 = rand(Ty,p2,n2); D2 = rand(Ty,p2,m2);
A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2); D = copy(D2); 
M2 = [A2 B2; C2 D2]
N2 = [E zeros(Ty,n2,m2); zeros(Ty,p2,n2+m2)]

@time M, N, Q, Z, n, m, p = sreduceBF(A,E,missing,C,missing,fast = fast)
@test norm(Q'*M2*Z-M) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N) < sqrt(eps(1.)) &&
      n == rank(E) && m == n2-n+m2 && p == n2-n+p2 
   
n2 = 3; m2 = 3; p2 = 0; 
A2 = rand(Ty,n2,n2); E2 = triu(rand(Ty,n2,n2),1); B2 = rand(Ty,n2,m2); C2 = rand(Ty,p2,n2); D2 = rand(Ty,p2,m2);
A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2); D = copy(D2); 
M2 = [A2 B2; C2 D2]
N2 = [E zeros(Ty,n2,m2); zeros(Ty,p2,n2+m2)]

@time M, N, Q, Z, n, m, p = sreduceBF(A,E,B,missing,missing,fast = fast)
@test norm(Q'*M2*Z-M) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N) < sqrt(eps(1.)) &&
      n == rank(E) && m == n2-n+m2 && p == n2-n+p2 
  
n2 = 3; m2 = 0; p2 = 0; 
A2 = rand(Ty,n2,n2); E2 = triu(rand(Ty,n2,n2),1); B2 = rand(Ty,n2,m2); C2 = rand(Ty,p2,n2); D2 = rand(Ty,p2,m2);
A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2); D = copy(D2); 
M2 = [A2 B2; C2 D2]
N2 = [E zeros(Ty,n2,m2); zeros(Ty,p2,n2+m2)]

@time M, N, Q, Z, n, m, p = sreduceBF(A,E,missing,missing,missing,fast = fast)
@test norm(Q'*M2*Z-M) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N) < sqrt(eps(1.)) &&
      n == rank(E) && m == n2-n+m2 && p == n2-n+p2 

n2 = 3; m2 = 0; p2 = 0; 
A2 = rand(Ty,n2,n2); E2 = I; B2 = rand(Ty,n2,m2); C2 = rand(Ty,p2,n2); D2 = rand(Ty,p2,m2);
A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2); D = copy(D2); 
M2 = [A2 B2; C2 D2]
N2 = [E zeros(Ty,n2,m2); zeros(Ty,p2,n2+m2)]

@time M, N, Q, Z, n, m, p = sreduceBF(A,E,missing,missing,missing,fast = fast)
@test norm(Q'*M2*Z-M) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N) < sqrt(eps(1.)) &&
      n == n2 && m == n2-n+m2 && p == n2-n+p2 


n2 = 3; m2 = 3; p2 = 2; 
A2 = rand(Ty,n2,n2); E2 = I; B2 = rand(Ty,n2,m2); C2 = rand(Ty,p2,n2); D2 = rand(Ty,p2,m2);
A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2); D = copy(D2); 
M2 = [A2 B2; C2 D2]
N2 = [E zeros(Ty,n2,m2); zeros(Ty,p2,n2+m2)]

@time M, N, Q, Z, n, m, p = sreduceBF(A,E,B,C,D,fast = fast)
@test norm(Q'*M2*Z-M) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N) < sqrt(eps(1.)) &&
      n == n2 && m == n2-n+m2 && p == n2-n+p2 
     

end
end
end


@testset "sklf, sklf_right and sklf_left" begin

fast = true; Ty = Float64; Ty = Complex{Float64}     
for fast in (true, false)
    # test example for SLICOT subroutine AB08ND
A = [
   1.0   0.0   0.0   0.0   0.0   0.0
   0.0   1.0   0.0   0.0   0.0   0.0
   0.0   0.0   3.0   0.0   0.0   0.0
   0.0   0.0   0.0  -4.0   0.0   0.0
   0.0   0.0   0.0   0.0  -1.0   0.0
   0.0   0.0   0.0   0.0   0.0   3.0];
E = I;
B = [
   0.0  -1.0
  -1.0   0.0
   1.0  -1.0
   0.0   0.0
   0.0   1.0
  -1.0  -1.0];
C = [
   1.0   0.0   0.0   1.0   0.0   0.0
   0.0   1.0   0.0   1.0   0.0   1.0
   0.0   0.0   1.0   0.0   0.0   1.0];
D = [
   0.0   0.0
   0.0   0.0
   0.0   0.0]; 
M2 = [A B; C D]; N2 = [E zeros(6,2); zeros(3,8)];  

M = copy(M2); N = copy(N2);
M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl = sklf(A, E, B, C, D, fast = fast, finite_infinite = false, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [] && μr == [] && νi == [2, 2] && νl == [1, 1, 1] && μl == [0, 1, 1] && nf == 2

M = copy(M2); N = copy(N2);
M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl = sklf(A, E, B, C, D, fast = fast, finite_infinite = true, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [] && μr == [] && νi == [2, 2] && νl == [1, 1, 1] && μl == [0, 1, 1] && nf == 2

M2 = [A; C]
N2 = [E; zeros(3,6)]
M = copy(M2); N = copy(N2);
M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl = sklf(A, E, missing, C, missing, fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [] && μr == [] && νi == [] && νl == [2, 3, 3] && μl == [0, 2, 3] && nf == 1

M2 = [A B]
N2 = [E zeros(6,2)]
M = copy(M2); N = copy(N2);
M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl = sklf(A, E, B, missing, missing, fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [2, 2, 1, 0] && μr == [2, 2, 2, 1] && νi == [] && νl == [] && μl == [] && nf == 1


M2 = [A B; C D]; N2 = [E zeros(6,2); zeros(3,8)];  
M = copy(M2); N = copy(N2);
M1, N1, Q1, Z1, νr, μr, nf, ν, μ  = sklf_right(A, E, B, C, D, fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [] && μr == [] && ν == [1, 3, 3] && μ == [0, 3, 3] && nf == 2

M2 = [A B; C D]; N2 = [E zeros(6,2); zeros(3,8)];  
M = copy(M2); N = copy(N2);
M1, N1, Q1, Z1, ν, μ, nf, νl, μl = sklf_left(A, E, B, C, D, fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      ν == [2, 2] && μ == [2, 2] && νl == [1, 1, 1] && μl == [0, 1, 1] && nf == 2

# test example for SLICOT subroutine AB08ND
A = [
    1     0     0     0     0     0     0     0     0
    0     1     0     0     0     0     0     0     0
    0     0     1     0     0     0     0     0     0
    0     0     0     1     0     0     0     0     0
    0     0     0     0     1     0     0     0     0
    0     0     0     0     0     1     0     0     0
    0     0     0     0     0     0     1     0     0
    0     0     0     0     0     0     0     1     0
    0     0     0     0     0     0     0     0     1];
E = [
    0     0     0     0     0     0     0     0     0
    1     0     0     0     0     0     0     0     0
    0     1     0     0     0     0     0     0     0
    0     0     0     0     0     0     0     0     0
    0     0     0     1     0     0     0     0     0
    0     0     0     0     1     0     0     0     0
    0     0     0     0     0     0     0     0     0
    0     0     0     0     0     0     1     0     0
    0     0     0     0     0     0     0     1     0];
B = [
    -1     0     0
     0     0     0
     0     0     0
     0    -1     0
     0     0     0
     0     0     0
     0     0    -1
     0     0     0
     0     0     0];
C = [
    0     1     1     0     3     4     0     0     2
    0     1     0     0     4     0     0     2     0
    0     0     1     0    -1     4     0    -2     2];
D = [
    1     2    -2
    0    -1    -2
    0     0     0]; 
M2 = [A B; C D]; N2 = [E zeros(9,3); zeros(3,12)];

M = copy(M2); N = copy(N2);
M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl = sklf(A, E, B, C, D, fast = fast, finite_infinite = false, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [1, 1, 0] && μr == [1, 1, 1 ] && νi == [1, 1, 5] && νl == [1, 1] && μl == [0, 1] && nf == 1

M = copy(M2); N = copy(N2);
M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl = sklf(A, E, B, C, D, fast = fast, finite_infinite = true, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [1, 1, 0] && μr == [1, 1, 1 ] && νi == [5, 1, 1] && νl == [1, 1] && μl == [0, 1] && nf == 1

M2 = [A; C]
N2 = [E; zeros(3,9)]
M = copy(M2); N = copy(N2);
M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl = sklf(A, E, missing, C, missing, fast = fast, finite_infinite = false, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [] && μr == [] && νi == [2, 2, 3] && νl == [2, 3] && μl == [0, 2] && nf == 0

M2 = [A; C]
N2 = [E; zeros(3,9)]
M = copy(M2); N = copy(N2);
M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl = sklf(A, E, missing, C, missing, fast = fast, finite_infinite = true, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [] && μr == [] && νi == [3, 2, 2] && νl == [2, 3] && μl == [0, 2] && nf == 0


M2 = [A B]
N2 = [E zeros(9,3)]
M = copy(M2); N = copy(N2);
M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl = sklf(A, E, B, missing, missing, fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [3, 3, 0] && μr == [3, 3, 3] && νi == [3] && νl == [] && μl == [] && nf == 0


M2 = [A B; C D]; N2 = [E zeros(9,3); zeros(3,12)];  
M = copy(M2); N = copy(N2);
M1, N1, Q1, Z1, νr, μr, nf, ν, μ  = sklf_right(A, E, B, C, D, fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [1, 1, 0] && μr == [1, 1, 1] && ν == [1, 2, 6] && μ == [1, 1, 6] && nf == 1

M2 = [A B; C D]; N2 = [E zeros(9,3); zeros(3,12)];  
M = copy(M2); N = copy(N2);
M1, N1, Q1, Z1, ν, μ, nf, νl, μl = sklf_left(A, E, B, C, D, fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      ν == [6, 2, 1] && μ == [6, 2, 2] && νl == [1, 1] && μl == [0, 1] && nf == 1

end
end

@testset "sklf_left! - generalized case" begin
A2 = zeros(0,0); E2 = zeros(0,0); C2 = zeros(0,0); B2 = missing;
A = copy(A2); E = copy(E2); C = copy(C2); B = missing;

@time Q, Z, μl, no, nfuo, niuo  = sklf_left!(A,E,C,B)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(C2*Z-C) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      μl == [] && no == 0 && nfuo == 0 && niuo == 0


A2 = zeros(0,0); E2 = zeros(0,0); C2 = zeros(3,0); B2 = missing;
A = copy(A2); E = copy(E2); C = copy(C2); B = missing;

@time Q, Z, μl, no, nfuo, niuo  = sklf_left!(A,E,C,B)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(C2*Z-C) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      μl == [0] && no == 0 && nfuo == 0 && niuo == 0

A2 = rand(3,3); E2 = zeros(3,3); C2 = zeros(0,3); B2 = rand(3,2);
A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2);

@time Q, Z, μl, no, nfuo, niuo  = sklf_left!(A,E,C,B)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(C2*Z-C) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      μl == [] && no == 0 && nfuo == 0 && niuo == 3

E2 = rand(3,3); A2 = zeros(3,3); C2 = zeros(0,3); B2 = rand(3,2); #B2 = Matrix{Float64}(I,3,3);
A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2);

@time Q, Z, μl, no, nfuo, niuo  = sklf_left!(A,E,C,B,fast=true)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(C2*Z-C) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      μl == [] && no == 0 && nfuo == 3 && niuo == 0


A2 = rand(3,3); E2 = rand(3,3); C2 = rand(1,3); B2 = rand(3,2);
A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2);

@time Q, Z, μl, no, nfuo, niuo  = sklf_left!(A,E,C,B,fast=true)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(C2*Z-C) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      μl == [1, 1, 1] && no == 3 && nfuo == 0 && niuo == 0


E2 = rand(3,3); A2 = rand(3,3); C2 = rand(1,3); B2 = rand(3,2);
A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2);

@time Q, Z, μl, no, nfuo, niuo  = sklf_left!(A,E,C,B,fast=true)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(C2*Z-C) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      μl == [1, 1, 1] && no == 3 && nfuo == 0 && niuo == 0

Ty = Complex{Float64}; fast = true
Ty = Float64; fast = true
for fast in (true, false)

for Ty in (Float64, Complex{Float64})

A2 = rand(Ty,3,3); E2 = rand(Ty,3,3); C2 = rand(Ty,1,3); B2 = rand(Ty,3,2);
A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2);

@time Q, Z, μl, no, nfuo, niuo  = sklf_left!(A,E,C,B,fast=fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(C2*Z-C) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      μl == [1, 1, 1] && no == 3 && nfuo == 0 && niuo == 0

A2 = rand(Ty,3,3); E2 = rand(Ty,3,3); C2 = rand(Ty,2,3); B2 = rand(Ty,3,2);
A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2);

@time Q, Z, μl, no, nfuo, niuo  = sklf_left!(A,E,C,B,fast=fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(C2*Z-C) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      μl == [1, 2] && no == 3 && nfuo == 0 && niuo == 0


A2 = rand(Ty,3,3); E2 = rand(Ty,3,3); C2 = rand(Ty,4,3); B2 = rand(Ty,3,2);
A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2);

@time Q, Z, μl, no, nfuo, niuo  = sklf_left!(A,E,C,B,fast=fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(C2*Z-C) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      μl == [3] && no == 3 && nfuo == 0 && niuo == 0


A2 = [rand(Ty,3,7) ; zeros(Ty,4,3) rand(Ty,4,4) ]; 
E2 = [rand(Ty,3,7) ; zeros(Ty,4,3) rand(Ty,4,4) ]; 
B2 = rand(Ty,7,4); C2 = [ zeros(Ty,2,3) rand(Ty,2,4)];
A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2);

@time Q, Z, μl, no, nfuo, niuo  = sklf_left!(A,E,C,B,fast=fast,atol1=1.e-7,atol2=1.e-7)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(C2*Z-C) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      μl == [2, 2] && no == 4 && nfuo == 3 && niuo == 0

A2 = [rand(Ty,3,3) zeros(Ty,3,4) ; zeros(Ty,4,3) rand(Ty,4,4) ]; 
E2 = [rand(Ty,3,3) zeros(Ty,3,4); rand(Ty,4,3) triu(rand(Ty,4,4),1) ]; 
B2 = rand(Ty,7,4); C2 = [ rand(Ty,2,3) zeros(Ty,2,4)];
A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2);

@time Q, Z, μl, no, nfuo, niuo  = sklf_left!(A,E,C,B,fast=fast,atol1=1.e-7,atol2=1.e-7)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(C2*Z-C) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      μl == [1, 2] && no == 3 && nfuo == 3 && niuo == 1

A2 = [rand(Ty,3,3) zeros(Ty,3,4) ; zeros(Ty,4,3) triu(rand(Ty,4,4)) ]; 
E2 = [rand(Ty,3,3) zeros(Ty,3,4); rand(Ty,4,3) triu(rand(Ty,4,4),1) ]; 
B2 = rand(Ty,7,4); C2 = [ rand(Ty,2,3) zeros(Ty,2,4)];
A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2);

@time Q, Z, μl, no, nfuo, niuo  = sklf_left!(A,E,C,B,fast=fast,atol1=1.e-7,atol2=1.e-7)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(C2*Z-C) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      μl == [1, 2] && no == 3 && nfuo == 0 && niuo == 4

end
end
end

@testset "sklf_right! - generalized case" begin

A2 = zeros(0,0); E2 = zeros(0,0); B2 = zeros(0,0); C2 = missing;
A = copy(A2); E = copy(E2); B = copy(B2); C = missing;

@time Q, Z, νr, nc, nfu, niu  = sklf_right!(A,E,B,C)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Z-C) < sqrt(eps(1.))) && 
      νr == [] && nc == 0 && nfu == 0 && niu == 0

A2 = zeros(0,0); E2 = zeros(0,0); B2 = zeros(0,3); C2 = missing;
A = copy(A2); E = copy(E2); B = copy(B2); C = missing;

@time Q, Z, νr, nc, nfu, niu  = sklf_right!(A,E,B,C)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Z-C) < sqrt(eps(1.))) && 
      νr == [0] && nc == 0 && nfu == 0 && niu == 0


A2 = zeros(3,3); E2 = rand(3,3); B2 = zeros(3,0); C2 = rand(2,3);
A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2);

@time Q, Z, νr, nc, nfu, niu  = sklf_right!(A,E,B,C)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Z-C) < sqrt(eps(1.))) && 
      νr == [] && nc == 0 && nfu == 3 && niu == 0

A2 = rand(3,3); E2 = zeros(3,3); B2 = zeros(3,0); C2 = rand(2,3);
A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2);

@time Q, Z, νr, nc, nfu, niu  = sklf_right!(A,E,B,C)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Z-C) < sqrt(eps(1.))) && 
      νr == [] && nc == 0 && nfu == 0 && niu == 3


A2 = rand(3,3); E2 = rand(3,3); B2 = zeros(3,1); C2 = rand(2,3);
A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2);

@time Q, Z, νr, nc, nfu, niu  = sklf_right!(A,E,B,C)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Z-C) < sqrt(eps(1.))) && 
      νr == [] && nc == 0 && nfu == 3 && niu == 0

A2 = rand(3,3); E2 = rand(3,3); B2 = rand(3,1); C2 = rand(2,3);
A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2);

@time Q, Z, νr, nc, nfu, niu  = sklf_right!(A,E,B,C)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Z-C) < sqrt(eps(1.))) && 
      νr == [1, 1, 1] && nc == 3 && nfu == 0 && niu == 0

Ty = Complex{Float64}
Ty = Float64
for fast in (true, false)

for Ty in (Float64, Complex{Float64})

A2 = rand(Ty,3,3); E2 = triu(rand(Ty,3,3),1); B2 = rand(Ty,3,1); C2 = rand(Ty,2,3);
A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2);

@time Q, Z, νr, nc, nfu, niu  = sklf_right!(A,E,B,C,fast = fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Z-C) < sqrt(eps(1.))) && 
      νr == [1, 1, 1] && nc == 3 && nfu == 0 && niu == 0

A2 = rand(Ty,3,3); E2 = triu(rand(Ty,3,3),1); B2 = rand(Ty,3,2); C2 = rand(Ty,2,3);
A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2);

@time Q, Z, νr, nc, nfu, niu  = sklf_right!(A,E,B,C,fast = fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Z-C) < sqrt(eps(1.))) && 
      νr == [2, 1] && nc == 3 && nfu == 0 && niu == 0

A2 = rand(Ty,3,3); E2 = rand(Ty,3,3); B2 = rand(Ty,3,4); C2 = rand(Ty,2,3);
A = copy(A2); E = copy(E2);  B = copy(B2); C = copy(C2);

@time Q, Z, νr, nc, nfu, niu  = sklf_right!(A,E,B,C,fast = fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Z-C) < sqrt(eps(1.))) && 
      νr == [3] && nc == 3 && nfu == 0 && niu == 0

Ty = Float64; fast = true;
A2 = [rand(Ty,3,7) ; zeros(Ty,4,3) rand(Ty,4,4) ]; 
E2 = [rand(Ty,3,7) ; zeros(Ty,4,3) triu(rand(Ty,4,4),1) ]; 
B2 = [rand(Ty,3,4); zeros(Ty,4,4)]; C2 = rand(Ty,2,7);
A = copy(A2); E = copy(E2);  B = copy(B2); C = copy(C2);


@time Q, Z, νr, nc, nfu, niu  = sklf_right!(A,E,B,C,atol1=1.e-7,atol2=1.e-7,fast = fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Z-C) < sqrt(eps(1.))) && 
      νr == [3] && nc == 3 && nfu == 3 && niu == 1

A2 = [rand(Ty,3,7) ; zeros(Ty,4,3) triu(rand(Ty,4,4)) ]; 
E2 = [rand(Ty,3,7) ; zeros(Ty,4,3) triu(rand(Ty,4,4),1) ]; 
B2 = [rand(Ty,3,4); zeros(Ty,4,4)]; C2 = rand(Ty,2,7);
A = copy(A2); E = copy(E2);  B = copy(B2); C = copy(C2);

@time Q, Z, νr, nc, nfu, niu  = sklf_right!(A,E,B,C,atol1=1.e-7,atol2=1.e-7,fast = fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Z-C) < sqrt(eps(1.))) && 
      νr == [3] && nc == 3 && nfu == 0 && niu == 4

A2 = [rand(Ty,3,7) ; zeros(Ty,4,3) rand(Ty,4,4) ]; 
E2 = [rand(Ty,3,7) ; zeros(Ty,4,3) rand(Ty,4,4) ]; 
B2 = [rand(Ty,3,4); zeros(Ty,4,4)]; C2 = rand(Ty,2,7);
A = copy(A2); E = copy(E2);  B = copy(B2); C = copy(C2);

@time Q, Z, νr, nc, nfu, niu  = sklf_right!(A,E,B,C,atol1=1.e-7,atol2=1.e-7,fast = fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Z-C) < sqrt(eps(1.))) && 
      νr == [3] && nc == 3 && nfu == 4 && niu == 0

end
end
end


@testset "sklf_left! - standard case" begin
A2 = zeros(0,0); C2 = zeros(0,0); B2 = missing;
A = copy(A2); C = copy(C2); B = missing;

@time Q, μl, no, nu  = sklf_left!(A,C,B)
@test norm(Q'*A2*Q-A) < sqrt(eps(1.)) &&
      norm(C2*Q-C) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      μl == [] && no == 0 && nu == 0


A2 = zeros(0,0); C2 = zeros(3,0); B2 = missing;
A = copy(A2); C = copy(C2); B = missing;

@time Q, μl, no, nu  = sklf_left!(A,C,B)
@test norm(Q'*A2*Q-A) < sqrt(eps(1.)) &&
      norm(C2*Q-C) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      μl == [0] && no == 0 && nu == 0

A2 = zeros(3,3); C2 = zeros(0,3); B2 = rand(3,2);
A = copy(A2); C = copy(C2); B = copy(B2);

@time Q, μl, no, nu  = sklf_left!(A,C,B)
@test norm(Q'*A2*Q-A) < sqrt(eps(1.)) &&
      norm(C2*Q-C) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      μl == [] && no == 0 && nu == 3

A2 = rand(3,3); C2 = rand(1,3); B2 = rand(3,2);
A = copy(A2); C = copy(C2); B = copy(B2);

@time Q, μl, no, nu  = sklf_left!(A,C,B)
@test norm(Q'*A2*Q-A) < sqrt(eps(1.)) &&
      norm(C2*Q-C) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      μl == [1, 1, 1] && no == 3 && nu == 0

Ty = Complex{Float64}
for Ty in (Float64, Complex{Float64})

A2 = rand(Ty,3,3); C2 = rand(Ty,1,3); B2 = rand(Ty,3,2);
A = copy(A2); C = copy(C2); B = copy(B2);

@time Q, μl, no, nu  = sklf_left!(A,C,B,fast = false)
@test norm(Q'*A2*Q-A) < sqrt(eps(1.)) &&
      norm(C2*Q-C) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      μl == [1, 1, 1] && no == 3 && nu == 0

A2 = rand(Ty,3,3); C2 = rand(Ty,2,3); B2 = rand(Ty,3,2);
A = copy(A2); C = copy(C2); B = copy(B2);

@time Q, μl, no, nu  = sklf_left!(A,C,B)
@test norm(Q'*A2*Q-A) < sqrt(eps(1.)) &&
      norm(C2*Q-C) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      μl == [1, 2] && no == 3 && nu == 0


A2 = rand(Ty,3,3); C2 = rand(Ty,2,3); B2 = rand(Ty,3,2);
A = copy(A2); C = copy(C2); B = copy(B2);

@time Q, μl, no, nu  = sklf_left!(A,C,B,fast = false)
@test norm(Q'*A2*Q-A) < sqrt(eps(1.)) &&
      norm(C2*Q-C) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      μl == [1, 2] && no == 3 && nu == 0

A2 = rand(Ty,3,3); C2 = rand(Ty,4,3); B2 = rand(Ty,3,2);
A = copy(A2); C = copy(C2); B = copy(B2);

@time Q, μl, no, nu  = sklf_left!(A,C,B)
@test norm(Q'*A2*Q-A) < sqrt(eps(1.)) &&
      norm(C2*Q-C) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      μl == [3] && no == 3 && nu == 0

A2 = rand(Ty,3,3); C2 = rand(Ty,4,3); B2 = rand(Ty,3,2);
A = copy(A2); C = copy(C2); B = copy(B2);

@time Q, μl, no, nu  = sklf_left!(A,C,B,fast = false)
@test norm(Q'*A2*Q-A) < sqrt(eps(1.)) &&
      norm(C2*Q-C) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      μl == [3] && no == 3 && nu == 0

A2 = [rand(Ty,3,7) ; zeros(Ty,4,3) rand(Ty,4,4) ]; B2 = rand(Ty,7,4); C2 = [ zeros(Ty,2,3) rand(Ty,2,4)];
A = copy(A2); B = copy(B2); C = copy(C2);

@time Q, μl, no, nu  = sklf_left!(A,C,B)
@test norm(Q'*A2*Q-A) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Q-C) < sqrt(eps(1.))) && 
      μl == [2, 2] && no == 4 && nu == 3

A2 = [rand(Ty,3,7) ; zeros(Ty,4,3) rand(Ty,4,4) ]; B2 = rand(Ty,7,4); C2 = [ zeros(Ty,2,3) rand(Ty,2,4)];
A = copy(A2); B = copy(B2); C = copy(C2);

@time Q, μl, no, nu  = sklf_left!(A,C,B,fast = false)
@test norm(Q'*A2*Q-A) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Q-C) < sqrt(eps(1.))) && 
      μl == [2, 2] && no == 4 && nu == 3
end
end

@testset "sklf_right! - standard case" begin

A2 = zeros(0,0); B2 = zeros(0,0); C2 = missing;
A = copy(A2); B = copy(B2); C = missing;

@time Q, νr, nc, nu  = sklf_right!(A,B,C)
@test norm(Q'*A2*Q-A) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Q-C) < sqrt(eps(1.))) && 
      νr == [] && nc == 0 && nu == 0

A2 = zeros(0,0); B2 = zeros(0,3);C2 = missing;
A = copy(A2); B = copy(B2); C = missing;

@time Q, νr, nc, nu  = sklf_right!(A,B,missing)
@test norm(Q'*A2*Q-A) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Q-C) < sqrt(eps(1.))) && 
      νr == [0] && nc == 0 && nu == 0

A2 = zeros(3,3); B2 = zeros(3,0); C2 = rand(2,3);
A = copy(A2); B = copy(B2); C = copy(C2);

@time Q, νr, nc, nu  = sklf_right!(A,B,C)
@test norm(Q'*A2*Q-A) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Q-C) < sqrt(eps(1.))) && 
      νr == [] && nc == 0 && nu == 3

Ty = Complex{Float64}
for Ty in (Float64, Complex{Float64})

A2 = rand(Ty,3,3); B2 = rand(Ty,3,1); C2 = rand(Ty,2,3);
A = copy(A2); B = copy(B2); C = copy(C2);

@time Q, νr, nc, nu  = sklf_right!(A,B,C)
@test norm(Q'*A2*Q-A) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Q-C) < sqrt(eps(1.))) && 
      νr == [1, 1, 1] && nc == 3 && nu == 0

A2 = rand(Ty,3,3); B2 = rand(Ty,3,1); C2 = rand(Ty,2,3);
A = copy(A2); B = copy(B2); C = copy(C2);

@time Q, νr, nc, nu  = sklf_right!(A,B,C,fast = false)
@test norm(Q'*A2*Q-A) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Q-C) < sqrt(eps(1.))) && 
      νr == [1, 1, 1] && nc == 3 && nu == 0

A2 = rand(Ty,3,3); B2 = rand(Ty,3,2); C2 = rand(Ty,2,3);
A = copy(A2); B = copy(B2); C = copy(C2);

@time Q, νr, nc, nu  = sklf_right!(A,B,C)
@test norm(Q'*A2*Q-A) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Q-C) < sqrt(eps(1.))) && 
      νr == [2, 1] && nc == 3 && nu == 0

A2 = rand(Ty,3,3); B2 = rand(Ty,3,2); C2 = rand(Ty,2,3);
A = copy(A2); B = copy(B2); C = copy(C2);

@time Q, νr, nc, nu  = sklf_right!(A,B,C,fast = false)
@test norm(Q'*A2*Q-A) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Q-C) < sqrt(eps(1.))) && 
      νr == [2, 1] && nc == 3 && nu == 0


A2 = rand(Ty,3,3); B2 = rand(Ty,3,4); C2 = rand(Ty,2,3);
A = copy(A2); B = copy(B2); C = copy(C2);

@time Q, νr, nc, nu  = sklf_right!(A,B,C)
@test norm(Q'*A2*Q-A) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Q-C) < sqrt(eps(1.))) && 
      νr == [3] && nc == 3 && nu == 0

A2 = rand(3,3); B2 = rand(3,4); C2 = rand(2,3);
A = copy(A2); B = copy(B2); C = copy(C2);

@time Q, νr, nc, nu  = sklf_right!(A,B,C,fast = false)
@test norm(Q'*A2*Q-A) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Q-C) < sqrt(eps(1.))) && 
      νr == [3] && nc == 3 && nu == 0


A2 = [rand(Ty,3,7) ; zeros(Ty,4,3) rand(Ty,4,4) ]; B2 = [rand(Ty,3,4); zeros(Ty,4,4)]; C2 = rand(Ty,2,7);
A = copy(A2); B = copy(B2); C = copy(C2);

@time Q, νr, nc, nu  = sklf_right!(A,B,C)
@test norm(Q'*A2*Q-A) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Q-C) < sqrt(eps(1.))) && 
      νr == [3] && nc == 3 && nu == 4

A2 = [rand(Ty,3,7) ; zeros(Ty,4,3) rand(Ty,4,4) ]; B2 = [rand(Ty,3,1); zeros(Ty,4,1)]; C2 = rand(Ty,2,7);
A = copy(A2); B = copy(B2); C = copy(C2);

@time Q, νr, nc, nu  = sklf_right!(A,B,C)
@test norm(Q'*A2*Q-A) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Q-C) < sqrt(eps(1.))) && 
      νr == [1, 1, 1] && nc == 3 && nu == 4
end
end

end
end