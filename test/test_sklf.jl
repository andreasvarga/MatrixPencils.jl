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

@time M, N, Q, Z, n, m, p = sreduceBF(missing,missing,missing,missing,missing,fast = fast)
@test n == 0 && m == 0 && p == 0 

    
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

M2 = [A' B; C D]; N2 = [E zeros(6,2); zeros(3,8)];  

M = copy(M2); N = copy(N2);
M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl = sklf(A', E, B, C, D, fast = fast, finite_infinite = false, atol1 = 1.e-7, atol2 = 1.e-7)
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

for fast in (true, false)

A2 = zeros(0,0); E2 = zeros(0,0); C2 = zeros(0,0); B2 = missing;
A = copy(A2); E = copy(E2); C = copy(C2); B = missing;

@time Q, Z, μl, no, nfuo, niuo  = sklf_left!(A,E,C,B, fast = fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(C2*Z-C) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      μl == [] && no == 0 && nfuo == 0 && niuo == 0


A2 = zeros(0,0); E2 = zeros(0,0); C2 = zeros(3,0); B2 = missing;
A = copy(A2); E = copy(E2); C = copy(C2); B = missing;

@time Q, Z, μl, no, nfuo, niuo  = sklf_left!(A,E,C,B, fast = fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(C2*Z-C) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      μl == [0] && no == 0 && nfuo == 0 && niuo == 0

A2 = rand(3,3); E2 = zeros(3,3); C2 = zeros(0,3); B2 = rand(3,2);
A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2);

@time Q, Z, μl, no, nfuo, niuo  = sklf_left!(A,E,C,B, fast = fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(C2*Z-C) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      μl == [] && no == 0 && nfuo == 0 && niuo == 3

A2 = rand(1,1); E2 = rand(1,1); C2 = rand(1,1); B2 = rand(1,2);
A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2);

@time Q, Z, μl, no, nfuo, niuo  = sklf_left!(A,E,C,B, fast = fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(C2*Z-C) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      μl == [1] && no == 1 && nfuo == 0 && niuo == 0

A2 = rand(1,1); E2 = rand(1,1); C2 = zeros(1,1); B2 = rand(1,2);
A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2);

@time Q, Z, μl, no, nfuo, niuo  = sklf_left!(A,E,C,B, fast = fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(C2*Z-C) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      μl == [] && no == 0 && nfuo == 1 && niuo == 0

# Ty = Complex{Float64}; fast = true
# Ty = Float64; fast = true

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

A2 = rand(Ty,3,3); E2 = rand(Ty,3,3); C2 = [rand(Ty,4,2) zeros(Ty,4,1)]; B2 = rand(Ty,3,2);
A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2);

@time Q, Z, μl, no, nfuo, niuo  = sklf_left!(A,E,C,B,fast=fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(C2*Z-C) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      μl == [1, 2] && no == 3 && nfuo == 0 && niuo == 0


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

for fast in (true, false)

A2 = zeros(0,0); E2 = zeros(0,0); B2 = zeros(0,0); C2 = missing;
A = copy(A2); E = copy(E2); B = copy(B2); C = missing;

@time Q, Z, νr, nc, nfu, niu  = sklf_right!(A,E,B,C,fast=fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Z-C) < sqrt(eps(1.))) && 
      νr == [] && nc == 0 && nfu == 0 && niu == 0

A2 = zeros(0,0); E2 = zeros(0,0); B2 = zeros(0,3); C2 = missing;
A = copy(A2); E = copy(E2); B = copy(B2); C = missing;

@time Q, Z, νr, nc, nfu, niu  = sklf_right!(A,E,B,C,fast=fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Z-C) < sqrt(eps(1.))) && 
      νr == [0] && nc == 0 && nfu == 0 && niu == 0

A2 = zeros(3,3); E2 = rand(3,3); B2 = zeros(3,0); C2 = rand(2,3);
A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2);

@time Q, Z, νr, nc, nfu, niu  = sklf_right!(A,E,B,C,fast=fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Z-C) < sqrt(eps(1.))) && 
      νr == [] && nc == 0 && nfu == 3 && niu == 0

A2 = rand(3,3); E2 = zeros(3,3); B2 = zeros(3,0); C2 = rand(2,3);
A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2);

@time Q, Z, νr, nc, nfu, niu  = sklf_right!(A,E,B,C,fast=fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Z-C) < sqrt(eps(1.))) && 
      νr == [] && nc == 0 && nfu == 0 && niu == 3

A2 = rand(3,3); E2 = rand(3,3); B2 = zeros(3,1); C2 = rand(2,3);
A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2);

@time Q, Z, νr, nc, nfu, niu  = sklf_right!(A,E,B,C,fast=fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Z-C) < sqrt(eps(1.))) && 
      νr == [] && nc == 0 && nfu == 3 && niu == 0

A2 = rand(3,3); E2 = rand(3,3); B2 = rand(3,1); C2 = rand(2,3);
A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2);

@time Q, Z, νr, nc, nfu, niu  = sklf_right!(A,E,B,C,fast=fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Z-C) < sqrt(eps(1.))) && 
      νr == [1, 1, 1] && nc == 3 && nfu == 0 && niu == 0

# Ty = Complex{Float64}
# fast = true; Ty = Float64

for Ty in (Float64, Complex{Float64})

A2 = rand(Ty,1,1); E2 = triu(rand(Ty,1,1),1); B2 = rand(Ty,1,1); C2 = rand(Ty,2,1);
A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2); 

@time Q, Z, νr, nc, nfu, niu  = sklf_right!(A,E,B,C,fast = fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Z-C) < sqrt(eps(1.))) && 
      νr == [1] && nc == 1 && nfu == 0 && niu == 0

A2 = rand(Ty,1,1); E2 = triu(rand(Ty,1,1),1); B2 = zeros(Ty,1,1); C2 = rand(Ty,2,1);
A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2); 

@time Q, Z, νr, nc, nfu, niu  = sklf_right!(A,E,B,C,fast = fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Z-C) < sqrt(eps(1.))) && 
      νr == [] && nc == 0 && nfu == 0 && niu == 1

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

A2 = rand(Ty,3,3); E2 = rand(Ty,3,3); B2 = [zeros(Ty,1,4); rand(Ty,2,4)]; C2 = rand(Ty,2,3);
A = copy(A2); E = copy(E2);  B = copy(B2); C = copy(C2);

@time Q, Z, νr, nc, nfu, niu  = sklf_right!(A,E,B,C,fast = fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Z-C) < sqrt(eps(1.))) && 
      νr == [2, 1]  && nc == 3 && nfu == 0 && niu == 0

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

@testset "sklf_rightfin! - generalized case" begin

for fast in (true, false)

A2 = zeros(0,0); E2 = zeros(0,0); B2 = zeros(0,0); C2 = missing;
A = copy(A2); E = copy(E2); B = copy(B2); C = missing;

@time Q, Z, νr, nc, nuc  = sklf_rightfin!(A,E,B,C,fast=fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Z-C) < sqrt(eps(1.))) && 
      νr == [] && nc == 0 && nuc == 0 

A2 = zeros(0,0); E2 = zeros(0,0); B2 = zeros(0,3); C2 = missing;
A = copy(A2); E = copy(E2); B = copy(B2); C = missing;

@time Q, Z, νr, nc, nuc  = sklf_rightfin!(A,E,B,C,fast=fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Z-C) < sqrt(eps(1.))) && 
      νr == [0] && nc == 0 && nuc == 0 

A2 = zeros(3,3); E2 = rand(3,3); B2 = zeros(3,0); C2 = rand(2,3);
A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2);

@time Q, Z, νr, nc, nuc  = sklf_rightfin!(A,E,B,C,fast=fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Z-C) < sqrt(eps(1.))) && 
      νr == [] && nc == 0 && nuc == 3 

A2 = rand(3,3); E2 = zeros(3,3); B2 = zeros(3,0); C2 = rand(2,3);
A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2);

@time Q, Z, νr, nc, nuc  = sklf_rightfin!(A,E,B,C,fast=fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Z-C) < sqrt(eps(1.))) && 
      νr == [] && nc == 0 && nuc == 3 

A2 = rand(3,3); E2 = rand(3,3); B2 = zeros(3,1); C2 = rand(2,3);
A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2);

@time Q, Z, νr, nc, nuc  = sklf_rightfin!(A,E,B,C,fast=fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Z-C) < sqrt(eps(1.))) && 
      νr == [] && nc == 0 && nuc == 3 

A2 = rand(3,3); E2 = rand(3,3); B2 = rand(3,1); C2 = rand(2,3);
A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2);

@time Q, Z, νr, nc, nuc  = sklf_rightfin!(A,E,B,C,fast=fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Z-C) < sqrt(eps(1.))) && 
      νr == [1, 1, 1] && nc == 3 && nuc == 0 

# Ty = Complex{Float64}
# fast = true; Ty = Float64

for Ty in (Float64, Complex{Float64})

A2 = rand(Ty,1,1); E2 = triu(rand(Ty,1,1),1); B2 = rand(Ty,1,1); C2 = rand(Ty,2,1);
A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2); 

@time Q, Z, νr, nc, nuc  = sklf_rightfin!(A,E,B,C,fast = fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Z-C) < sqrt(eps(1.))) && 
      νr == [1] && nc == 1 && nuc == 0

A2 = rand(Ty,1,1); E2 = triu(rand(Ty,1,1),1); B2 = zeros(Ty,1,1); C2 = rand(Ty,2,1);
A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2); 

@time Q, Z, νr, nc, nuc  = sklf_rightfin!(A,E,B,C,fast = fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Z-C) < sqrt(eps(1.))) && 
      νr == [] && nc == 0 && nuc == 1

A2 = rand(Ty,3,3); E2 = triu(rand(Ty,3,3),1); B2 = rand(Ty,3,1); C2 = rand(Ty,2,3);
A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2); 

@time Q, Z, νr, nc, nuc  = sklf_rightfin!(A,E,B,C,fast = fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Z-C) < sqrt(eps(1.))) && 
      νr == [1, 1, 1] && nc == 3 && nuc == 0

A2 = rand(Ty,3,3); E2 = triu(rand(Ty,3,3),1); B2 = rand(Ty,3,2); C2 = rand(Ty,2,3);
A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2);

@time Q, Z, νr, nc, nuc  = sklf_rightfin!(A,E,B,C,fast = fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Z-C) < sqrt(eps(1.))) && 
      νr == [2, 1] && nc == 3 && nuc == 0

A2 = rand(Ty,3,3); E2 = rand(Ty,3,3); B2 = rand(Ty,3,4); C2 = rand(Ty,2,3);
A = copy(A2); E = copy(E2);  B = copy(B2); C = copy(C2);

@time Q, Z, νr, nc, nuc  = sklf_rightfin!(A,E,B,C,fast = fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Z-C) < sqrt(eps(1.))) && 
      νr == [3] && nc == 3 && nuc == 0

A2 = rand(Ty,3,3); E2 = rand(Ty,3,3); B2 = [zeros(Ty,1,4); rand(Ty,2,4)]; C2 = rand(Ty,2,3);
A = copy(A2); E = copy(E2);  B = copy(B2); C = copy(C2);

@time Q, Z, νr, nc, nuc  = sklf_rightfin!(A,E,B,C,fast = fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Z-C) < sqrt(eps(1.))) && 
      νr == [2, 1]  && nc == 3 && nuc == 0

A2 = [rand(Ty,3,7) ; zeros(Ty,4,3) rand(Ty,4,4) ]; 
E2 = [rand(Ty,3,7) ; zeros(Ty,4,3) triu(rand(Ty,4,4),1) ]; 
B2 = [rand(Ty,3,4); zeros(Ty,4,4)]; C2 = rand(Ty,2,7);
A = copy(A2); E = copy(E2);  B = copy(B2); C = copy(C2);

@time Q, Z, νr, nc, nuc  = sklf_rightfin!(A,E,B,C,atol1=1.e-7,atol2=1.e-7,fast = fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Z-C) < sqrt(eps(1.))) && 
      νr == [3] && nc == 3 && nuc == 4 

A2 = [rand(Ty,3,7) ; zeros(Ty,4,3) rand(Ty,4,4) ]; 
E2 = [rand(Ty,3,7) ; zeros(Ty,4,3) rand(Ty,4,4) ]; 
B2 = [rand(Ty,3,4); zeros(Ty,4,4)]; C2 = rand(Ty,2,7);
A = copy(A2); E = copy(E2);  B = copy(B2); C = copy(C2);

@time Q, Z, νr, nc, nuc  = sklf_rightfin!(A,E,B,C,atol1=1.e-7,atol2=1.e-7,fast = fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Z-C) < sqrt(eps(1.))) && 
      νr == [3] && nc == 3 && nuc == 4 

end
end
end

@testset "sklf_leftfin! - generalized case" begin

for fast in (true, false)

Ty = Float64; fast = false
A2 = zeros(0,0); E2 = zeros(0,0); C2 = zeros(0,0); B2 = missing;
A = copy(A2); E = copy(E2); C = copy(C2); B = missing;

@time Q, Z, μl, no, nuo  = sklf_leftfin!(A,E,C,B, fast = fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(C2*Z-C) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      μl == [] && no == 0 && nuo == 0


A2 = zeros(0,0); E2 = zeros(0,0); C2 = zeros(3,0); B2 = missing;
A = copy(A2); E = copy(E2); C = copy(C2); B = missing;

@time Q, Z, μl, no, nuo  = sklf_leftfin!(A,E,C,B, fast = fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(C2*Z-C) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      μl == [0] && no == 0 && nuo  == 0

A2 = rand(3,3); E2 = zeros(3,3); C2 = zeros(0,3); B2 = rand(3,2);
A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2);

@time Q, Z, μl, no, nuo  = sklf_leftfin!(A,E,C,B, fast = fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(C2*Z-C) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      μl == [] && no == 0 && nuo == 3

A2 = rand(1,1); E2 = rand(1,1); C2 = rand(1,1); B2 = rand(1,2);
A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2);

@time Q, Z, μl, no, nuo  = sklf_leftfin!(A,E,C,B, fast = fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(C2*Z-C) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      μl == [1] && no == 1 && nuo  == 0

A2 = rand(1,1); E2 = rand(1,1); C2 = zeros(1,1); B2 = rand(1,2);
A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2);

@time Q, Z, μl, no, nuo  = sklf_leftfin!(A,E,C,B, fast = fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(C2*Z-C) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      μl == [] && no == 0 && nuo == 1

# Ty = Complex{Float64}; fast = true
# Ty = Float64; fast = true

for Ty in (Float64, Complex{Float64})

A2 = rand(Ty,3,3); E2 = rand(Ty,3,3); C2 = rand(Ty,1,3); B2 = rand(Ty,3,2);
A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2);

@time Q, Z, μl, no, nuo  = sklf_leftfin!(A,E,C,B,fast=fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(C2*Z-C) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      μl == [1, 1, 1] && no == 3 && nuo  == 0

A2 = rand(Ty,3,3); E2 = rand(Ty,3,3); C2 = rand(Ty,2,3); B2 = rand(Ty,3,2);
A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2);

@time Q, Z, μl, no, nuo  = sklf_leftfin!(A,E,C,B,fast=fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(C2*Z-C) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      μl == [1, 2] && no == 3 && nuo  == 0


A2 = rand(Ty,3,3); E2 = rand(Ty,3,3); C2 = rand(Ty,4,3); B2 = rand(Ty,3,2);
A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2);

@time Q, Z, μl, no, nuo  = sklf_leftfin!(A,E,C,B,fast=fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(C2*Z-C) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      μl == [3] && no == 3 && nuo  == 0

A2 = rand(Ty,3,3); E2 = rand(Ty,3,3); C2 = [rand(Ty,4,2) zeros(Ty,4,1)]; B2 = rand(Ty,3,2);
A = copy(A2); E = copy(E2); C = copy(C2); B = copy(B2);

@time Q, Z, μl, no, nuo  = sklf_leftfin!(A,E,C,B,fast=fast)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(C2*Z-C) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      μl == [1, 2] && no == 3 && nuo  == 0


A2 = [rand(Ty,3,7) ; zeros(Ty,4,3) rand(Ty,4,4) ]; 
E2 = [rand(Ty,3,7) ; zeros(Ty,4,3) rand(Ty,4,4) ]; 
B2 = rand(Ty,7,4); C2 = [ zeros(Ty,2,3) rand(Ty,2,4)];
A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2);

@time Q, Z, μl, no, nuo  = sklf_leftfin!(A,E,C,B,fast=fast,atol1=1.e-7,atol2=1.e-7)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(C2*Z-C) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      μl == [2, 2] && no == 4 && nuo == 3 

A2 = [rand(Ty,3,3) zeros(Ty,3,4) ; zeros(Ty,4,3) rand(Ty,4,4) ]; 
E2 = [rand(Ty,3,3) zeros(Ty,3,4); rand(Ty,4,3) triu(rand(Ty,4,4),1) ]; 
B2 = rand(Ty,7,4); C2 = [ rand(Ty,2,3) zeros(Ty,2,4)];
A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2);

@time Q, Z, μl, no, nuo  = sklf_leftfin!(A,E,C,B,fast=fast,atol1=1.e-7,atol2=1.e-7)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(C2*Z-C) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      μl == [2, 2] && no == 4 && nuo == 3 

A2 = [rand(Ty,3,3) zeros(Ty,3,4) ; zeros(Ty,4,3) triu(rand(Ty,4,4)) ]; 
E2 = [rand(Ty,3,3) zeros(Ty,3,4); rand(Ty,4,3) triu(rand(Ty,4,4),1) ]; 
B2 = rand(Ty,7,4); C2 = [ rand(Ty,2,3) zeros(Ty,2,4)];
A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2);

@time Q, Z, μl, no, nuo  = sklf_leftfin!(E,A,C,B,fast=fast,atol1=1.e-7,atol2=1.e-7)
@test norm(Q'*A2*Z-A) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E) < sqrt(eps(1.)) &&
      norm(C2*Z-C) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      μl == [1, 2] && no == 3 && nuo == 4

end
end
end


@testset "sklf_left! - standard case" begin

for fast in (true, false)

A2 = zeros(0,0); C2 = zeros(0,0); B2 = missing;
A = copy(A2); C = copy(C2); B = missing;

@time Q, μl, no, nu  = sklf_left!(A,C,B,fast = fast)
@test norm(Q'*A2*Q-A) < sqrt(eps(1.)) &&
      norm(C2*Q-C) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      μl == [] && no == 0 && nu == 0


A2 = zeros(0,0); C2 = zeros(3,0); B2 = missing;
A = copy(A2); C = copy(C2); B = missing;

@time Q, μl, no, nu  = sklf_left!(A,C,B,fast = fast)
@test norm(Q'*A2*Q-A) < sqrt(eps(1.)) &&
      norm(C2*Q-C) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      μl == [0] && no == 0 && nu == 0

A2 = zeros(3,3); C2 = zeros(0,3); B2 = rand(3,2);
A = copy(A2); C = copy(C2); B = copy(B2);

@time Q, μl, no, nu  = sklf_left!(A,C,B,fast = fast)
@test norm(Q'*A2*Q-A) < sqrt(eps(1.)) &&
      norm(C2*Q-C) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      μl == [] && no == 0 && nu == 3


Ty = Complex{Float64}
for Ty in (Float64, Complex{Float64})

A2 = rand(Ty,3,3); C2 = rand(Ty,1,3); B2 = rand(Ty,3,2);
A = copy(A2); C = copy(C2); B = copy(B2);

@time Q, μl, no, nu  = sklf_left!(A,C,B,fast = fast)
@test norm(Q'*A2*Q-A) < sqrt(eps(1.)) &&
      norm(C2*Q-C) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      μl == [1, 1, 1] && no == 3 && nu == 0

A2 = rand(Ty,3,3); C2 = rand(Ty,2,3); B2 = rand(Ty,3,2);
A = copy(A2); C = copy(C2); B = copy(B2);

@time Q, μl, no, nu  = sklf_left!(A,C,B,fast = fast)
@test norm(Q'*A2*Q-A) < sqrt(eps(1.)) &&
      norm(C2*Q-C) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      μl == [1, 2] && no == 3 && nu == 0

A2 = rand(Ty,3,3); C2 = rand(Ty,4,3); B2 = rand(Ty,3,2);
A = copy(A2); C = copy(C2); B = copy(B2);

@time Q, μl, no, nu  = sklf_left!(A,C,B,fast = fast)
@test norm(Q'*A2*Q-A) < sqrt(eps(1.)) &&
      norm(C2*Q-C) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      μl == [3] && no == 3 && nu == 0

A2 = rand(Ty,3,3); C2 = [rand(Ty,4,2) zeros(Ty,4,1)]; B2 = rand(Ty,3,2);
A = copy(A2); C = copy(C2); B = copy(B2);

@time Q, μl, no, nu  = sklf_left!(A,C,B,fast = fast)
@test norm(Q'*A2*Q-A) < sqrt(eps(1.)) &&
      norm(C2*Q-C) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B) < sqrt(eps(1.))) && 
      μl == [1, 2] && no == 3 && nu == 0


A2 = [rand(Ty,3,7) ; zeros(Ty,4,3) rand(Ty,4,4) ]; B2 = rand(Ty,7,4); C2 = [ zeros(Ty,2,3) rand(Ty,2,4)];
A = copy(A2); B = copy(B2); C = copy(C2);

@time Q, μl, no, nu  = sklf_left!(A,C,B,fast = fast)
@test norm(Q'*A2*Q-A) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Q-C) < sqrt(eps(1.))) && 
      μl == [2, 2] && no == 4 && nu == 3

end
end
end

@testset "sklf_right! - standard case" begin

for fast in (true, false)

A2 = zeros(0,0); B2 = zeros(0,0); C2 = missing;
A = copy(A2); B = copy(B2); C = missing;

@time Q, νr, nc, nu  = sklf_right!(A,B,C,fast = fast)
@test norm(Q'*A2*Q-A) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Q-C) < sqrt(eps(1.))) && 
      νr == [] && nc == 0 && nu == 0

A2 = zeros(0,0); B2 = zeros(0,3);C2 = missing;
A = copy(A2); B = copy(B2); C = missing;

@time Q, νr, nc, nu  = sklf_right!(A,B,missing,fast = fast)
@test norm(Q'*A2*Q-A) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Q-C) < sqrt(eps(1.))) && 
      νr == [0] && nc == 0 && nu == 0

A2 = zeros(3,3); B2 = zeros(3,0); C2 = rand(2,3);
A = copy(A2); B = copy(B2); C = copy(C2);

@time Q, νr, nc, nu  = sklf_right!(A,B,C,fast = fast)
@test norm(Q'*A2*Q-A) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Q-C) < sqrt(eps(1.))) && 
      νr == [] && nc == 0 && nu == 3

Ty = Complex{Float64}
for Ty in (Float64, Complex{Float64})

A2 = rand(Ty,3,3); B2 = rand(Ty,3,1); C2 = rand(Ty,2,3);
A = copy(A2); B = copy(B2); C = copy(C2);

@time Q, νr, nc, nu  = sklf_right!(A,B,C,fast = fast)
@test norm(Q'*A2*Q-A) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Q-C) < sqrt(eps(1.))) && 
      νr == [1, 1, 1] && nc == 3 && nu == 0

A2 = rand(Ty,3,3); B2 = rand(Ty,3,2); C2 = rand(Ty,2,3);
A = copy(A2); B = copy(B2); C = copy(C2);

@time Q, νr, nc, nu  = sklf_right!(A,B,C,fast = fast)
@test norm(Q'*A2*Q-A) < sqrt(eps(1.)) &&
      norm(Q'*B2-B) < sqrt(eps(1.)) &&
      (ismissing(C) || norm(C2*Q-C) < sqrt(eps(1.))) && 
      νr == [2, 1] && nc == 3 && nu == 0

A2 = rand(Ty,3,3); B2 = rand(Ty,3,4); C2 = rand(Ty,2,3);
A = copy(A2); B = copy(B2); C = copy(C2);

@time Q, νr, nc, nu  = sklf_right!(A,B,C,fast = fast)
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


@testset "sklf_right! and sklf_left! - pencil case" begin

fast = true; Ty = Float64; Ty = Complex{Float64}     
for fast in (true, false)

# ensuring strong observability
# Example 4: Van Dooren, Dewilde, LAA 1983
# P = zeros(3,3,3)
# P[:,:,1] = [1 2 -2; 0 -1 -2; 0 0 0]
# P[:,:,2] = [1 3 0; 1 4 2; 0 -1 -2]
# P[:,:,3] = [1 4 2; 0 0 0; 1 4 2]

# use a strongly controllable pencil realization

sys1  = ([1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0], 
[0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0], 
[0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0], 
[1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0], 
[1.0 3.0 0.0; 1.0 4.0 2.0; 0.0 -1.0 -2.0], 
[-1.0 -4.0 -2.0; -0.0 -0.0 -0.0; -1.0 -4.0 -2.0], 
[1.0 2.0 -2.0; 0.0 -1.0 -2.0; 0.0 0.0 0.0], 
[0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0])

# the initial realization is already strongly controllable
(A, E, B, F, C, G, D, H) = copy.(sys1)
Q, Z, nc = sklf_right!(A, E, B, F, C, G, D, H, fast = fast) 
i1 = 1:nc
W = Z[1:3,1:3]
@test Q'*[sys1[3] sys1[1]]*Z ≈ [B A] &&
      Q'*[sys1[4] sys1[2]]*Z ≈ [F E] &&
      [sys1[8] sys1[6]]*Z ≈ [H G] &&
      [sys1[7] sys1[5]]*Z ≈ [D C] && nc == size(A,1) &&
      lpsequal(A[i1,i1], E[i1,i1], B[i1,:]/W, F[i1,:]/W, C[:,i1], G[:,i1], D/W, H/W, sys1...,atol1=1.e-7,atol2=1.e-7)  


# determine a strongly controllable and strongly observable realization
(A, E, B, F, C, G, D, H) = copy.(sys1)
Q, Z, no = sklf_left!(A, E, C, G, B, F, D, H, fast = fast) 
p,n = size(C)
i1 = n-no+1:n
V = Q[end-p+1:end,end-p+1:end]'
@test Q'*[sys1[1]; sys1[5]]*Z ≈ [A;C] &&
      Q'*[sys1[2]; sys1[6]]*Z ≈ [E; G] &&
      Q'*[sys1[4]; sys1[8]] ≈ [F; H] &&
      Q'*[sys1[3]; sys1[7]] ≈ [B; D] &&
      lpsequal(A[i1,i1], E[i1,i1], B[i1,:], F[i1,:], V\C[:,i1], V\G[:,i1], V\D, V\H, sys1...,atol1=1.e-7,atol2=1.e-7)

# the eigenvalues and Kronecker structure of P(s) are the expected ones
Mo = [A[i1,i1] B[i1,:]; C[:,i1] D]
No = [E[i1,i1] F[i1,:]; G[:,i1] H]
val, info = peigvals(Mo,No)
@test val ≈ [1, Inf] && (info.rki, info.lki,info.id, info.nf) == ([0], [1], [1], 1)


# use a strongly observable pencil realizations

sys2  = ([1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0], 
[0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0], 
[1.0 3.0 0.0; 1.0 4.0 2.0; 0.0 -1.0 -2.0], 
[-1.0 -4.0 -2.0; -0.0 -0.0 -0.0; -1.0 -4.0 -2.0], 
[0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0], 
[1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0], 
[1.0 2.0 -2.0; 0.0 -1.0 -2.0; 0.0 0.0 0.0], 
[0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0])

# the initial realization is already strongly observable 
n = size(A,1)
(A, E, B, F, C, G, D, H) = copy.(sys2)
V, Z, no = sklf_left!(A, E, C, G, B, F, D, H, fast = fast, withQ = false, withZ = false) 
n = size(A,1)
i1 = n-no+1:n
@test n == no && lpsequal(A[i1,i1], E[i1,i1], B[i1,:], F[i1,:], V\C[:,i1], V\G[:,i1], V\D, V\H, sys2...,atol1=1.e-7,atol2=1.e-7)

# compute a strongly controllable realization
(A, E, B, F, C, G, D, H) = copy.(sys2)
Q, Z, nc = sklf_right!(A, E, B, F, C, G, D, H, fast = fast) 
i1 = 1:nc
W = Z[1:3,1:3]
@test Q'*[sys2[3] sys2[1]]*Z ≈ [B A] &&
      Q'*[sys2[4] sys2[2]]*Z ≈ [F E] &&
      [sys2[8] sys2[6]]*Z ≈ [H G] &&
      [sys2[7] sys2[5]]*Z ≈ [D C] &&
      lpsequal(A[i1,i1], E[i1,i1], B[i1,:]/W, F[i1,:]/W, C[:,i1], G[:,i1], D/W, H/W, sys2...,atol1=1.e-7,atol2=1.e-7)

# the eigenvalues and Kronecker structure of P(s) are the expected ones
Mc = [A[i1,i1] B[i1,:]; C[:,i1] D]
Nc = [E[i1,i1] F[i1,:]; G[:,i1] H]
val, info = peigvals(Mc,Nc)
@test val ≈ [1, Inf] && (info.rki, info.lki,info.id, info.nf) == ([0], [1], [1], 1)

# use a minimal descriptor realization (which is however not strongly minimal)
(A, E, B, C, D) = 
([0.2413793103448278 -1.2772592536173888 -0.18569533817705197; -0.04561640191490692 0.2413793103448277 -0.9826073688810348; -0.9826073688810347 -0.18569533817705186 0.0], 
[1.0 -4.5102810375397046e-17 0.0; 0.0 -1.0 0.0; 0.0 0.0 0.0], 
[0.9191450300180584 1.8382900600361163 -1.838290060036116; -0.17370208344491272 -0.34740416688982534 0.34740416688982545; -1.8708286933869704 -7.483314773547881 -3.741657386773941], 
[-0.0656532164298613 0.34740416688982556 0.7071067811865471; -0.1313064328597228 0.6948083337796513 4.163336342344337e-17; 0.06565321642986145 -0.3474041668898257 0.7071067811865477], 
[0.7500000000000001 1.5000000000000004 -1.5; -0.5 -1.9999999999999996 -0.9999999999999996; 0.24999999999999997 0.4999999999999998 -0.5000000000000001])
F = zeros(size(B)...)
G = zeros(size(C)...)
H = zeros(size(D)...)
sys = (A, E, B, F, C, G, D, H)
sys2 = copy.(sys)

# this realization is not strongly minimal
# the computed eigenvalues contains two spurious infinite eigenvalues
val, info = peigvals([A B;C D],[E F;G H])
@test val ≈ [1, Inf, Inf, Inf] && (info.rki, info.lki,info.id, info.nf) == ([0], [1], [1, 1, 1], 1)

# compute a strongly controllable pencil realization
#(A, E, B, F, C, G, D, H) = copy.(sys2)
Q, Z, nc = sklf_right!(A, E, B, F, C, G, D, H, fast = fast) 
i1 = 1:nc
W = Z[1:3,1:3]
@test Q'*[sys2[3] sys2[1]]*Z ≈ [B A] &&
      Q'*[sys2[4] sys2[2]]*Z ≈ [F E] &&
      [sys2[8] sys2[6]]*Z ≈ [H G] &&
      [sys2[7] sys2[5]]*Z ≈ [D C] &&
      lpsequal(A[i1,i1], E[i1,i1], B[i1,:]/W, F[i1,:]/W, C[:,i1], G[:,i1], D/W, H/W, sys2...,atol1=1.e-7,atol2=1.e-7)

# there is still a spurious infinite eigenvalue 
Mc = [A[i1,i1] B[i1,:]; C[:,i1] D]
Nc = [E[i1,i1] F[i1,:]; G[:,i1] H]
val, info = peigvals(Mc,Nc)
@test val ≈ [1, Inf, Inf] && (info.rki, info.lki,info.id, info.nf) == ([0], [1], [1, 1], 1)


# compute a strongly observable pencil realization
(A, E, B, F, C, G, D, H) = copy.(sys2)
Q, Z, no = sklf_left!(A, E, C, G, B, F, D, H, fast = fast) 
p, n = size(C)
i1 = n-no+1:n
V = Q[end-p+1:end,end-p+1:end]'
@test Q'*[sys2[1]; sys2[5]]*Z ≈ [A;C] &&
      Q'*[sys2[2]; sys2[6]]*Z ≈ [E; G] &&
      Q'*[sys2[4]; sys2[8]] ≈ [F; H] &&
      Q'*[sys2[3]; sys2[7]] ≈ [B; D] &&
      lpsequal(A[i1,i1], E[i1,i1], B[i1,:], F[i1,:], V\C[:,i1], V\G[:,i1], V\D, V\H, sys2...,atol1=1.e-7,atol2=1.e-7)

# there is still a spurious infinite eigenvalue 
Mo = [A[i1,i1] B[i1,:]; C[:,i1] D]
No = [E[i1,i1] F[i1,:]; G[:,i1] H]
val, info = peigvals(Mo,No)
@test val ≈ [1, Inf, Inf] && (info.rki, info.lki,info.id, info.nf) == ([0], [1], [1, 1], 1)


end
end

end
end