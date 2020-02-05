module Test_klf

using LinearAlgebra
using MatrixPencils
using Test


@testset "Matrix Pencils Utilities" begin


M2 = zeros(0,0); N2 = zeros(0,0);
M = copy(M2); N = copy(N2); 

@time M1, N1, Q, Z, ν, μ, n, m, p  = klf_rlsplit(M, N)
@test norm(Q'*M2*Z-M1) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N1) < sqrt(eps(1.)) &&
      ν == [] && μ == [] && n == 0 && m == 0  && p == 0

M2 = zeros(0,0); N2 = zeros(0,0);
M = copy(M2); N = copy(N2); 

@time Q, Z, ν, μ, nf, νl, μl  = klf_left!(M, N)
@test norm(Q'*M2*Z-M) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N) < sqrt(eps(1.)) &&
      ν == [] && μ == [] && νl == [] && μl == []  && nf == 0

M2 = zeros(0,0); N2 = zeros(0,0);
M = copy(M2); N = copy(N2);

@time Q, Z, νr, μr, nf, ν, μ  = klf_right!(M, N)
@test norm(Q'*M2*Z-M) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N) < sqrt(eps(1.)) &&
      νr == [] && μr == [] && ν == [] && μ == []  && nf == 0

M2 = zeros(0,0); N2 = zeros(0,0);
M = copy(M2); N = copy(N2);

@time M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl  = klf(M, N)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [] && μr == [] && νi == [] && νl == [] && μl == []  && nf == 0      

M2 = zeros(3,0); N2 = zeros(3,0);
M = copy(M2); N = copy(N2); 

@time M1, N1, Q, Z, ν, μ, n, m, p  = klf_rlsplit(M, N)
@test norm(Q'*M2*Z-M1) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N1) < sqrt(eps(1.)) &&
      ν == [] && μ == [] && n == 0 && m == 0  && p == 3

M2 = zeros(3,0); N2 = zeros(3,0);
M = copy(M2); N = copy(N2); 

@time M1, N1, Q, Z, ν, μ, n, m, p  = klf_rlsplit(M, N, finite_infinite = true)
@test norm(Q'*M2*Z-M1) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N1) < sqrt(eps(1.)) &&
      ν == [3] && μ == [0] && n == 0 && m == 0  && p == 0

M2 = zeros(3,0); N2 = zeros(3,0);
M = copy(M2); N = copy(N2);

@time Q, Z, ν, μ, nf, νl, μl = klf_left!(M, N)
@test norm(Q'*M2*Z-M) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N) < sqrt(eps(1.)) &&
      ν == [] && μ == [] && νl == [3] && μl == [0]  && nf == 0   

M2 = zeros(3,0); N2 = zeros(3,0);
M = copy(M2); N = copy(N2);
@time Q, Z, νr, μr, nf, ν, μ  = klf_right!(M, N)
@test norm(Q'*M2*Z-M) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N) < sqrt(eps(1.)) &&
      νr == [] && μr == [] && ν == [3] && μ == [0] && nf == 0


M2 = zeros(3,0); N2 = zeros(3,0);
M = copy(M2); N = copy(N2);

@time M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl  = klf(M, N)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [] && μr == [] && νi == [] && νl == [3] && μl == [0]  && nf == 0           

M2 = zeros(0,3); N2 = zeros(0,3);
M = copy(M2); N = copy(N2); 

@time M1, N1, Q, Z, ν, μ, n, m, p  = klf_rlsplit(M, N)
@test norm(Q'*M2*Z-M1) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N1) < sqrt(eps(1.)) &&
      ν == [0] && μ == [3] && n == 0 && m == 0  && p == 0

M2 = zeros(0,3); N2 = zeros(0,3);
M = copy(M2); N = copy(N2); 

@time M1, N1, Q, Z, ν, μ, n, m, p  = klf_rlsplit(M, N, finite_infinite = true)
@test norm(Q'*M2*Z-M1) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N1) < sqrt(eps(1.)) &&
      ν == [] && μ == [] && n == 0 && m == 3  && p == 0

M2 = zeros(0,3); N2 = zeros(0,3);
M = copy(M2); N = copy(N2);

@time Q, Z, ν, μ, nf, νl, μl = klf_left!(M, N)
@test norm(Q'*M2*Z-M) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N) < sqrt(eps(1.)) &&
      ν == [0] && μ == [3] && νl == [] && μl == [] && nf == 0


M2 = zeros(0,3); N2 = zeros(0,3);
M = copy(M2); N = copy(N2);
@time Q, Z, νr, μr, nf, ν, μ  = klf_right!(M, N)
@test norm(Q'*M2*Z-M) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N) < sqrt(eps(1.)) &&
      νr == [0] && μr == [3] && ν == [] && μ == [] && nf == 0


M2 = zeros(0,3); N2 = zeros(0,3); 
M = copy(M2); N = copy(N2);

@time M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl  = klf(M, N)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [0] && μr == [3] && νi == [] && νl == [] && μl == []  && nf == 0           

M2 = ones(1,1); N2 = zeros(1,1); 
M = copy(M2); N = copy(N2); 

@time M1, N1, Q, Z, ν, μ, n, m, p  = klf_rlsplit(M, N)
@test norm(Q'*M2*Z-M1) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N1) < sqrt(eps(1.)) &&
      ν == [1] && μ == [1] && n == 0 && m == 0  && p == 0

M2 = ones(1,1); N2 = zeros(1,1); 
M = copy(M2); N = copy(N2); 

@time M1, N1, Q, Z, ν, μ, n, m, p  = klf_rlsplit(M, N, finite_infinite = true)
@test norm(Q'*M2*Z-M1) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N1) < sqrt(eps(1.)) &&
      ν == [1] && μ == [1] && n == 0 && m == 0  && p == 0

M2 = ones(1,1); N2 = zeros(1,1); 
M = copy(M2); N = copy(N2);
@time Q, Z, ν, μ, nf, νl, μl = klf_left!(M, N, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q'*M2*Z-M) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N) < sqrt(eps(1.)) &&
      ν == [1] && μ == [1] && νl == [] && μl == [] && nf == 0      


M2 = ones(1,1); N2 = zeros(1,1); 
M = copy(M2); N = copy(N2);
@time Q, Z, νr, μr, nf, ν, μ  = klf_right!(M, N)
@test norm(Q'*M2*Z-M) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N) < sqrt(eps(1.)) &&
      νr == [] && μr == [] && ν == [1] && μ == [1] && nf == 0

M2 = ones(1,1); N2 = zeros(1,1); 
M = copy(M2); N = copy(N2);

@time M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl  = klf(M, N)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [] && μr == [] && νi == [1] && νl == [] && μl == []  && nf == 0           


M2 = zeros(1,1); N2 = ones(1,1); 
M = copy(M2); N = copy(N2); 

@time M1, N1, Q, Z, ν, μ, n, m, p  = klf_rlsplit(M, N)
@test norm(Q'*M2*Z-M1) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N1) < sqrt(eps(1.)) &&
      ν == [] && μ == [] && n == 1 && m == 0  && p == 0

M2 = zeros(1,1); N2 = ones(1,1); 
M = copy(M2); N = copy(N2); 

@time M1, N1, Q, Z, ν, μ, n, m, p  = klf_rlsplit(M, N, finite_infinite = true)
@test norm(Q'*M2*Z-M1) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N1) < sqrt(eps(1.)) &&
      ν == [] && μ == [] && n == 1 && m == 0  && p == 0

M2 = zeros(1,1); N2 = ones(1,1); 
M = copy(M2); N = copy(N2);
     
@time Q, Z, ν, μ, nf, νl, μl = klf_left!(M, N, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q'*M2*Z-M) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N) < sqrt(eps(1.)) &&
      ν == [ ] && μ == [ ]  && νl == [] && μl == [] && nf == 1    

M2 = zeros(1,1); N2 = ones(1,1); 
M = copy(M2); N = copy(N2);
@time Q, Z, νr, μr, nf, ν, μ  = klf_right!(M, N)
@test norm(Q'*M2*Z-M) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N) < sqrt(eps(1.)) &&
      νr == [] && μr == [] && ν == [] && μ == [] && nf == 1


M2 = zeros(1,1); N2 = ones(1,1);  
M = copy(M2); N = copy(N2);

@time M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl  = klf(M, N)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [] && μr == [] && νi == [] && νl == [] && μl == []  && nf == 1

N2 = zeros(2,1); M2 = [zeros(1,1); ones(1,1)];  
M = copy(M2); N = copy(N2);
@time Q, Z, ν, μ, nf, νl, μl = klf_left!(M, N, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q'*M2*Z-M) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N) < sqrt(eps(1.)) &&
      ν == [1] && μ == [1] && νl == [1] && μl == [0] && nf == 0    

N2 = zeros(2,1); M2 = [zeros(1,1); ones(1,1)];  
M = copy(M2); N = copy(N2);
@time Q, Z, νr, μr, nf, ν, μ  = klf_right!(M, N)
@test norm(Q'*M2*Z-M) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N) < sqrt(eps(1.)) &&
      νr == [] && μr == [] && ν == [2] && μ == [1] && nf == 0

N2 = zeros(2,1); M2 = [zeros(1,1); ones(1,1)];   
M = copy(M2); N = copy(N2);

@time M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl  = klf(M, N)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [] && μr == [] && νi == [1] && νl == [1] && μl == [0]  && nf == 0

M2 = zeros(1,1); N2 = zeros(1,1); 
M = copy(M2); N = copy(N2);
@time Q, Z, ν, μ, nf, νl, μl = klf_left!(M, N, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q'*M2*Z-M) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N) < sqrt(eps(1.)) &&
      ν == [0] && μ == [1]  && νl == [1] && μl == [0] && nf == 0   

M2 = zeros(1,1); N2 = zeros(1,1); 
M = copy(M2); N = copy(N2);
@time Q, Z, νr, μr, nf, ν, μ  = klf_right!(M, N)
@test norm(Q'*M2*Z-M) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N) < sqrt(eps(1.)) &&
      νr == [0] && μr == [1] && ν == [1] && μ == [0] && nf == 0    

M2 = zeros(1,1); N2 = zeros(1,1);   
M = copy(M2); N = copy(N2);

@time M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl  = klf(M, N)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [0] && μr == [1] && νi == [] && νl == [1] && μl == [0]  && nf == 0


# Test Suite 1 (Kagstrom)
M2 = [  22  34  31   31  17
        45  45  42   19  29
        39  47  49   26  34
        27  31  26   21  15
        38  44  44   24  30 ];
    
N2 = [   13  26  25  17  24 
        31  46  40  26  37 
        26  40  19  25  25 
        16  25  27  14  23 
        24  35  18  21  22  ];

M = copy(M2); N = copy(N2); 

@time M1, N1, Q, Z, ν, μ, n, m, p  = klf_rlsplit(M, N)
@test norm(Q'*M2*Z-M1) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N1) < sqrt(eps(1.)) &&
      ν == [1] && μ == [2] && n == 3 && m == 0  && p == 1

M = copy(M2); N = copy(N2); 

@time M1, N1, Q, Z, ν, μ, n, m, p  = klf_rlsplit(M, N, finite_infinite = true)
@test norm(Q'*M2*Z-M1) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N1) < sqrt(eps(1.)) &&
      ν == [2] && μ == [1] && n == 3 && m == 1  && p == 0

M = copy(M2); N = copy(N2); 
@time M1, N1, Q1, Z1, ν, μ, nf, νl, μl = klf_left(M, N)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      ν == [1] && μ == [2] && νl == [1] && μl == [0] && nf == 3   

M = copy(M2); N = copy(N2); 
@time M1, N1, Q1, Z1, ν, μ, nf, νl, μl = klf_left(M, N, fast = false)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      ν == [1] && μ == [2] && νl == [1] && μl == [0] && nf == 3    

M = copy(M2); N = copy(N2); 
@time M1, N1, Q1, Z1, νr, μr, nf, ν, μ = klf_right(M, N)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [0] && μr == [1] && ν == [2] && μ == [1] && nf == 3 

M = copy(M2); N = copy(N2); 
@time M1, N1, Q1, Z1, νr, μr, nf, ν, μ = klf_right(M, N, fast = false)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [0] && μr == [1] && ν == [2] && μ == [1] && nf == 3 

  
M = copy(M2); N = copy(N2);

@time M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl  = klf(M, N,atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [0] && μr == [1] && νi == [1] && νl == [1] && μl == [0]  && nf == 3


M = copy(M2); N = copy(N2);

@time M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl  = klf(M, N, finite_infinite = true)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [0] && μr == [1] && νi == [1] && νl == [1] && μl == [0]  && nf == 3

M = copy(M2); N = copy(N2);

@time N1, M1, Q1, Z1, νr, μr, νi, nf, νl, μl  = klf(N, M)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [0] && μr == [1] && νi == [1, 1] && νl == [1] && μl == [0]  && nf == 2      


M = copy(M2); N = copy(N2);

@time N1, M1, Q1, Z1, νr, μr, νi, nf, νl, μl  = klf(N, M, finite_infinite = true)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [0] && μr == [1] && νi == [1, 1] && νl == [1] && μl == [0]  && nf == 2      


# Test data for MB04UD
M2 = [
   2.0  0.0  2.0 -2.0
   0.0 -2.0  0.0  2.0
   2.0  0.0 -2.0  0.0
   2.0 -2.0  0.0  2.0];
N2 = [
   1.0  0.0  1.0 -1.0
   0.0 -1.0  0.0  1.0
   1.0  0.0 -1.0  0.0
   1.0 -1.0  0.0  1.0];

M = copy(M2); N = copy(N2); 

@time M1, N1, Q, Z, ν, μ, n, m, p  = klf_rlsplit(M, N)
@test norm(Q'*M2*Z-M1) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N1) < sqrt(eps(1.)) &&
      ν == [] && μ == [] && n == 4 && m == 0  && p == 0

M = copy(M2); N = copy(N2); 

@time M1, N1, Q, Z, ν, μ, n, m, p  = klf_rlsplit(M, N, finite_infinite = true)
@test norm(Q'*M2*Z-M1) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N1) < sqrt(eps(1.)) &&
      ν == [] && μ == [] && n == 4 && m == 0  && p == 0


M = copy(M2); N = copy(N2); 
@time Q1, Z1, ν, μ, nf, νl, μl = klf_left!(M, N)
@test norm(Q1'*M2*Z1-M) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N) < sqrt(eps(1.)) &&
      ν == [ ] && μ == [ ]  && νl == [] && μl == [] && nf == 4  

M = copy(M2); N = copy(N2); 
@time Q1, Z1, νr, μr, nf, ν, μ  = klf_right!(M, N)
@test norm(Q1'*M2*Z1-M) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N) < sqrt(eps(1.)) &&
      ν == [ ] && μ == [ ]  && νl == [] && μl == [] && nf == 4  


M = copy(M2); N = copy(N2);

@time M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl  = klf(M, N)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [] && μr == [] && νi == [] && νl == [] && μl == []  && nf == 4

# Test data for MB04VD
M2 = [
   1.0  0.0 -1.0  0.0
   1.0  1.0  0.0 -1.0];
N2 = [
   0.0 -1.0  0.0  0.0
   0.0 -1.0  0.0  0.0];

M = copy(M2); N = copy(N2); 

@time M1, N1, Q, Z, ν, μ, n, m, p  = klf_rlsplit(M, N)
@test norm(Q'*M2*Z-M1) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N1) < sqrt(eps(1.)) &&
      ν == [2, 0] && μ == [3, 1]  && n == 0 && m == 0  && p == 0

M = copy(M2); N = copy(N2); 

@time M1, N1, Q, Z, ν, μ, n, m, p  = klf_rlsplit(M, N, finite_infinite = true)
@test norm(Q'*M2*Z-M1) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N1) < sqrt(eps(1.)) &&
      ν == [1] && μ == [1] && n == 1 && m == 2  && p == 0


M = copy(M2); N = copy(N2); 
@time Q1, Z1, ν, μ, nf, νl, μl = klf_left!(M, N, atol1=1.e-7)
@test norm(Q1'*M2*Z1-M) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N) < sqrt(eps(1.)) &&
      ν == [2, 0] && μ == [3, 1] && νl == [] && μl == [] && nf == 0   

M = copy(M2); N = copy(N2); 
@time Q1, Z1, νr, μr, nf, ν, μ  = klf_right!(M, N, atol1=1.e-7)
@test norm(Q1'*M2*Z1-M) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N) < sqrt(eps(1.)) &&
      νr == [1, 0] && μr == [2, 1] && ν == [1] && μ == [1] && nf == 0   


M = copy(M2); N = copy(N2);

@time M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl  = klf(M, N)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [1, 0] && μr == [2, 1] && νi == [1] && νl == [] && μl == []  && nf == 0

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

@time M1, N1, Q, Z, ν, μ, n, m, p  = klf_rlsplit(M, N, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q'*M2*Z-M1) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N1) < sqrt(eps(1.)) &&
      ν == [4, 2, 0 ] && μ == [6, 3, 1]  && n == 6 && m == 0  && p == 2

M = copy(M2); N = copy(N2); 

@time M1, N1, Q, Z, ν, μ, n, m, p  = klf_rlsplit(M, N, finite_infinite = true, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q'*M2*Z-M1) < sqrt(eps(1.)) &&
      norm(Q'*N2*Z-N1) < sqrt(eps(1.)) &&
      ν == [1, 1, 2, 4] && μ == [0, 1, 2, 3] && n == 6 && m == 4  && p == 0


M = copy(M2); N = copy(N2); 
@time Q1, Z1, νr, μr, nf, ν, μ  = klf_right!(M, N, fast = false, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N) < sqrt(eps(1.)) &&
      νr == [2, 1, 0] && μr == [4, 2, 1]  && ν == [1, 1, 2, 4] && μ == [0, 1, 2, 3] && nf == 3

M = copy(M2); N = copy(N2); 
@time Q1, Z1, νr, μr, nf, ν, μ  = klf_right!(M, N, fast = true, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N) < sqrt(eps(1.)) &&
      νr == [2, 1, 0] && μr == [4, 2, 1]  && ν == [1, 1, 2, 4] && μ == [0, 1, 2, 3] && nf == 3

# the above two cases must coincide : test = OK     

M = copy(M2); N = copy(N2); 
@time Q1, Z1, ν, μ, nf, νl, μl = klf_left!(M, N, fast = false, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N) < sqrt(eps(1.)) &&
      ν == [4, 2, 0 ] && μ == [6, 3, 1] && νl == [1, 1, 1, 2] && μl == [0, 1, 1, 1] && nf == 3


M = copy(M2); N = copy(N2); 
@time Q1, Z1, ν, μ, nf, νl, μl = klf_left!(M, N, fast = true, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N) < sqrt(eps(1.)) &&
      ν == [4, 2, 0 ] && μ == [6, 3, 1] && νl == [1, 1, 1, 2] && μl == [0, 1, 1, 1] && nf == 3

        
M = copy(M2'); N = copy(N2'); 
@time Q1, Z1, ν, μ, nf, νl, μl = klf_left!(M, N, fast = true, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2'*Z1-M) < sqrt(eps(1.)) &&
      norm(Q1'*N2'*Z1-N) < sqrt(eps(1.)) &&
      ν == [3, 2, 1, 0] && μ == [4, 2, 1, 1] && νl == [1, 2, 4] && μl == [0, 1, 2] && nf == 3

M = copy(M2'); N = copy(N2'); 
@time Q1, Z1, ν, μ, nf, νl, μl = klf_left!(M, N, fast = false, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2'*Z1-M) < sqrt(eps(1.)) &&
      norm(Q1'*N2'*Z1-N) < sqrt(eps(1.)) &&
      ν == [3, 2, 1, 0] && μ == [4, 2, 1, 1] && νl == [1, 2, 4] && μl == [0, 1, 2] && nf == 3

M = copy(M2'); N = copy(N2'); 
@time Q1, Z1, νr, μr, nf, ν, μ  = klf_right!(M, N, fast = true, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2'*Z1-M) < sqrt(eps(1.)) &&
      norm(Q1'*N2'*Z1-N) < sqrt(eps(1.)) &&
      νr == [1, 1, 1, 0] && μr == [2, 1, 1, 1] && ν == [1, 3, 6] && μ == [0, 2, 4] && nf == 3
      
M = copy(M2'); N = copy(N2'); 
@time Q1, Z1, νr, μr, nf, ν, μ  = klf_right!(M, N, fast = false, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2'*Z1-M) < sqrt(eps(1.)) &&
      norm(Q1'*N2'*Z1-N) < sqrt(eps(1.)) &&
      νr == [1, 1, 1, 0] && μr == [2, 1, 1, 1] && ν == [1, 3, 6] && μ == [0, 2, 4] && nf == 3


M = copy(M2); N = copy(N2); 
@time M1, N1, Q1, Z1, ν, μ, nf, νl, μl = klf_left(M, N, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      ν == [4, 2, 0 ] && μ == [6, 3, 1] && νl == [1, 1, 1, 2] && μl == [0, 1, 1, 1] && nf == 3

M = copy(M2); N = copy(N2); 
@time M1, N1, Q1, Z1, ν, μ, nf, νl, μl = klf_left(M, N, fast = false, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      ν == [4, 2, 0 ] && μ == [6, 3, 1] && νl == [1, 1, 1, 2] && μl == [0, 1, 1, 1] && nf == 3

M = copy(M2); N = copy(N2); 
@time M1, N1, Q1, Z1, νr, μr, nf, ν, μ = klf_right(M, N, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [2, 1, 0] && μr == [4, 2, 1]  && ν == [1, 1, 2, 4] && μ == [0, 1, 2, 3] && nf == 3

M = copy(M2); N = copy(N2); 
@time M1, N1, Q1, Z1, νr, μr, nf, ν, μ = klf_right(M, N, fast = false, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [2, 1, 0] && μr == [4, 2, 1]  && ν == [1, 1, 2, 4] && μ == [0, 1, 2, 3] && nf == 3



M = copy(M2); N = copy(N2); 
@time M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl = klf(M, N, fast = false, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [2, 1, 0] && μr == [4, 2, 1] && νi == [1, 2] && νl == [1, 1, 1, 2] && μl == [0, 1, 1, 1] && nf == 3

M = copy(M2); N = copy(N2);
@time M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl  = klf(M, N, fast = false, finite_infinite = true, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [2, 1, 0] && μr == [4, 2, 1] && νi == [1, 2] && νl == [1, 1, 1, 2] && μl == [0, 1, 1, 1] && nf == 3


M = copy(M2); N = copy(N2); 
@time M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl = klf(M, N, fast = true, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [2, 1, 0] && μr == [4, 2, 1] && νi == [1, 2] && νl == [1, 1, 1, 2] && μl == [0, 1, 1, 1] && nf == 3

M = copy(M2); N = copy(N2);
@time M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl  = klf(M, N, fast = true, finite_infinite = true, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [2, 1, 0] && μr == [4, 2, 1] && νi == [1, 2] && νl == [1, 1, 1, 2] && μl == [0, 1, 1, 1] && nf == 3



M = copy(M2'); N = copy(N2'); 
@time M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl = klf(M, N, fast = false, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2'*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2'*Z1-N1) < sqrt(eps(1.)) &&
      νr == [1, 1, 1, 0] && μr == [2, 1, 1, 1] && νi == [1, 2] && νl == [1, 2, 4] && μl == [0, 1, 2] && nf == 3

M = copy(M2'); N = copy(N2'); 
@time M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl = klf(M, N, fast = false, finite_infinite = true, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2'*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2'*Z1-N1) < sqrt(eps(1.)) &&
      νr == [1, 1, 1, 0] && μr == [2, 1, 1, 1] && νi == [1, 2] && νl == [1, 2, 4] && μl == [0, 1, 2] && nf == 3


# intermediary computation
M2 = [
    1.0  -1.14463e-18  -1.47479e-15  4.87132e-15  6.96416e-31  -2.32652e-31  -1.50216e-16   1.69785e-16   1.42103e-16   7.77705e-17
    0.0   1.0           0.0          0.0          0.0           0.0          -2.34988e-16  -3.55922e-16  -6.84622e-17  -6.11787e-17
    0.0   0.0           1.0          0.0          0.0           0.0           4.90878e-16   3.80428e-16   3.62292e-16   3.74223e-16
    0.0   0.0           0.0          1.0          0.0           0.0           8.12392e-16   5.94559e-16   1.17495e-16   6.02872e-16
    0.0   0.0           0.0          0.0          0.0           0.0           1.0           5.10024e-16  -2.95823e-31  -2.69854e-16
    0.0   0.0           0.0          0.0          0.0           0.0           0.0           1.0           0.0           5.41115e-16 
]
N2 = [
    0.0  0.0  0.0  0.0  0.0  0.0  -1.29895e-15   4.68167e-15   1.51642e-15  -2.88014e-17
    0.0  0.0  0.0  0.0  0.0  0.0   0.689011     -0.0434045     0.72345      -3.72429e-16
    0.0  0.0  0.0  0.0  0.0  0.0   0.723189      0.10666      -0.682364     -7.51122e-18
    0.0  0.0  0.0  0.0  0.0  0.0  -0.0475455     0.993348      0.10488       1.43459e-17
    0.0  0.0  0.0  0.0  0.0  0.0   0.0           0.0           0.0          -5.10024e-16
    0.0  0.0  0.0  0.0  0.0  0.0   0.0           0.0           0.0          -1.0   
]

M = copy(M2); N = copy(N2); 
@time Q1, Z1, νr, μr, nf, νl, μl  = klf_right!(M, N, fast = false, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N) < sqrt(eps(1.)) &&
      νr == [2, 1, 0] && μr == [4, 2, 1] &&  νl == [1, 2] && μl == [1, 2] && nf == 0


M = copy(M2); N = copy(N2); 
@time Q1, Z1, νr, μr, nf, νl, μl  = klf_right!(M, N, fast = true, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N) < sqrt(eps(1.)) &&
      νr == [2, 1, 0] && μr == [4, 2, 1] &&  νl == [1, 2] && μl == [1, 2] && nf == 0

M = copy(M2); N = copy(N2); 
@time M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl = klf(M, N, fast = false, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [2, 1, 0] && μr == [4, 2, 1] && νi == [1, 2] && νl == [] && μl == [] && nf == 0      


M = copy(M2); N = copy(N2); 
@time M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl = klf(M, N, fast = true, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [2, 1, 0] && μr == [4, 2, 1] && νi == [1, 2] && νl == [] && μl == [] && nf == 0 
      

M = copy(M2); N = copy(N2); 
@time M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl = klf(M', N', fast = false, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2'*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2'*Z1-N1) < sqrt(eps(1.)) &&
      μl == [0, 1, 2] && νl == [1, 2, 4] && νi == [1, 2] && νr == [] && μr == [] && nf == 0   




M2 = [
1.0  -1.51959e-15   4.9582e-15   -7.05588e-17   6.96416e-31  -2.32652e-31   2.38551e-16  -2.02364e-16   5.79165e-17   7.77705e-17
0.0   1.0           2.77556e-17  -1.73472e-18  -2.76763e-47  -1.35446e-47  -1.43825e-16  -3.50328e-16  -5.85653e-16   3.74223e-16
0.0   0.0          -1.0           2.77556e-17   4.26419e-48   9.29779e-48   5.70351e-16   5.60542e-16   6.19106e-16  -6.02872e-16
0.0   0.0           0.0          -1.0           7.98872e-48   1.74576e-47  -1.64948e-16  -3.84582e-16  -1.94459e-16   6.11787e-17
0.0   0.0           0.0           0.0           0.0           0.0           0.731013      0.0979        0.675304      2.52908e-16
0.0   0.0           0.0           0.0           0.0           0.0           0.0          -0.989654      0.143472      5.77433e-16
]
N2 = [
    0.0  0.0  0.0  0.0  0.0  0.0   1.40857e-15  -4.84718e-15  -4.04657e-17  -4.16334e-17
    0.0  0.0  0.0  0.0  0.0  0.0  -1.0           2.21239e-7    5.83691e-7    2.19061e-17
    0.0  0.0  0.0  0.0  0.0  0.0   0.0           1.0           2.90233e-7   -1.24119e-17
    0.0  0.0  0.0  0.0  0.0  0.0   0.0           0.0           1.0           3.44402e-16
    0.0  0.0  0.0  0.0  0.0  0.0   0.0           0.0           0.0           0.145907
    0.0  0.0  0.0  0.0  0.0  0.0   0.0           0.0           0.0          -0.989298
]   
M = copy(M2); N = copy(N2); 
@time Q1, Z1, νr, μr, nf, νl, μl  = klf_right!(M, N, fast = true, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N) < sqrt(eps(1.)) &&
      νr == [2, 1, 0] && μr == [4, 2, 1] &&  νl == [1, 2] && μl == [1, 2] && nf == 0 
      
      
# Example  Karathanasi and Karampetakis IMA J. Math. Contr Inform. (2019)
N2 = [
1. 1 0 0 1 0 -1 0
1 1 1 1 0 0 0 0
-1 1 0 0 0 0 0 0
1 0 0 0 0 -1 0 0
0 -2 0 0 0 1 1 0
1 0 0 0 0 -1 0 0
]
M2 = [
2. 1 0 0 0 -2 0 0
0 2 1 0 0 0 -2 0
-2 1 0 0 0 1 -1 1
3 -1 1 1 0 -2 2 0
0 -2 0 0 0 1 2 -1
2 -1 0 0 0 -2 1 0
]

M = copy(M2); N = copy(N2); 
@time M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl = klf(M, N, fast = false, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [2, 0] && μr == [2, 2] && νi == [1, 1] && νl == [] && μl == [] && nf == 2

M = copy(M2); N = copy(N2); 
@time M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl = klf(M, N, fast = true, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N1) < sqrt(eps(1.)) &&
      νr == [2, 0] && μr == [2, 2] && νi == [1, 1] && νl == [] && μl == [] && nf == 2

M = copy(M2); N = copy(N2); 
@time M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl = klf(M', N', fast = false, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2'*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2'*Z1-N1) < sqrt(eps(1.)) &&
      νl == [2, 2] && μl == [0, 2] && νi == [1, 1] && νr == [] && μr == [] && nf == 2


M = copy(M2); N = copy(N2); 
@time M1, N1, Q1, Z1, νr, μr, νi, nf, νl, μl = klf(M', N', finite_infinite = true, fast = true, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2'*Z1-M1) < sqrt(eps(1.)) &&
      norm(Q1'*N2'*Z1-N1) < sqrt(eps(1.)) &&
      νl == [2, 2] && μl == [0, 2] && νi == [1, 1] && νr == [] && μr == [] && nf == 2


M = copy(M2); N = copy(N2); 
@time Q1, Z1, ν, μ, nf, νl, μl = klf_left!(M, N, fast = false, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N) < sqrt(eps(1.)) &&
      ν == [3, 1 ] && μ == [3, 3] && νl == [] && μl == [] && nf == 2

M = copy(M2); N = copy(N2); 
@time Q1, Z1, ν, μ, nf, νl, μl = klf_left!(M, N, fast = true, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N) < sqrt(eps(1.)) &&
      ν == [3, 1 ] && μ == [3, 3] && νl == [] && μl == [] && nf == 2


M = copy(M2); N = copy(N2); 
@time Q1, Z1, νr, μr, nf, ν, μ  = klf_right!(M, N, fast = true, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N) < sqrt(eps(1.)) &&
      νr == [2, 0 ] && μr == [2, 2] && ν == [1, 1] && μ == [1, 1] && nf == 2


M = copy(M2); N = copy(N2); 
@time Q1, Z1, νr, μr, nf, ν, μ  = klf_right!(M, N, fast = false, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q1'*M2*Z1-M) < sqrt(eps(1.)) &&
      norm(Q1'*N2*Z1-N) < sqrt(eps(1.)) &&
      νr == [2, 0 ] && μr == [2, 2] && ν == [1, 1] && μ == [1, 1] && nf == 2

for fast in (true, false)

for Ty in (Float64, Float32, Complex{Float64},  Complex{Float32})

abstol = sqrt(eps(one(real(Ty))))

# generic cases 
m = 1; n = 2;
M2 = rand(Ty,m,n)
N2 = rand(Ty,m,n)
M = copy(M2); N = copy(N2);
@time Q, Z, ν, μ, nf, νl, μl = klf_left!(M, N, fast = fast)
@test norm(Q'*M2*Z-M) < abstol &&
      norm(Q'*N2*Z-N) < abstol &&
      ν == [1, 0 ] && μ == [1, 1] && νl == [] && μl == [] && nf == 0

M = copy(M2); N = copy(N2);
@time Q, Z, νr, μr, nf, ν, μ = klf_right!(M, N, fast = fast)
@test norm(Q'*M2*Z-M) < abstol &&
      norm(Q'*N2*Z-N) < abstol &&
      νr == [1, 0 ] && μr == [1, 1] && ν == [] && μ == [] && nf == 0

m = 3; n = 5;
M2 = rand(Ty,m,n)
N2 = rand(Ty,m,n)
M = copy(M2); N = copy(N2);
@time Q, Z, ν, μ, nf, νl, μl = klf_left!(M, N, fast = fast)
@test norm(Q'*M2*Z-M) < abstol &&
      norm(Q'*N2*Z-N) < abstol &&
      ν == [2, 1, 0] && μ == [2, 2, 1] && νl == [] && μl == [] && nf == 0

m = 3; n = 5;
M2 = rand(Ty,m,n)
N2 = rand(Ty,m,n)
M = copy(M2); N = copy(N2);
@time Q, Z, νr, μr, nf, ν, μ = klf_right!(M, N, fast = fast)
@test norm(Q'*M2*Z-M) < abstol &&
      norm(Q'*N2*Z-N) < abstol &&
      νr == [2, 1, 0] && μr == [2, 2, 1] && ν == [] && μ == [] && nf == 0

      
m = 5; n = 3; 
M2 = rand(Ty,m,n)
N2 = rand(Ty,m,n)
M = copy(M2); N = copy(N2);
@time Q, Z, ν, μ, nf, νl, μl = klf_left!(M, N, fast = fast);
@test norm(Q'*M2*Z-M) < abstol &&
      norm(Q'*N2*Z-N) < abstol &&
      ν == [] && μ == [] && νl == [1, 2, 2] && μl == [0, 1, 2] && nf == 0

m = 5; n = 3; 
M2 = rand(Ty,m,n)
N2 = rand(Ty,m,n)
M = copy(M2); N = copy(N2);
@time Q, Z, νr, μr, nf, ν, μ = klf_right!(M, N, fast = fast);
@test norm(Q'*M2*Z-M) < abstol &&
      norm(Q'*N2*Z-N) < abstol &&
      νr == [] && μr == [] && ν == [1, 2, 2] && μ == [0, 1, 2] && nf == 0

m = 3; n = 5; r = 2;
M2 = rand(Ty,m,n)
N2 = rand(Ty,m,m)*[rand(Ty,r,r) zeros(Ty,r,n-r); zeros(Ty,m-r,n)]*rand(Ty,n,n)
M = copy(M2); N = copy(N2);
@time Q, Z, ν, μ, nf, νl, μl = klf_left!(M, N; fast = fast);
@test norm(Q'*M2*Z-M) < abstol &&
      norm(Q'*N2*Z-N) < abstol &&
      ν == [3, 0] && μ == [3, 2] && νl == [] && μl == [] && nf == 0

m = 3; n = 5; r = 2;
M2 = rand(Ty,m,n)
N2 = rand(Ty,m,m)*[rand(Ty,r,r) zeros(Ty,r,n-r); zeros(Ty,m-r,n)]*rand(Ty,n,n)
M = copy(M2); N = copy(N2);
@time Q, Z, νr, μr, nf, ν, μ = klf_right!(M, N; fast = fast)
@test norm(Q'*M2*Z-M) < abstol &&
      norm(Q'*N2*Z-N) < abstol &&
      νr == [2, 0] && μr == [2, 2] && ν == [1] && μ == [1] && nf == 0

m = 3; n = 5; r = 2;
M2 = rand(Ty,m,n)
N2 = rand(Ty,m,m)*[rand(Ty,r,r) zeros(Ty,r,n-r); zeros(Ty,m-r,n)]*rand(Ty,n,n)
M = copy(M2); N = copy(N2);
@time M1, N1, Q, Z, νr, μr, νi, nf, νl, μl = klf(M, N; fast = fast)
@test norm(Q'*M2*Z-M1) < abstol &&
      norm(Q'*N2*Z-N1) < abstol &&
      νr == [2, 0] && μr == [2, 2] && νi == [1] && νl == [] && μl == [] && nf == 0

m = 3; n = 5; r = 2;
M2 = rand(Ty,m,n)
N2 = rand(Ty,m,m)*[rand(Ty,r,r) zeros(Ty,r,n-r); zeros(Ty,m-r,n)]*rand(Ty,n,n)
M = copy(M2); N = copy(N2);
@time M1, N1, Q, Z, νr, μr, nf, ν, μ = klf_right(M, N; fast = fast)
@test norm(Q'*M2*Z-M1) < abstol &&
      norm(Q'*N2*Z-N1) < abstol &&
      νr == [2, 0] && μr == [2, 2] && ν == [1] && μ == [1] && nf == 0

m = 3; n = 5; r = 2;
M2 = rand(Ty,m,n)
N2 = rand(Ty,m,m)*[rand(Ty,r,r) zeros(Ty,r,n-r); zeros(Ty,m-r,n)]*rand(Ty,n,n)
M = copy(M2); N = copy(N2);
@time M1, N1, Q, Z, ν, μ, nf, νl, μl = klf_left(M, N; fast = fast)
@test norm(Q'*M2*Z-M1) < abstol &&
      norm(Q'*N2*Z-N1) < abstol &&
      ν == [3, 0] && μ == [3, 2] && νl == [] && μl == [] && nf == 0


M2 = rand(Ty,7,7)
N2 = [zeros(Ty,4,3) rand(Ty,4,4); zeros(Ty,3,7)]
M = copy(M2); N = copy(N2);
@time Q, Z, ν, μ, nf, νl, μl = klf_left!(M, N, fast = fast);
@test norm(Q'*M2*Z-M) < abstol &&
      norm(Q'*N2*Z-N) < abstol &&
      ν == [3] && μ == [3] && νl == [] && μl == [] && nf == 4


M2 = rand(Ty,7,7)
N2 = [zeros(Ty,4,3) rand(Ty,4,4); zeros(Ty,3,7)]
M = copy(M2); N = copy(N2);
@time Q, Z, νr, μr, nf, ν, μ = klf_right!(M, N, fast = fast);
@test norm(Q'*M2*Z-M) < abstol &&
      norm(Q'*N2*Z-N) < abstol &&
      νr == [] && μr == [] && ν == [3] && μ == [3] && nf == 4
end
end
   
end

end
