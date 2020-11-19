module Test_gsep


using LinearAlgebra
using MatrixPencils
using Test


@testset "Spectrum separation functions" begin

@testset "fihess" begin

A2 = zeros(0,0); E2 = zeros(0,0); 
A = copy(A2); E = copy(E2); 

@time A1, E1, Q, Z, ν, blkdims  = fihess(A,E)
@test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && 
      ν == [] && blkdims == (0,0)


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
    0     0     1     0     0     0     0     0     0
    0     0     0     1     0     0     0     0     0
    0     0     0     0     1     0     0     0     0
    0     0     0     0     0     0     0     0     0
    0     0     0     0     0     0     1     0     0
    0     0     0     0     0     0     0     1     0];


A = copy(A2); E = copy(E2); 
@time A1, E1, Q, Z, ν, blkdims  = fihess(A,E,finite_infinite=false, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && istriu(A1) &&
      ν == [2, 2, 2, 1, 1, 1] && blkdims == (9,0)  

A = copy(A2); E = copy(E2); 
@time A1, E1, Q, Z, ν, blkdims  = fihess(A,E,finite_infinite=true, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && istriu(A1) &&
      ν == [1, 1, 1, 2, 2, 2] && blkdims == (0,9)   


fast = true; Ty = Float64      

for fast in (true, false)

for Ty in (Float64, Complex{Float64})

A2 = rand(Ty,3,3); E2 = zeros(Ty,3,3); 
A = copy(A2); E = copy(E2); 

@time A1, E1, Q, Z, ν, blkdims  = fihess(A,E,fast = fast, finite_infinite = true)
@test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && istriu(A1) &&
      ν == [3] && blkdims == (0,3)   
      
@time A1, E1, Q, Z, ν, blkdims  = fihess(A,E,fast = fast, finite_infinite = false)
@test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && istriu(A1) &&
      ν == [3] && blkdims == (3,0)  
      
A2 = rand(3,3); E2 = triu(rand(Ty,3,3),1); 
A = copy(A2); E = copy(E2);  

@time A1, E1, Q, Z, ν, blkdims  = fihess(A,E,fast = fast, finite_infinite = true)
@test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) &&
      ν == [1] && blkdims == (2,1)   
      

@time A1, E1, Q, Z, ν, blkdims  = fihess(A,E,fast = fast, finite_infinite = false)
@test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) &&
      ν == [1] && blkdims == (1,2)   
      

A2 = zeros(3,3); E2 = rand(3,3); 
A = copy(A2); E = copy(E2); 

@time A1, E1, Q, Z, ν, blkdims  = fihess(A,E,fast = fast, finite_infinite = true)
@test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) &&
      ν == [] && blkdims == (3,0)   
           
@time A1, E1, Q, Z, ν, blkdims  = fihess(A,E,fast = fast, finite_infinite = false)
@test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) &&
      ν == [] && blkdims == (0,3)  
      
A2 = rand(3,3); E2 = rand(3,3); 
A = copy(A2); E = copy(E2); 

@time A1, E1, Q, Z, ν, blkdims  = fihess(A',E',fast = fast, finite_infinite = true)
@test norm(Q'*A2'*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2'*Z-E1) < sqrt(eps(1.)) && 
      ν == [] && blkdims == (3,0)       
      
@time A1, E1, Q, Z, ν, blkdims  = fihess(A',E',fast = fast, finite_infinite = false)
@test norm(Q'*A2'*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2'*Z-E1) < sqrt(eps(1.)) && 
      ν == [] && blkdims == (0,3)      
      
end
end
end # fihess

@testset "fischur" begin

A2 = zeros(0,0); E2 = zeros(0,0); 
A = copy(A2); E = copy(E2); 

@time A1, E1, Q, Z, ν, blkdims  = fischur(A,E)
@test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && 
      ν == [] && blkdims == (0,0)


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
    0     0     1     0     0     0     0     0     0
    0     0     0     1     0     0     0     0     0
    0     0     0     0     1     0     0     0     0
    0     0     0     0     0     0     0     0     0
    0     0     0     0     0     0     1     0     0
    0     0     0     0     0     0     0     1     0];


A = copy(A2); E = copy(E2); 
@time A1, E1, Q, Z, ν, blkdims  = fischur(A,E,finite_infinite=false, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && isqtriu(A1) && istriu(A1) && istriu(E1) &&
      ν == [2, 2, 2, 1, 1, 1] && blkdims == (9,0)  

A = copy(A2); E = copy(E2); 
@time A1, E1, Q, Z, ν, blkdims  = fischur(A,E,finite_infinite=true, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && isqtriu(A1) && istriu(A1) && istriu(E1) &&
      ν == [1, 1, 1, 2, 2, 2] && blkdims == (0,9)   


fast = true; finite_infinite = true; Ty = Float64      

for fast in (true, false)

for Ty in (Float64, Complex{Float64})

A2 = rand(Ty,3,3); E2 = zeros(Ty,3,3); 
A = copy(A2); E = copy(E2); 

@time A1, E1, Q, Z, ν, blkdims  = fischur(A,E,fast = fast, finite_infinite = true)
@test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && istriu(A1) && istriu(E1) &&
      ν == [3] && blkdims == (0,3)   
      
@time A1, E1, Q, Z, ν, blkdims  = fischur(A,E,fast = fast, finite_infinite = false)
@test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && istriu(A1) && istriu(E1) &&
      ν == [3] && blkdims == (3,0)   
      
A2 = rand(3,3); E2 = triu(rand(Ty,3,3),1); 
A = copy(A2); E = copy(E2);  

@time A1, E1, Q, Z, ν, blkdims  = fischur(A,E,fast = fast, finite_infinite = true)
@test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && isqtriu(A1) && istriu(E1) &&
      ν == [1] && blkdims == (2,1)   
      
@time A1, E1, Q, Z, ν, blkdims  = fischur(A,E,fast = fast, finite_infinite = false)
@test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && isqtriu(A1) && istriu(E1) &&
      ν == [1] && blkdims == (1,2)   
      

A2 = zeros(3,3); E2 = rand(3,3); 
A = copy(A2); E = copy(E2); 

@time A1, E1, Q, Z, ν, blkdims  = fischur(A,E,fast = fast, finite_infinite = true)
@test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && isqtriu(A1) && istriu(E1) &&
      ν == [] && blkdims == (3,0)   
      
@time A1, E1, Q, Z, ν, blkdims  = fischur(A,E,fast = fast, finite_infinite = false)
@test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && isqtriu(A1) && istriu(E1) &&
      ν == [] && blkdims == (0,3)   
      
A2 = rand(3,3); E2 = rand(3,3); 
A = copy(A2); E = copy(E2); 

@time A1, E1, Q, Z, ν, blkdims  = fischur(A',E',fast = fast, finite_infinite = true)
@test norm(Q'*A2'*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2'*Z-E1) < sqrt(eps(1.)) && isqtriu(A1) && istriu(E1) &&
      ν == [] && blkdims == (3,0)     
      
@time A1, E1, Q, Z, ν, blkdims  = fischur(A',E',fast = fast, finite_infinite = false)
@test norm(Q'*A2'*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2'*Z-E1) < sqrt(eps(1.)) && isqtriu(A1) && istriu(E1) &&
      ν == [] && blkdims == (0,3)     
      
end
end
end # fischur

@testset "fischursep" begin

A2 = zeros(0,0); E2 = zeros(0,0); 
A = copy(A2); E = copy(E2); 

@time A1, E1, Q, Z, ν, blkdims  = fischursep(A,E)
@test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && 
      ν == [] && blkdims == (0,0,0)

Q = qr(rand(5,5)).Q; Z = qr(rand(5,5)).Q;
A2 = Q*[
 1.0  0.0  0.0  0.0  0.0
 0.0  1.0  0.0  0.0  0.0
 0.0  0.0  1.0  0.0  0.0
 0.0  0.0  0.0  1.0  0.0
 0.0  0.0  0.0  0.0  1.0]*Z;
E2 = Q*[ 0.0  1.0  0.0  0.0  0.0
 0.0  0.0  1.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  1.0
 0.0  0.0  0.0  0.0  0.0]*Z;
 A = copy(A2); E = copy(E2); 
 @time A1, E1, Q, Z, ν, blkdims  = fischursep(A,E,finite_infinite=true, atol1 = 1.e-7, atol2 = 1.e-7) #error
 @test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
       norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && isqtriu(A1) && istriu(A1) && istriu(E1) &&
       ν == [1, 2, 2] && blkdims == (0,0,5) 

 @time A1, E1, Q, Z, ν, blkdims  = fischursep(A,E,finite_infinite=false, atol1 = 1.e-7, atol2 = 1.e-7) #error
 @test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
       norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && isqtriu(A1) && istriu(A1) && istriu(E1) &&
       ν == [2, 2, 1] && blkdims == (5,0,0)   

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
    0     0     1     0     0     0     0     0     0
    0     0     0     1     0     0     0     0     0
    0     0     0     0     1     0     0     0     0
    0     0     0     0     0     0     0     0     0
    0     0     0     0     0     0     1     0     0
    0     0     0     0     0     0     0     1     0];


A = copy(A2); E = copy(E2); 
@time A1, E1, Q, Z, ν, blkdims  = fischursep(A,E,finite_infinite=false, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && isqtriu(A1) && istriu(A1) && istriu(E1) &&
      ν == [2, 2, 2, 1, 1, 1] && blkdims == (9,0,0)  

A = copy(A2); E = copy(E2); 
@time A1, E1, Q, Z, ν, blkdims  = fischursep(A,E,finite_infinite=true, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && isqtriu(A1) && istriu(A1) && istriu(E1) &&
      ν == [1, 1, 1, 2, 2, 2] && blkdims == (0,0,9)   

fast = true; Ty = Float64      

for fast in (true, false)

for Ty in (Float64, Complex{Float64})

A2 = rand(Ty,3,3); E2 = zeros(Ty,3,3); 
A = copy(A2); E = copy(E2); 

@time A1, E1, Q, Z, ν, blkdims  = fischursep(A,E,fast = fast, finite_infinite = true)
@test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && istriu(A1) && istriu(E1) &&
      ν == [3] && blkdims == (0,0,3)   
      
@time A1, E1, Q, Z, ν, blkdims  = fischursep(A,E,fast = fast, finite_infinite = false)
@test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && istriu(A1) && istriu(E1) &&
      ν == [3] && blkdims == (3,0,0)   
      
A2 = rand(Ty,3,3); E2 = triu(rand(Ty,3,3),1); B2 = rand(Ty,3,3);

F2, = saloc(A2,E2,B2; evals = [-.1;2])
A2c = copy(A2+B2*F2); A = copy(A2c); E = copy(E2);  
@time A1, E1, Q, Z, ν, blkdims  = fischursep(A,E,fast = fast, finite_infinite = true,stable_unstable=true)
@test norm(Q'*A2c*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && isqtriu(A1) && istriu(E1) &&
      ν == [1] && blkdims == (1,1,1) 

F2, = saloc(A2,E2,B2; evals = [-.1;-2])
A2c = copy(A2+B2*F2); A = copy(A2c); E = copy(E2);  
@time A1, E1, Q, Z, ν, blkdims  = fischursep(A,E,fast = fast, finite_infinite = true,stable_unstable=true)
@test norm(Q'*A2c*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && isqtriu(A1) && istriu(E1) &&
      ν == [1] && blkdims == (2,0,1) 

F2, = saloc(A2,E2,B2; evals = [-.1;-.2])
A2c = copy(A2+B2*F2); A = copy(A2c); E = copy(E2);  
@time A1, E1, Q, Z, ν, blkdims  = fischursep(A,E,fast = fast, finite_infinite = false,stable_unstable=true)
@test norm(Q'*A2c*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && isqtriu(A1) && istriu(E1) &&
      ν == [1] && blkdims == (1,2,0) 

F2, = saloc(A2,E2,B2; evals = [.1;.2])
A2c = copy(A2+B2*F2); A = copy(A2c); E = copy(E2);  
@time A1, E1, Q, Z, ν, blkdims  = fischursep(A,E,fast = fast, finite_infinite = false,stable_unstable=true)
@test norm(Q'*A2c*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && isqtriu(A1) && istriu(E1) &&
      ν == [1] && blkdims == (1,0,2) 

F2, = saloc(A2,E2,B2; evals = [.1;.2])
A2c = copy(A2+B2*F2); A = copy(A2c); E = copy(E2);  
@time A1, E1, Q, Z, ν, blkdims  = fischursep(A,E,fast = fast, finite_infinite = false,stable_unstable=true,disc=true)
@test norm(Q'*A2c*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && isqtriu(A1) && istriu(E1) &&
      ν == [1] && blkdims == (1,2,0) 

F2, = saloc(A2,E2,B2; evals = [.1;.2])
A2c = copy(A2+B2*F2); A = copy(A2c); E = copy(E2);  
@time A1, E1, Q, Z, ν, blkdims  = fischursep(A,E,fast = fast, finite_infinite = false,stable_unstable=false,disc=true)
@test norm(Q'*A2c*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && isqtriu(A1) && istriu(E1) &&
      ν == [1] && blkdims == (1,0,2) 

F2, = saloc(A2,E2,B2; evals = [.1;2])
A2c = copy(A2+B2*F2); A = copy(A2c); E = copy(E2);  
@time A1, E1, Q, Z, ν, blkdims  = fischursep(A,E,fast = fast, finite_infinite = false,stable_unstable=true,disc=true)
@test norm(Q'*A2c*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && isqtriu(A1) && istriu(E1) &&
      ν == [1] && blkdims == (1,1,1) 

A2 = zeros(3,3); E2 = rand(3,3); B2 = rand(Ty,3,3);
A = copy(A2); E = copy(E2); 

F2, = saloc(A2,E2,B2; evals = [-.1,-2,2])
A2c = copy(A2+B2*F2); A = copy(A2c); E = copy(E2);  
@time A1, E1, Q, Z, ν, blkdims  = fischursep(A,E,fast = fast, finite_infinite = true, stable_unstable=true)
@test norm(Q'*A2c*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && isqtriu(A1) && istriu(E1) &&
      ν == [] && blkdims == (2, 1, 0)   

F2, = saloc(A2,E2,B2; evals = [-.1,-2,2])
A2c = copy(A2+B2*F2); A = copy(A2c); E = copy(E2);  
@time A1, E1, Q, Z, ν, blkdims  = fischursep(A,E,fast = fast, finite_infinite = true, stable_unstable=false)
@test norm(Q'*A2c*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && isqtriu(A1) && istriu(E1) &&
      ν == [] && blkdims == (1, 2, 0)   
    
end
end
end # fischursep

@testset "sfischursep" begin

A2 = zeros(0,0); E2 = zeros(0,0); 
A = copy(A2); E = copy(E2); 

@time A1, E1, Q, Z, ν, blkdims  = sfischursep(A,E)
@test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && 
      ν == [] && blkdims == (0,0,0,0)
     
Q = qr(rand(5,5)).Q; Z = qr(rand(5,5)).Q;
A2 = Q*[
 1.0  0.0  0.0  0.0  0.0
 0.0  1.0  0.0  0.0  0.0
 0.0  0.0  1.0  0.0  0.0
 0.0  0.0  0.0  1.0  0.0
 0.0  0.0  0.0  0.0  1.0]*Z;
E2 = Q*[ 0.0  1.0  0.0  0.0  0.0
 0.0  0.0  1.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  1.0
 0.0  0.0  0.0  0.0  0.0]*Z;
 A = copy(A2); E = copy(E2); 
 @time A1, E1, Q, Z, ν, blkdims  = sfischursep(A,E,finite_infinite=true, atol1 = 1.e-7, atol2 = 1.e-7) #OK
 @test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
       norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && isqtriu(A1) && istriu(A1) && istriu(E1) &&
       ν == [1, 2] && blkdims == (2,0,0,3) 

 @time A1, E1, Q, Z, ν, blkdims  = sfischursep(A,E,finite_infinite=false, atol1 = 1.e-7, atol2 = 1.e-7) #OK
 @test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
       norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && isqtriu(A1) && istriu(A1) && istriu(E1) &&
       ν == [2, 1] && blkdims == (3,0,0,2)   

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
    0     0     1     0     0     0     0     0     0
    0     0     0     1     0     0     0     0     0
    0     0     0     0     1     0     0     0     0
    0     0     0     0     0     0     0     0     0
    0     0     0     0     0     0     1     0     0
    0     0     0     0     0     0     0     1     0];

A = copy(A2); E = copy(E2); 
@time A1, E1, Q, Z, ν, blkdims  = sfischursep(A,E,finite_infinite=false, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && isqtriu(A1) && istriu(A1) && istriu(E1) &&
      ν == [2, 2, 1, 1, 1] && blkdims == (7,0,0,2)  

A = copy(A2); E = copy(E2); 
@time A1, E1, Q, Z, ν, blkdims  = sfischursep(A,E,finite_infinite=true, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && isqtriu(A1) && istriu(A1) && istriu(E1) &&
      ν == [1, 1, 1, 2, 2] && blkdims == (2,0,0,7)   

fast = true; Ty = Float64      

for fast in (true, false)

for Ty in (Float64, Complex{Float64})

A2 = rand(Ty,3,3); E2 = zeros(Ty,3,3); 
A = copy(A2); E = copy(E2); 

@time A1, E1, Q, Z, ν, blkdims  = sfischursep(A,E,fast = fast, finite_infinite = true)
@test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && istriu(A1) && istriu(E1) &&
      ν == [] && blkdims == (3,0,0,0)   
      
@time A1, E1, Q, Z, ν, blkdims  = sfischursep(A,E,fast = fast, finite_infinite = false)
@test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && istriu(A1) && istriu(E1) &&
      ν == [] && blkdims == (0,0,0,3)   
      
A2 = rand(Ty,3,3); E2 = triu(rand(Ty,3,3),1); B2 = rand(Ty,3,3);

F2, = saloc(A2,E2,B2; evals = [-.1;2])
A2c = copy(A2+B2*F2); A = copy(A2c); E = copy(E2);  
@time A1, E1, Q, Z, ν, blkdims  = sfischursep(A,E,fast = fast, finite_infinite = true,stable_unstable=true)
@test norm(Q'*A2c*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && isqtriu(A1) && istriu(E1) &&
      ν == [] && blkdims == (1,1,1,0) 

F2, = saloc(A2,E2,B2; evals = [-.1;-2])
A2c = copy(A2+B2*F2); A = copy(A2c); E = copy(E2);  
@time A1, E1, Q, Z, ν, blkdims  = sfischursep(A,E,fast = fast, finite_infinite = true,stable_unstable=true)
@test norm(Q'*A2c*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && isqtriu(A1) && istriu(E1) &&
      ν == [] && blkdims == (1,2,0,0) 

F2, = saloc(A2,E2,B2; evals = [-.1;-.2])
A2c = copy(A2+B2*F2); A = copy(A2c); E = copy(E2);  
@time A1, E1, Q, Z, ν, blkdims  = sfischursep(A,E,fast = fast, finite_infinite = false,stable_unstable=true)
@test norm(Q'*A2c*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && isqtriu(A1) && istriu(E1) &&
      ν == [] && blkdims == (0, 2, 0, 1) 

F2, = saloc(A2,E2,B2; evals = [.1;.2])
A2c = copy(A2+B2*F2); A = copy(A2c); E = copy(E2);  
@time A1, E1, Q, Z, ν, blkdims  = sfischursep(A,E,fast = fast, finite_infinite = false,stable_unstable=true)
@test norm(Q'*A2c*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && isqtriu(A1) && istriu(E1) &&
      ν == [] && blkdims == (0, 0, 2, 1) 

F2, = saloc(A2,E2,B2; evals = [.1;.2])
A2c = copy(A2+B2*F2); A = copy(A2c); E = copy(E2);  
@time A1, E1, Q, Z, ν, blkdims  = sfischursep(A,E,fast = fast, finite_infinite = false,stable_unstable=true,disc=true)
@test norm(Q'*A2c*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && isqtriu(A1) && istriu(E1) &&
      ν == [] && blkdims == (0, 2, 0, 1) 

F2, = saloc(A2,E2,B2; evals = [.1;.2])
A2c = copy(A2+B2*F2); A = copy(A2c); E = copy(E2);  
@time A1, E1, Q, Z, ν, blkdims  = sfischursep(A,E,fast = fast, finite_infinite = false,stable_unstable=false,disc=true)
@test norm(Q'*A2c*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && isqtriu(A1) && istriu(E1) &&
      ν == [] && blkdims == (0, 0, 2, 1) 

F2, = saloc(A2,E2,B2; evals = [.1;2])
A2c = copy(A2+B2*F2); A = copy(A2c); E = copy(E2);  
@time A1, E1, Q, Z, ν, blkdims  = sfischursep(A,E,fast = fast, finite_infinite = false,stable_unstable=true,disc=true)
@test norm(Q'*A2c*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && isqtriu(A1) && istriu(E1) &&
      ν == [] && blkdims == (0,1,1,1) 

A2 = zeros(3,3); E2 = rand(3,3); B2 = rand(Ty,3,3);
A = copy(A2); E = copy(E2); 

F2, = saloc(A2,E2,B2; evals = [-.1,-2,2])
A2c = copy(A2+B2*F2); A = copy(A2c); E = copy(E2);  
@time A1, E1, Q, Z, ν, blkdims  = sfischursep(A,E,fast = fast, finite_infinite = true, stable_unstable=true)
@test norm(Q'*A2c*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && isqtriu(A1) && istriu(E1) &&
      ν == [] && blkdims == (0,2, 1, 0)   

F2, = saloc(A2,E2,B2; evals = [-.1,-2,2])
A2c = copy(A2+B2*F2); A = copy(A2c); E = copy(E2);  
@time A1, E1, Q, Z, ν, blkdims  = sfischursep(A,E,fast = fast, finite_infinite = true, stable_unstable=false)
@test norm(Q'*A2c*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) && isqtriu(A1) && istriu(E1) &&
      ν == [] && blkdims == (0,1, 2, 0)   
    
end
end
end # sfischursep

@testset "fiblkdiag" begin

A2 = zeros(0,0); E2 = zeros(0,0); 
A = copy(A2); E = copy(E2); 

@time A1, E1, B1, C1, Q, Z, ν, blkdims  = fiblkdiag(A,E,missing,missing)
@test norm(Q*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q*E2*Z-E1) < sqrt(eps(1.)) && 
      ν == [] && blkdims == (0,0)


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
    0     0     1     0     0     0     0     0     0
    0     0     0     1     0     0     0     0     0
    0     0     0     0     1     0     0     0     0
    0     0     0     0     0     0     0     0     0
    0     0     0     0     0     0     1     0     0
    0     0     0     0     0     0     0     1     0];

B2 = rand(9,2); C2 = rand(3,9);
A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2);
@time A1, E1, B1, C1, Q, Z, ν, blkdims  = fiblkdiag(A,E,B,C, finite_infinite=false, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q*E2*Z-E1) < sqrt(eps(1.)) && 
      norm(Q*B2-B1) < sqrt(eps(1.)) &&
      norm(C2*Z-C1) < sqrt(eps(1.)) && 
      isqtriu(A1) && istriu(A1) && istriu(E1) &&
      ν == [2, 2, 2, 1, 1, 1] && blkdims == (9,0)  

A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2);
@time A1, E1, B1, C1, Q, Z, ν, blkdims  = fiblkdiag(A,E,B,C,finite_infinite=true, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q*E2*Z-E1) < sqrt(eps(1.)) && 
      norm(Q*B2-B1) < sqrt(eps(1.)) &&
      norm(C2*Z-C1) < sqrt(eps(1.)) && 
      isqtriu(A1) && istriu(A1) && istriu(E1) &&
      ν == [1, 1, 1, 2, 2, 2] && blkdims == (0,9)   

fast = true; Ty = Float64      

for fast in (true, false)

for Ty in (Float64, Complex{Float64})

A2 = rand(Ty,3,3); E2 = zeros(Ty,3,3); B2 = rand(Ty,3,2); C2 = rand(Ty,3,3);

A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2);

@time A1, E1, B1, C1, Q, Z, ν, blkdims  = fiblkdiag(A,E,B,C,fast = fast, finite_infinite = true)
@test norm(Q*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q*E2*Z-E1) < sqrt(eps(1.)) && 
      norm(Q*B2-B1) < sqrt(eps(1.)) &&
      norm(C2*Z-C1) < sqrt(eps(1.)) && 
      istriu(A1) && istriu(E1) &&
      ν == [3] && blkdims == (0,3)   

@time A1, E1, B1, C1, Q, Z, ν, blkdims  = fiblkdiag(A,E,B,C,fast = fast, finite_infinite = true, trinv=true)
@test norm(Q*A1*Z-A2) < sqrt(eps(1.)) &&
      norm(Q*E1*Z-E2) < sqrt(eps(1.)) && 
      norm(Q*B1-B2) < sqrt(eps(1.)) &&
      norm(C1*Z-C2) < sqrt(eps(1.)) && 
      istriu(A1) && istriu(E1) &&
      ν == [3] && blkdims == (0,3)   
      
@time A1, E1, B1, C1, Q, Z, ν, blkdims  = fiblkdiag(A,E,B,C,fast = fast, finite_infinite = false)
@test norm(Q*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q*E2*Z-E1) < sqrt(eps(1.)) && 
      norm(Q*B2-B1) < sqrt(eps(1.)) &&
      norm(C2*Z-C1) < sqrt(eps(1.)) && 
      istriu(A1) && istriu(E1) &&
      ν == [3] && blkdims == (3,0)  
      

A2 = rand(3,3); E2 = triu(rand(Ty,3,3),1); B2 = rand(Ty,3,2); C2 = rand(Ty,3,3);
A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2);


@time A1, E1, B1, C1, Q, Z, ν, blkdims  = fiblkdiag(A,E,B,C,fast = fast, finite_infinite = true)
@test norm(Q*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q*E2*Z-E1) < sqrt(eps(1.)) && 
      norm(Q*B2-B1) < sqrt(eps(1.)) &&
      norm(C2*Z-C1) < sqrt(eps(1.)) && 
      isqtriu(A1) && istriu(E1) &&
      ν == [1] && blkdims == (2,1)   
      
@time A1, E1, B1, C1, Q, Z, ν, blkdims  = fiblkdiag(A,E,B,C,fast = fast, finite_infinite = false)
@test norm(Q*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q*E2*Z-E1) < sqrt(eps(1.)) && 
      norm(Q*B2-B1) < sqrt(eps(1.)) &&
      norm(C2*Z-C1) < sqrt(eps(1.)) && 
      isqtriu(A1) && istriu(E1) &&
      ν == [1] && blkdims == (1,2)   
      
Q2 = qr(rand(Ty,6,6)).Q;
A2 = Q2*[zeros(Ty,3,6); zeros(Ty,3,3) rand(Ty,3,3)]; E2 = Q2*[rand(3,6); zeros(Ty,3,6) ]; 
B2 = rand(Ty,6,2); C2 = rand(Ty,3,6);

A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2);

@time A1, E1, B1, C1, Q, Z, ν, blkdims  = fiblkdiag(A,E,B,C,fast = fast, finite_infinite = true)
@test norm(Q*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q*E2*Z-E1) < sqrt(eps(1.)) && 
      norm(Q*B2-B1) < sqrt(eps(1.)) &&
      norm(C2*Z-C1) < sqrt(eps(1.)) && 
      isqtriu(A1) && istriu(E1) &&
      ν == [3] && blkdims == (3,3)   
      
@time A1, E1, B1, C1, Q, Z, ν, blkdims  = fiblkdiag(A,E,B,C,fast = fast, finite_infinite = false)
@test norm(Q*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q*E2*Z-E1) < sqrt(eps(1.)) && 
      norm(Q*B2-B1) < sqrt(eps(1.)) &&
      norm(C2*Z-C1) < sqrt(eps(1.)) && 
      isqtriu(A1) && istriu(E1) &&
      ν == [3] && blkdims == (3,3)   
      

@time A1, E1, B1, C1, Q, Z, ν, blkdims  = fiblkdiag(A,E,B,C,fast = fast, finite_infinite = true, trinv = true)
@test norm(Q*A1*Z-A2) < sqrt(eps(1.)) &&
      norm(Q*E1*Z-E2) < sqrt(eps(1.)) && 
      norm(Q*B1-B2) < sqrt(eps(1.)) &&
      norm(C1*Z-C2) < sqrt(eps(1.)) && 
      isqtriu(A1) && istriu(E1) &&
      ν == [3] && blkdims == (3,3)   
      
@time A1, E1, B1, C1, Q, Z, ν, blkdims  = fiblkdiag(A,E,B,C,fast = fast, finite_infinite = false, trinv = true)
@test norm(Q*A1*Z-A2) < sqrt(eps(1.)) &&
      norm(Q*E1*Z-E2) < sqrt(eps(1.)) && 
      norm(Q*B1-B2) < sqrt(eps(1.)) &&
      norm(C1*Z-C2) < sqrt(eps(1.)) && 
      isqtriu(A1) && istriu(E1) &&
      ν == [3] && blkdims == (3,3)   
      
     
end
end
end # fiblkdiag


@testset "gsblkdiag" begin

A2 = zeros(0,0); E2 = zeros(0,0); 
A = copy(A2); E = copy(E2); 

@time A1, E1, B1, C1, Q, Z, ν, blkdims, sep  = gsblkdiag(A,E,missing,missing)
@test norm(Q*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q*E2*Z-E1) < sqrt(eps(1.)) && 
      ν == [] && blkdims == (0,0,0)


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
    0     0     1     0     0     0     0     0     0
    0     0     0     1     0     0     0     0     0
    0     0     0     0     1     0     0     0     0
    0     0     0     0     0     0     0     0     0
    0     0     0     0     0     0     1     0     0
    0     0     0     0     0     0     0     1     0];


B2 = rand(9,2); C2 = rand(3,9);
A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2);

@time A1, E1, B1, C1, Q, Z, ν, blkdims, sep  = gsblkdiag(A,E,B,C,finite_infinite=false, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q*E2*Z-E1) < sqrt(eps(1.)) && 
      norm(Q*B2-B1) < sqrt(eps(1.)) &&
      norm(C2*Z-C1) < sqrt(eps(1.)) && 
      isqtriu(A1) && istriu(A1) && istriu(E1) &&
      ν == [2, 2, 2, 1, 1, 1] && blkdims == (9,0,0)  

A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2);
@time A1, E1, B1, C1, Q, Z, ν, blkdims, sep  = gsblkdiag(A,E,B,C,finite_infinite=true, atol1 = 1.e-7, atol2 = 1.e-7)
@test norm(Q*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q*E2*Z-E1) < sqrt(eps(1.)) && 
      norm(Q*B2-B1) < sqrt(eps(1.)) &&
      norm(C2*Z-C1) < sqrt(eps(1.)) && 
      isqtriu(A1) && istriu(A1) && istriu(E1) &&
      ν == [1, 1, 1, 2, 2, 2] && blkdims == (0,0,9)   

fast = true; Ty = Float64; Ty =  Complex{Float64};    

for fast in (true, false)

for Ty in (Float64, Complex{Float64})

A2 = rand(Ty,3,3); E2 = zeros(Ty,3,3); B2 = rand(Ty,3,2); C2 = rand(Ty,3,3);

A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2);

@time A1, E1, B1, C1, Q, Z, ν, blkdims, sep  = gsblkdiag(A,E,B,C,fast = fast, finite_infinite=true)
@test norm(Q*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q*E2*Z-E1) < sqrt(eps(1.)) && 
      norm(Q*B2-B1) < sqrt(eps(1.)) &&
      norm(C2*Z-C1) < sqrt(eps(1.)) && 
      istriu(A1) && istriu(E1) &&
      ν == [3] && blkdims == (0,0,3)   
      
@time A1, E1, B1, C1, Q, Z, ν, blkdims, sep  = gsblkdiag(A,E,B,C,fast = fast, finite_infinite=false)
@test norm(Q*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q*E2*Z-E1) < sqrt(eps(1.)) && 
      norm(Q*B2-B1) < sqrt(eps(1.)) &&
      norm(C2*Z-C1) < sqrt(eps(1.)) && 
      istriu(A1) && istriu(E1) &&
      ν == [3] && blkdims == (3,0,0)   

@time A1, E1, B1, C1, Q, Z, ν, blkdims, sep  = gsblkdiag(A,E,B,C,fast = fast, finite_infinite = true, trinv=true)
@test norm(Q*A1*Z-A2) < sqrt(eps(1.)) &&
      norm(Q*E1*Z-E2) < sqrt(eps(1.)) && 
      norm(Q*B1-B2) < sqrt(eps(1.)) &&
      norm(C1*Z-C2) < sqrt(eps(1.)) && 
      istriu(A1) && istriu(E1) &&
      ν == [3] && blkdims == (0,0,3)   

A2 = rand(3,3); E2 = triu(rand(Ty,3,3),1); B2 = rand(Ty,3,2); C2 = rand(Ty,3,3);   
F2, = saloc(A2,E2,B2; evals = [-.1;2])
A2c = copy(A2+B2*F2); A = copy(A2c); E = copy(E2); B = copy(B2); C = copy(C2); 

@time A1, E1, B1, C1, Q, Z, ν, blkdims, sep = gsblkdiag(A,E,B,C,fast = fast, finite_infinite = true, stable_unstable=true)
@test norm(Q*A2c*Z-A1) < sqrt(eps(1.)) &&
      norm(Q*E2*Z-E1) < sqrt(eps(1.)) && 
      norm(Q*B2-B1) < sqrt(eps(1.)) &&
      norm(C2*Z-C1) < sqrt(eps(1.)) && 
      isqtriu(A1) && istriu(E1) &&
      ν == [1] && blkdims == (1,1,1) 


A2 = rand(3,3); E2 = triu(rand(Ty,3,3),1); B2 = rand(Ty,3,2); C2 = rand(Ty,3,3);   
F2, = saloc(A2,E2,B2; evals = [-.1;-2])
A2c = copy(A2+B2*F2); A = copy(A2c); E = copy(E2); B = copy(B2); C = copy(C2); 

@time A1, E1, B1, C1, Q, Z, ν, blkdims, sep = gsblkdiag(A,E,B,C,fast = fast, finite_infinite = true, stable_unstable=true)
@test norm(Q*A2c*Z-A1) < sqrt(eps(1.)) &&
      norm(Q*E2*Z-E1) < sqrt(eps(1.)) && 
      norm(Q*B2-B1) < sqrt(eps(1.)) &&
      norm(C2*Z-C1) < sqrt(eps(1.)) && 
      isqtriu(A1) && istriu(E1) &&
      ν == [1] && blkdims == (2,0,1) 

@time A1, E1, B1, C1, Q, Z, ν, blkdims, sep = gsblkdiag(A,E,B,C,fast = fast, finite_infinite = true, stable_unstable=true, trinv=true)
@test norm(Q*A1*Z-A2c) < sqrt(eps(1.)) &&
      norm(Q*E1*Z-E2) < sqrt(eps(1.)) && 
      norm(Q*B1-B2) < sqrt(eps(1.)) &&
      norm(C1*Z-C2) < sqrt(eps(1.)) && 
      isqtriu(A1) && istriu(E1) &&
      ν == [1] && blkdims == (2,0,1) 

F2, = saloc(A2,E2,B2; evals = [-.1;-.2])
A2c = copy(A2+B2*F2); A = copy(A2c); E = copy(E2); B = copy(B2); C = copy(C2); 
@time A1, E1, B1, C1, Q, Z, ν, blkdims, sep = gsblkdiag(A,E,B,C,fast = fast, finite_infinite = false,stable_unstable=true)
@test norm(Q*A2c*Z-A1) < sqrt(eps(1.)) &&
      norm(Q*E2*Z-E1) < sqrt(eps(1.)) && 
      norm(Q*B2-B1) < sqrt(eps(1.)) &&
      norm(C2*Z-C1) < sqrt(eps(1.)) && 
      isqtriu(A1) && istriu(E1) &&
      ν == [1] && blkdims == (1,2,0) 

F2, = saloc(A2,E2,B2; evals = [.1;.2])
A2c = copy(A2+B2*F2); A = copy(A2c); E = copy(E2); B = copy(B2); C = copy(C2); 
@time A1, E1, B1, C1, Q, Z, ν, blkdims, sep = gsblkdiag(A,E,B,C,fast = fast, finite_infinite = false,stable_unstable=true)
@test norm(Q*A2c*Z-A1) < sqrt(eps(1.)) &&
      norm(Q*E2*Z-E1) < sqrt(eps(1.)) && 
      norm(Q*B2-B1) < sqrt(eps(1.)) &&
      norm(C2*Z-C1) < sqrt(eps(1.)) && 
      isqtriu(A1) && istriu(E1) &&
      ν == [1] && blkdims == (1,0,2) 

F2, = saloc(A2,E2,B2; evals = [.1;.2])
A2c = copy(A2+B2*F2); A = copy(A2c); E = copy(E2); B = copy(B2); C = copy(C2); 
@time A1, E1, B1, C1, Q, Z, ν, blkdims, sep = gsblkdiag(A,E,B,C,fast = fast, finite_infinite = false,stable_unstable=true,disc=true)
@test norm(Q*A2c*Z-A1) < sqrt(eps(1.)) &&
      norm(Q*E2*Z-E1) < sqrt(eps(1.)) && 
      norm(Q*B2-B1) < sqrt(eps(1.)) &&
      norm(C2*Z-C1) < sqrt(eps(1.)) && 
      isqtriu(A1) && istriu(E1) &&
      ν == [1] && blkdims == (1,2,0) 

F2, = saloc(A2,E2,B2; evals = [.1;.2])
A2c = copy(A2+B2*F2); A = copy(A2c); E = copy(E2); B = copy(B2); C = copy(C2); 
@time A1, E1, B1, C1, Q, Z, ν, blkdims, sep = gsblkdiag(A,E,B,C,fast = fast, finite_infinite = false,stable_unstable=false,disc=true)
@test norm(Q*A2c*Z-A1) < sqrt(eps(1.)) &&
      norm(Q*E2*Z-E1) < sqrt(eps(1.)) && 
      norm(Q*B2-B1) < sqrt(eps(1.)) &&
      norm(C2*Z-C1) < sqrt(eps(1.)) && 
      isqtriu(A1) && istriu(E1) &&
      ν == [1] && blkdims == (1,0,2) 

F2, = saloc(A2,E2,B2; evals = [.1;2])
A2c = copy(A2+B2*F2); A = copy(A2c); E = copy(E2); B = copy(B2); C = copy(C2); 
@time A1, E1, B1, C1, Q, Z, ν, blkdims, sep = gsblkdiag(A,E,B,C,fast = fast, finite_infinite = false,stable_unstable=true,disc=true)
@test norm(Q*A2c*Z-A1) < sqrt(eps(1.)) &&
      norm(Q*E2*Z-E1) < sqrt(eps(1.)) && 
      norm(Q*B2-B1) < sqrt(eps(1.)) &&
      norm(C2*Z-C1) < sqrt(eps(1.)) && 
      isqtriu(A1) && istriu(E1) &&
      ν == [1] && blkdims == (1,1,1) 

A2 = zeros(3,3); E2 = rand(3,3); B2 = rand(Ty,3,3); C2 = rand(Ty,3,3);   

F2, = saloc(A2,E2,B2; evals = [-.1,-2,2])
A2c = copy(A2+B2*F2); A = copy(A2c); E = copy(E2); B = copy(B2); C = copy(C2); 
@time A1, E1, B1, C1, Q, Z, ν, blkdims, sep = gsblkdiag(A,E,B,C,fast = fast, finite_infinite = true, stable_unstable=true)
@test norm(Q*A2c*Z-A1) < sqrt(eps(1.)) &&
      norm(Q*E2*Z-E1) < sqrt(eps(1.)) && 
      norm(Q*B2-B1) < sqrt(eps(1.)) &&
      norm(C2*Z-C1) < sqrt(eps(1.)) && 
      isqtriu(A1) && istriu(E1) &&
      ν == [] && blkdims == (2, 1, 0)   

F2, = saloc(A2,E2,B2; evals = [-.1,-2,2])
A2c = copy(A2+B2*F2); A = copy(A2c); E = copy(E2); B = copy(B2); C = copy(C2); 
@time A1, E1, B1, C1, Q, Z, ν, blkdims, sep = gsblkdiag(A,E,B,C,fast = fast, finite_infinite = true, stable_unstable=false)
@test norm(Q*A2c*Z-A1) < sqrt(eps(1.)) &&
      norm(Q*E2*Z-E1) < sqrt(eps(1.)) && 
      norm(Q*B2-B1) < sqrt(eps(1.)) &&
      norm(C2*Z-C1) < sqrt(eps(1.)) && 
      isqtriu(A1) && istriu(E1) &&
      ν == [] && blkdims == (1, 2, 0)   

F2, = saloc(A2,E2,B2; evals = [-.1,-2,-0.5])
A2c = copy(A2+B2*F2); A = copy(A2c); E = copy(E2); B = copy(B2); C = copy(C2); 
@time A1, E1, B1, C1, Q, Z, ν, blkdims, sep = gsblkdiag(A,E,B,C,smarg = -0.6, fast = fast, finite_infinite = true, stable_unstable=false)
@test norm(Q*A2c*Z-A1) < sqrt(eps(1.)) &&
      norm(Q*E2*Z-E1) < sqrt(eps(1.)) && 
      norm(Q*B2-B1) < sqrt(eps(1.)) &&
      norm(C2*Z-C1) < sqrt(eps(1.)) && 
      isqtriu(A1) && istriu(E1) &&
      ν == [] && blkdims == (2, 1, 0)   

F2, = saloc(A2,E2,B2; evals = [.1,-.2,-0.5])
A2c = copy(A2+B2*F2); A = copy(A2c); E = copy(E2); B = copy(B2); C = copy(C2); 
@time A1, E1, B1, C1, Q, Z, ν, blkdims, sep = gsblkdiag(A,E,B,C,smarg = 0.3, disc = true, fast = fast, 
           finite_infinite = true, stable_unstable=false)
@test norm(Q*A2c*Z-A1) < sqrt(eps(1.)) &&
      norm(Q*E2*Z-E1) < sqrt(eps(1.)) && 
      norm(Q*B2-B1) < sqrt(eps(1.)) &&
      norm(C2*Z-C1) < sqrt(eps(1.)) && 
      isqtriu(A1) && istriu(E1) &&
      ν == [] && blkdims == (1, 2, 0)   
    
   

end
end
end # gsblkdiag

Ty = Float64; Ty =  Complex{Float64};    

for Ty in (Float64, Complex{Float64})

A2 = rand(3,3); B2 = rand(Ty,3,2); C2 = rand(Ty,3,3);   
F2, = saloc(A2,B2; evals = [-.1,-1,2])
A2c = copy(A2+B2*F2); A = copy(A2c); B = copy(B2); C = copy(C2); 

@time A1, B1, C1, Q, Z, blkdims, sep  = ssblkdiag(A,B,C,stable_unstable = true);
@test norm(Q*A2c*Z-A1) < sqrt(eps(1.)) &&
      norm(Q*B2-B1) < sqrt(eps(1.)) &&
      norm(C2*Z-C1) < sqrt(eps(1.)) && 
      istriu(A1) && blkdims == (2,1)   
      
F2, = saloc(A2,B2; evals = [-.1,-1,2])
A2c = copy(A2+B2*F2); A = copy(A2c); B = copy(B2); C = copy(C2); 

@time A1, B1, C1, Q, Z, blkdims, sep  = ssblkdiag(A,B,C,stable_unstable = false);
@test norm(Q*A2c*Z-A1) < sqrt(eps(1.)) &&
      norm(Q*B2-B1) < sqrt(eps(1.)) &&
      norm(C2*Z-C1) < sqrt(eps(1.)) && 
      istriu(A1) && blkdims == (1,2)   

F2, = saloc(A2,B2; evals = [.1,.2,2])
A2c = copy(A2+B2*F2); A = copy(A2c); B = copy(B2); C = copy(C2); 

@time A1, B1, C1, Q, Z, blkdims, sep  = ssblkdiag(A,B,C,stable_unstable = true, disc = true);
@test norm(Q*A2c*Z-A1) < sqrt(eps(1.)) &&
      norm(Q*B2-B1) < sqrt(eps(1.)) &&
      norm(C2*Z-C1) < sqrt(eps(1.)) && 
      istriu(A1) && blkdims == (2,1)   
      
F2, = saloc(A2,B2; evals = [.1,.2,2])
A2c = copy(A2+B2*F2); A = copy(A2c); B = copy(B2); C = copy(C2); 

@time A1, B1, C1, Q, Z, blkdims, sep  = ssblkdiag(A,B,C,stable_unstable = false, disc = true);
@test norm(Q*A2c*Z-A1) < sqrt(eps(1.)) &&
      norm(Q*B2-B1) < sqrt(eps(1.)) &&
      norm(C2*Z-C1) < sqrt(eps(1.)) && 
      istriu(A1) && blkdims == (1,2)   
       
end # ssblkdiag


end #test

end #module





