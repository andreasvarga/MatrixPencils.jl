module Test_regular

using LinearAlgebra
using MatrixPencils
using Test


@testset "Regular Matrix Pencils Utilities" begin


@testset "isregular" begin

M = zeros(0,0); N = zeros(0,0);
@test isregular(M, N) 


M = zeros(3,0); N = zeros(3,0);
@test !isregular(M, N) 

M = zeros(0,3); N = zeros(0,3);
@test !isregular(M, N) 

M = zeros(1,1); N = ones(1,1);
@test isregular(M, N) 

N = zeros(1,1); M = ones(1,1);
@test isregular(M, N) 

M = [1 0;0 1]; N = [0 1; 0 0]; 
@test isregular(M, N) 

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
@test !isregular(M, N) 

Ty = Float64
for Ty in (Float64, Complex{Float64})

    abstol = sqrt(eps(one(real(Ty))))
    
    # given structure 
    mr = 2; nr = 3; ni = 2; nf = 10; ml = 5; nl = 4;
    mM = mr+ni+nf+ml;
    nM = nr+ni+nf+nl;
    M = [rand(Ty,mr,nM); 
         zeros(Ty,ni,nr) Matrix{Ty}(I,ni,ni) rand(Ty,ni,nf+nl); 
         zeros(Ty,nf,nr+ni) rand(Ty,nf,nf+nl);
         zeros(ml,nr+ni+nf) rand(Ty,ml,nl)]
    N = [rand(Ty,mr,nM); 
         zeros(Ty,ni,nr) diagm(1 => ones(Ty,ni-1)) rand(Ty,ni,nf+nl); 
         zeros(Ty,nf,nr+ni) rand(Ty,nf,nf+nl);
         zeros(ml,nr+ni+nf) rand(Ty,ml,nl)]
    Q = qr(rand(Ty,mM,mM)).Q;
    Z = qr(rand(Ty,nM,nM)).Q; 
    M = Q*M*Z;
    N = Q*N*Z;
    
    @test !isregular(M, N, atol1 = abstol, atol2 = abstol)
    
   # given structure 
   mr = 0; nr = 0; ni = 4; nf = 10; ml = 0; nl = 0;
   mM = mr+ni+nf+ml;
   nM = nr+ni+nf+nl;
   M = [rand(Ty,mr,nM); 
        zeros(Ty,ni,nr) Matrix{Ty}(I,ni,ni) rand(Ty,ni,nf+nl); 
        zeros(Ty,nf,nr+ni) rand(Ty,nf,nf+nl);
        zeros(ml,nr+ni+nf) rand(Ty,ml,nl)]
   N = [rand(Ty,mr,nM); 
        zeros(Ty,ni,nr) diagm(1 => ones(Ty,ni-1)) rand(Ty,ni,nf+nl); 
        zeros(Ty,nf,nr+ni) rand(Ty,nf,nf+nl);
        zeros(ml,nr+ni+nf) rand(Ty,ml,nl)]
   Q = qr(rand(Ty,mM,mM)).Q;
   Z = qr(rand(Ty,nM,nM)).Q; 
   M = Q*M*Z;
   N = Q*N*Z;
   
   @test isregular(M, N, atol1 = abstol, atol2 = abstol)
   
end
    
end # isregular

@testset "fisplit" begin
A2 = zeros(0,0); E2 = zeros(0,0); C2 = zeros(0,0); B2 = missing;
A = copy(A2); E = copy(E2); C = copy(C2); B = missing;

@time A1, E1, B1, C1, Q, Z, ν, nf, ni  = fisplit(A,E,B,C)
@test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B1) < sqrt(eps(1.))) && 
      (ismissing(C) || norm(C2*Z-C1)  < sqrt(eps(1.))) && 
      ν == [] && nf == 0 && ni == 0

A2 = zeros(0,0); E2 = zeros(0,0); B2 = zeros(0,2); C2 = missing;
A = copy(A2); E = copy(E2); B = copy(B2); C = missing;

@time A1, E1, B1, C1, Q, Z, ν, nf, ni  = fisplit(A,E,B,C)
@test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B1) < sqrt(eps(1.))) && 
      (ismissing(C) || norm(C2*Z-C1)  < sqrt(eps(1.))) && 
      ν == [] && nf == 0 && ni == 0 

fast = true; finite_infinite = true; Ty = Float64      
for finite_infinite in (true, false)

for fast in (true, false)

for Ty in (Float64, Complex{Float64})

A2 = rand(Ty,3,3); E2 = zeros(Ty,3,3); B2 = zeros(Ty,3,2); C2 = zeros(4,3);
A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2); 

@time A1, E1, B1, C1, Q, Z, ν, nf, ni  = fisplit(A,E,B,C,fast = fast, finite_infinite = finite_infinite)
@test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B1) < sqrt(eps(1.))) && 
      (ismissing(C) || norm(C2*Z-C1)  < sqrt(eps(1.))) && 
      ν == [3] && nf == 0 && ni == 3   
      
A2 = rand(3,3); E2 = triu(rand(Ty,3,3),1); B2 = zeros(3,2); C2 = zeros(4,3);
A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2); 

@time A1, E1, B1, C1, Q, Z, ν, nf, ni  = fisplit(A,E,B,C,fast = fast, finite_infinite = finite_infinite)
@test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B1) < sqrt(eps(1.))) && 
      (ismissing(C) || norm(C2*Z-C1)  < sqrt(eps(1.))) && 
      ν == [1] && nf == 2 && ni == 1   
      

A2 = zeros(3,3); E2 = rand(3,3); B2 = zeros(Ty,3,2); C2 = zeros(4,3);
A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2); 

@time A1, E1, B1, C1, Q, Z, ν, nf, ni  = fisplit(A,E,B,C,fast = fast, finite_infinite = finite_infinite)
@test norm(Q'*A2*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2*Z-E1) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(Q'*B2-B1) < sqrt(eps(1.))) && 
      (ismissing(C) || norm(C2*Z-C1)  < sqrt(eps(1.))) && 
      ν == [] && nf == 3 && ni == 0   
      
A2 = rand(3,3); E2 = rand(3,3); B2 = zeros(Ty,3,2); C2 = zeros(4,3);
A = copy(A2); E = copy(E2); B = copy(B2); C = copy(C2); 

@time A1, E1, C1, B1, Q, Z, ν, nf, ni  = fisplit(A',E',C',B',fast = fast, finite_infinite = finite_infinite)
@test norm(Q'*A2'*Z-A1) < sqrt(eps(1.)) &&
      norm(Q'*E2'*Z-E1) < sqrt(eps(1.)) &&
      (ismissing(B) || norm(B2'*Z-B1) < sqrt(eps(1.))) && 
      (ismissing(C) || norm(Q'*C2'-C1)   < sqrt(eps(1.))) && 
      ν == [] && nf == 3 && ni == 0     
      
end
end
end
end # fisplit

end # regular testset
end # module