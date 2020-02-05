module Test_klfapps

using LinearAlgebra
using MatrixPencils
using Test


@testset "Matrix Pencils Structure Applications" begin

M = zeros(0,0); N = zeros(0,0);
@time info  = pkstruct(M, N, fast = false)
@test info.rki == [] && info.lki == [] && info.id == [] && info.nf == 0


M = zeros(3,0); N = zeros(3,0);
@time info = pkstruct(M, N, fast = false)
@test info.rki == [] && info.lki == [3] && info.id == [] && info.nf == 0

M = zeros(0,3); N = zeros(0,3);
@time info = pkstruct(M, N, fast = false)
@test info.rki == [3] && info.lki == [] && info.id == [] && info.nf == 0

M = zeros(1,1); N = ones(1,1);
@time info = pkstruct(M, N, fast = false)
@test info.rki == [] && info.lki == [] && info.id == [] && info.nf == 1

M = ones(1,1); N = zeros(1,1); 
@time info = pkstruct(M, N, fast = false)
@test info.rki == [] && info.lki == [] && info.id == [1] && info.nf == 0

M = [1 0;0 1]; N = [0 1; 0 0]; 
@time info = pkstruct(M, N)
@test info.rki == [] && info.lki == [] && info.id == [0, 1] && info.nf == 0

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

@time info = pkstruct(M, N)
@test info.rki == [1] && info.lki == [1] && info.id == [1] && info.nf == 3

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
@time info = pkstruct(M, N, fast = true, atol1 = 1.e-7, atol2 = 1.e-7)
@test info.rki == [2, 1, 1] && info.lki == [1, 0, 0, 1] && info.id == [1,1] && info.nf == 3

#for Ty in (Float64, Float32, Complex{Float64},  Complex{Float32})
Ty = Float64
Ty = Complex{Float64}
for Ty in (Float64, Complex{Float64})

abstol = sqrt(eps(one(real(Ty))))

# given structure 
mr = 2; nr = 3; ni = 2; nf = 10; ml = 5; nl = 4;
mM = mr+ni+nf+ml;
nM = nr+ni+nf+nl;
M = [rand(Ty,mr,nM); 
     zeros(Ty,ni,nr) one(Ty)*I(ni) rand(Ty,ni,nf+nl); 
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

@time info = pkstruct(M, N, fast = true, atol1 = abstol, atol2 = abstol)
@test info.rki == [0, 0, 1] && info.lki == [0, 0, 0, 0, 1] && info.id == [0, 1] && info.nf == 10

@time info = pkstruct(M, N, fast = false, atol1 = abstol, atol2 = abstol)
@test info.rki == [0, 0, 1] && info.lki == [0, 0, 0, 0, 1] && info.id == [0, 1] && info.nf == 10

end
end

@testset "Matrix Pencils Zeros Applications" begin

M = zeros(0,0); N = zeros(0,0);
@time val  = pzeros(M, N, fast = false)
@test val == []


M = zeros(3,0); N = zeros(3,0);
@time val  = pzeros(M, N, fast = false)
@test val == []

M = zeros(0,3); N = zeros(0,3);
@time val  = pzeros(M, N, fast = false)
@test val == []

M = zeros(1,1); N = ones(1,1);
@time val  = pzeros(M, N, fast = false)
@test val == [0]

N = zeros(1,1); M = ones(1,1);
@time val  = pzeros(M, N, fast = false)
@test val == []

M = [1 0;0 1]; N = [0 1; 0 0]; 

@time val  = pzeros(M, N)
@test val == [Inf]

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

@time val  = pzeros(M, N)
@test length(val) == 3 &&
      length(filter(y-> y == true,val .== Inf)) == 0 &&
      length(filter(y-> y == true,val .< Inf)) == 3

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
@time val  = pzeros(M, N, fast = true, atol1 = 1.e-7, atol2 = 1.e-7)
@test length(val) == 4 &&
      length(filter(y-> y == true,val .== Inf)) == 1 &&
      length(filter(y-> y == true,abs.(val) .< Inf)) == 3

#for Ty in (Float64, Float32, Complex{Float64},  Complex{Float32})
for Ty in (Float64, Complex{Float64})

abstol = sqrt(eps(one(real(Ty))))

# given structure 
mr = 2; nr = 3; ni = 2; nf = 10; ml = 5; nl = 4;
mM = mr+ni+nf+ml;
nM = nr+ni+nf+nl;
M = [rand(Ty,mr,nM); 
     zeros(Ty,ni,nr) one(Ty)*I(ni) rand(Ty,ni,nf+nl); 
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

@time val  = pzeros(M, N, fast = true, atol1 = abstol, atol2 = abstol)
@test length(val) == ni+nf-1 &&
      length(filter(y-> y == true,isinf.(val))) == ni-1 &&
      length(filter(y-> y == true,abs.(val) .< Inf)) == nf

@time val  = pzeros(M, N, fast = false, atol1 = abstol, atol2 = abstol)
@test length(val) == ni+nf-1 &&
      length(filter(y-> y == true,isinf.(val))) == ni-1 &&
      length(filter(y-> y == true,abs.(val) .< Inf)) == nf

end
end


@testset "Matrix Pencils Eigenvalues Applications" begin

M = zeros(0,0); N = zeros(0,0);
@time val  = peigvals(M, N, fast = false)
@test val == []


M = zeros(3,0); N = zeros(3,0);
@time val  = peigvals(M, N, fast = false)
@test val == []

M = zeros(0,3); N = zeros(0,3);
@time val  = peigvals(M, N, fast = false)
@test val == []

M = zeros(1,1); N = ones(1,1);
@time val  = peigvals(M, N, fast = false)
@test val == [0]

N = zeros(1,1); M = ones(1,1);
@time val  = peigvals(M, N, fast = false)
@test val == [Inf]

M = [1 0;0 1]; N = [0 1; 0 0]; 

@time val  = peigvals(M, N)
@test val == [Inf;Inf]

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

@time val  = peigvals(M, N)
@test length(val) == 4 &&
      length(filter(y-> y == true,val .== Inf)) == 1 &&
      length(filter(y-> y == true,val .< Inf)) == 3

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
@time val  = peigvals(M, N, fast = true, atol1 = 1.e-7, atol2 = 1.e-7)
@test length(val) == 6 &&
      length(filter(y-> y == true,val .== Inf)) == 3 &&
      length(filter(y-> y == true,abs.(val) .< Inf)) == 3

#for Ty in (Float64, Float32, Complex{Float64},  Complex{Float32})
Ty = Float64
Ty = Complex{Float64}
for Ty in (Float64, Complex{Float64})

abstol = sqrt(eps(one(real(Ty))))

# given structure 
mr = 2; nr = 3; ni = 2; nf = 10; ml = 5; nl = 4;
mM = mr+ni+nf+ml;
nM = nr+ni+nf+nl;
M = [rand(Ty,mr,nM); 
     zeros(Ty,ni,nr) one(Ty)*I(ni) rand(Ty,ni,nf+nl); 
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

@time val  = peigvals(M, N, fast = true, atol1 = abstol, atol2 = abstol)
@test length(val) == ni+nf &&
      length(filter(y-> y == true,isinf.(val))) == ni &&
      length(filter(y-> y == true,abs.(val) .< Inf)) == nf

@time val  = peigvals(M, N, fast = false, atol1 = abstol, atol2 = abstol)
@test length(val) == ni+nf &&
      length(filter(y-> y == true,isinf.(val))) == ni &&
      length(filter(y-> y == true,abs.(val) .< Inf)) == nf

end
end

@testset "Matrix Pencils Zeros Applications" begin

M = zeros(0,0); N = zeros(0,0);
@time val  = pzeros(M, N, fast = false)
@test val == []


M = zeros(3,0); N = zeros(3,0);
@time val  = pzeros(M, N, fast = false)
@test val == []

M = zeros(0,3); N = zeros(0,3);
@time val  = pzeros(M, N, fast = false)
@test val == []

M = zeros(1,1); N = ones(1,1);
@time val  = pzeros(M, N, fast = false)
@test val == [0]

N = zeros(1,1); M = ones(1,1);
@time val  = pzeros(M, N, fast = false)
@test val == []

M = [1 0;0 1]; N = [0 1; 0 0]; 

@time val  = pzeros(M, N)
@test val == [Inf]

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

@time val  = pzeros(M, N)
@test length(val) == 3 &&
      length(filter(y-> y == true,val .== Inf)) == 0 &&
      length(filter(y-> y == true,val .< Inf)) == 3

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
@time val  = pzeros(M, N, fast = true, atol1 = 1.e-7, atol2 = 1.e-7)
@test length(val) == 4 &&
      length(filter(y-> y == true,val .== Inf)) == 1 &&
      length(filter(y-> y == true,abs.(val) .< Inf)) == 3

#for Ty in (Float64, Float32, Complex{Float64},  Complex{Float32})
for Ty in (Float64, Complex{Float64})

abstol = sqrt(eps(one(real(Ty))))

# given structure 
mr = 2; nr = 3; ni = 2; nf = 10; ml = 5; nl = 4;
mM = mr+ni+nf+ml;
nM = nr+ni+nf+nl;
M = [rand(Ty,mr,nM); 
     zeros(Ty,ni,nr) one(Ty)*I(ni) rand(Ty,ni,nf+nl); 
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

@time val  = pzeros(M, N, fast = true, atol1 = abstol, atol2 = abstol)
@test length(val) == ni+nf-1 &&
      length(filter(y-> y == true,isinf.(val))) == ni-1 &&
      length(filter(y-> y == true,abs.(val) .< Inf)) == nf

@time val  = pzeros(M, N, fast = false, atol1 = abstol, atol2 = abstol)
@test length(val) == ni+nf-1 &&
      length(filter(y-> y == true,isinf.(val))) == ni-1 &&
      length(filter(y-> y == true,abs.(val) .< Inf)) == nf

end
end



@testset "Matrix Pencils Rank Applications" begin

M = zeros(0,0); N = zeros(0,0);
@test prank(M, N, fast = false) == 0
@test prank(M, N, fast = true) == 0

M = zeros(3,0); N = zeros(3,0);
@test prank(M, N, fast = false) == 0
@test prank(M, N, fast = true) == 0

M = zeros(0,3); N = zeros(0,3);
@test prank(M, N, fast = false) == 0
@test prank(M, N, fast = true) == 0

M = zeros(1,1); N = ones(1,1);
@test prank(M, N, fast = false) == 1
@test prank(M, N, fast = true) == 1
@test prank(N, M, fast = false) == 1
@test prank(N, M, fast = true) == 1

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
@test prank(M, N, fast = false) == 4
@test prank(M, N, fast = true) == 4

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

@test prank(M, N, fast = false) == 12
@test prank(M, N, fast = true) == 12

Ty = Float64
for Ty in (Float64, Float32, Complex{Float64},  Complex{Float32})

abstol = sqrt(eps(one(real(Ty))))

# full row rank
r = 5; n = 20;
M = rand(Ty,r,n)
N = rand(Ty,r,n)

@time prnk = prank(M, N, fast = false)
@test prnk == r

@time prnk = prank(M, N, fast = true)
@test prnk == r

# full column rank
r = 5; m = 20;
M = rand(Ty,m,r)
N = rand(Ty,m,r)

@time prnk = prank(M, N, fast = false)
@test prnk == r

@time prnk = prank(M, N, fast = true)
@test prnk == r

# given rank
r = 5; m = 20; n = 30;
Q = rand(Ty,m,m);
Z = rand(Ty,n,n); 
M = Q*[ rand(Ty,r,r) zeros(Ty,r,n-r); zeros(Ty,m-r,n) ]*Z;
N = Q*[ rand(Ty,r,r) zeros(Ty,r,n-r); zeros(Ty,m-r,n) ]*Z;

@time prnk = prank(M, N, atol1 = 100*eps(opnorm(M,1)), atol2 = 100*eps(opnorm(N,1)), fast = false)
@test prnk == r

@time prnk = prank(M, N, atol1 = 100*eps(opnorm(M,1)), atol2 = 100*eps(opnorm(N,1)), fast = false)
@test prnk == r

end

end


@testset "Regular Matrix Pencils Applications" begin

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
         zeros(Ty,ni,nr) one(Ty)*I(ni) rand(Ty,ni,nf+nl); 
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
        zeros(Ty,ni,nr) one(Ty)*I(ni) rand(Ty,ni,nf+nl); 
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
    
end

end