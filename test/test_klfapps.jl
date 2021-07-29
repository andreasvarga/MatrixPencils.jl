module Test_klfapps

using Random
using LinearAlgebra
using MatrixPencils
using Test

@testset "Matrix Pencils Applications" begin

Random.seed!(2351);

@testset "Matrix Pencils Structure Applications" begin

fast = true
for fast in (false,true)

M = zeros(0,0); N = zeros(0,0);
@time info  = pkstruct(M, N, fast = fast)
@test info.rki == [] && info.lki == [] && info.id == [] && info.nf == 0 && info.nrank == 0


M = zeros(3,0); N = zeros(3,0);
@time info = pkstruct(M, N, fast = fast)
@test info.rki == [] && info.lki == [0, 0, 0] && info.id == [] && info.nf == 0 && info.nrank == 0

M = zeros(0,3); N = zeros(0,3);
@time info = pkstruct(M, N, fast = fast)
@test info.rki == [0, 0, 0] && info.lki == [] && info.id == [] && info.nf == 0 && info.nrank == 0

M = zeros(1,1); N = ones(1,1);
@time info = pkstruct(M, N, fast = fast)
@test info.rki == [] && info.lki == [] && info.id == [] && info.nf == 1 && info.nrank == 1

M = ones(1,1); N = zeros(1,1); 
@time info = pkstruct(M, N, fast = fast)
@test info.rki == [] && info.lki == [] && info.id == [1] && info.nf == 0 && info.nrank == 1

M = ones(1,1); N = nothing; 
@time info = pkstruct(M, N, fast = fast)
@test info.rki == [] && info.lki == [] && info.id == [] && info.nf == 0 && info.nrank == 1

M = [1 0;0 1]; N = [0 1; 0 0]; 
@time info = pkstruct(M, N)
@test info.rki == [] && info.lki == [] && info.id == [2] && info.nf == 0 && info.nrank == 2

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
rki, lki, id, nf, nrank = info
@test rki == [0] && lki == [0] && id == [1] && nf == 3 && nrank == 4

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
@test info.rki == [0, 0, 1, 2] && info.lki == [0, 3] && info.id == [1,2] && info.nf == 3 && info.nrank == 12


#for Ty in (Float64, Float32, Complex{Float64},  Complex{Float32})
Ty = Float64
Ty = Complex{Float64}
for Ty in (Float64, Complex{Float64})

#Ty = Complex{Float64}
abstol = sqrt(eps(one(real(Ty))))

# given structure 
mr = 2; nr = 3; ni = 4; nf = 10; ml = 5; nl = 4;
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

atol1 = 1.e-7; atol2 = 1.e-7;
@time info = pkstruct(M, N, fast = fast, atol1 = atol1, atol2 = atol2)
@test info.rki == [mr] && info.lki == [nl] && info.id == [ni] && info.nf == nf && info.nrank == mr+nl+ni+nf


end
end
end


Random.seed!(2351);

@testset "Matrix Pencils Eigenvalues Applications" begin

fast = true
for fast in (false,true)

M = zeros(0,0); N = zeros(0,0);
@time val, kinfo  = peigvals(M, N, fast = fast)
@test val == [] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == (Int64[], Int64[], Int64[], 0, 0)

M = zeros(3,0); N = zeros(3,0);
@time val, kinfo = peigvals(M, N, fast = fast)
@test val == [] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == (Int64[], [0, 0, 0], Int64[], 0, 0)

M = zeros(0,3); N = zeros(0,3);
@time val, kinfo = peigvals(M, N, fast = fast)
@test val == [] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == ([0, 0, 0], Int64[], Int64[], 0, 0)

M = zeros(1,1); N = ones(1,1);
@time val, kinfo = peigvals(M, N, fast = fast)
@test val == [0] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == (Int64[], Int64[], Int64[], 1, 1)

N = zeros(1,1); M = ones(1,1);
@time val, kinfo = peigvals(M, N, fast = fast)
@test val == [Inf] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == (Int64[], Int64[], [1], 0, 1)

M = ones(1,1); N = nothing; 
@time val, info = peigvals(M, N, fast = fast)
@test val == Float64[] && info.rki == [] && info.lki == [] && info.id == [] && info.nf == 0 && info.nrank == 1

M = [1 0;0 1]; N = [0 1; 0 0]; 

@time val, kinfo = peigvals(M, N)
@test val == [Inf;Inf] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == (Int64[], Int64[], [2], 0, 2)

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

@time val, kinfo = peigvals(M, N)
@test length(val) == 4 &&
      length(filter(y-> y == true,isinf.(val))) == 1 &&
      length(filter(y-> y == true,isfinite.(val))) == 3 &&
      (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == ([0], [0], [1], 3, 4)

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
@time val, kinfo = peigvals(M, N, fast = true, atol1 = 1.e-7, atol2 = 1.e-7)
@test length(val) == 6 &&
      length(filter(y-> y == true,isinf.(val))) == 3 &&
      length(filter(y-> y == true,isfinite.(val))) == 3 &&
      (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == ([0, 0, 1, 2], [0, 3], [1, 2], 3, 12)

#for Ty in (Float64, Float32, Complex{Float64},  Complex{Float32})
#Ty = Float64; fast = true;
for Ty in (Float64, Complex{Float64})

abstol = sqrt(eps(one(real(Ty))))

# given structure 
mr = 2; nr = 3; ni = 4; nf = 10; ml = 5; nl = 4;
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

@time val, kinfo = peigvals(M, N, fast = fast, atol1 = abstol, atol2 = abstol)
@test length(val) == ni+nf &&
      length(filter(y-> y == true,isinf.(val))) == ni &&
      length(filter(y-> y == true,isfinite.(val))) == nf && 
      (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == ([mr], [nl], [ni], nf, mr+nl+ni+nf)
end
end
end

Random.seed!(2151);

@testset "Matrix Pencils Zeros Applications" begin


fast = true
for fast in (false,true)


M = zeros(0,0); N = zeros(0,0);
@time val, iz, kinfo  = pzeros(M, N, fast = fast)
@test val == [] && iz == [] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == (Int64[], Int64[], Int64[], 0, 0)


M = zeros(3,0); N = zeros(3,0);
@time val, iz, kinfo  = pzeros(M, N, fast = fast)
@test val == [] && iz == [] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == (Int64[], [0, 0, 0], Int64[], 0, 0)


M = zeros(0,3); N = zeros(0,3);
@time val, iz, kinfo  = pzeros(M, N, fast = fast)
@test val == [] && iz == [] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == ([0, 0, 0], Int64[], Int64[], 0, 0)

M = zeros(1,1); N = ones(1,1);
@time val, iz, kinfo  = pzeros(M, N, fast = fast)
@test val == [0] && iz == [] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == (Int64[], Int64[], Int64[], 1, 1)

N = zeros(1,1); M = ones(1,1);
@time val, iz, kinfo  = pzeros(M, N, fast = fast)
@test val == [] && iz == [] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == (Int64[], Int64[], [1], 0, 1)

M = ones(1,1); N = nothing; 
@time val, iz, info = pzeros(M, N, fast = fast)
@test val == Float64[] && iz == [] && info.rki == [] && info.lki == [] && info.id == [] && info.nf == 0  && info.nrank == 1

M = [1 0;0 1]; N = [0 1; 0 0]; 
@time val, iz, kinfo  = pzeros(M, N, fast = fast)
@test val == [Inf] && iz == [1] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == (Int64[], Int64[], [2], 0, 2)

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

@time val, iz, kinfo   = pzeros(M, N, fast = fast)
@test length(val) == 3 &&
      length(filter(y-> y == true,isinf.(val))) == 0 &&
      length(filter(y-> y == true,isfinite.(val))) == 3 &&
      iz == [] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == ([0], [0], [1], 3, 4)

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
@time val, iz, kinfo   = pzeros(M, N, fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
@test length(val) == 4 &&
      length(filter(y-> y == true,isinf.(val))) == 1 &&
      length(filter(y-> y == true,isfinite.(val))) == 3 &&
      iz == [1] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == ([0, 0, 1, 2], [0, 3], [1, 2], 3, 12)


#Ty = Float64; fast = true;
for Ty in (Float64, Complex{Float64})

abstol = sqrt(eps(one(real(Ty))))

# given structure 
mr = 2; nr = mr+1; ni = 4; nf = 8; nl = 4; ml = nl+1; 
mM = mr+ni+nf+ml;
nM = nr+ni+nf+nl;
M2 = [rand(Ty,mr,nM); 
     zeros(Ty,ni,nr) Matrix{Ty}(I,ni,ni) rand(Ty,ni,nf+nl); 
     zeros(Ty,nf,nr+ni) rand(Ty,nf,nf+nl);
     zeros(ml,nr+ni+nf) rand(Ty,ml,nl)]
N2 = [rand(Ty,mr,nM); 
     zeros(Ty,ni,nr) diagm(1 => ones(Ty,ni-1)) rand(Ty,ni,nf+nl); 
     zeros(Ty,nf,nr+ni) rand(Ty,nf,nf+nl);
     zeros(ml,nr+ni+nf) rand(Ty,ml,nl)]
Q = qr(rand(Ty,mM,mM)).Q;
Z = qr(rand(Ty,nM,nM)).Q; 
M2 = Q*M2*Z;
N2 = Q*N2*Z;

M = copy(M2); N = copy(N2);
atol1 = 1.e-7; atol2 = 1.e-7; 
@time val, iz, kinfo  = pzeros(M, N, fast = fast, atol1 = atol1, atol2 = atol2)
@test length(val) == ni+nf-1 &&
      length(val[isinf.(val)]) == ni-1 &&
      length(val[isfinite.(val)]) == nf && iz == (ni > 1 ? [ni-1] : [ ]) && 
      (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == ([mr], [nl], ni > 0 ? [ni] : [], nf, mr+nl+ni+nf)

end
end
end # pzeros



@testset "Matrix Pencils Rank Applications" begin



fast = true
for fast in (false,true)


M = zeros(0,0); N = zeros(0,0);
@test prank(M, N, fastrank = fast) == 0

M = zeros(3,0); N = zeros(3,0);
@test prank(M, N, fastrank = fast) == 0


M = zeros(0,3); N = zeros(0,3);
@test prank(M, N, fastrank = fast) == 0

M = zeros(1,1); N = ones(1,1);
@test prank(M, N, fastrank = fast) == 1

@test prank(N, M, fastrank = fast) == 1

M = ones(1,1); N = nothing; 
@test prank(M, N, fastrank = fast) == 1


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
@test prank(M, N, fastrank = fast) == 4


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

@test prank(M, N, fastrank = fast) == 12


Ty = Float64
Ty = Complex{Float64}
#for Ty in (Float64, Float32, Complex{Float64},  Complex{Float32})
for Ty in (Float64, Complex{Float64})

abstol = sqrt(eps(one(real(Ty))))

# full row rank
r = 5; n = 20;
M = rand(Ty,r,n)
N = rand(Ty,r,n)

@time prnk = prank(M, N, fastrank = fast)
@test prnk == r


# full column rank
r = 5; m = 20;
M = rand(Ty,m,r)
N = rand(Ty,m,r)

@time prnk = prank(M, N, fastrank = fast)
@test prnk == r


# given rank
r = 5; m = 20; n = 30;
Q = rand(Ty,m,m);
Z = rand(Ty,n,n); 
M = Q*[ rand(Ty,r,r) zeros(Ty,r,n-r); zeros(Ty,m-r,n) ]*Z;
N = Q*[ rand(Ty,r,r) zeros(Ty,r,n-r); zeros(Ty,m-r,n) ]*Z;

@time prnk = prank(M, N, atol1 = 100*eps(opnorm(M,1)), atol2 = 100*eps(opnorm(N,1)), fastrank = fast)
@test prnk == r

end
end
end  # prank

end  # applications testset

end  # module