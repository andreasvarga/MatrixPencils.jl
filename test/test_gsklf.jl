module Test_gsklf

using LinearAlgebra
using MatrixPencils
using Test


@testset "gsklf" begin

fast = false; Ty = Float64; #Ty = Complex{Float64}  


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
n = 6; m = 2; p = 3;

M2 = [A B; C D]; N2 = [E zeros(n,m); zeros(p,n+m)];  
zerref, ziref, = pzeros(M2,N2)

M, N, Q, Z, dimsc, nmsz, niz = gsklf(A, E, B, C, D, jobopt = "none", fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
nr = dimsc[1]; nl = dimsc[2]; ml = dimsc[3]; nsinf = dimsc[4]; 
il = n-nsinf-nl+1:n-nsinf; jal = nr+1:nr+nl; jbl = nr+nl+1:nr+nl+ml; icl = n+1:n+p
Abl = M[il,jal]; Ebl = N[il,jal]; Bbl = M[il,jbl]; Cbl = M[icl,jal]; Dbl = M[icl,jbl];
zer, zi, = spzeros(Abl,Ebl,Bbl,Cbl,Dbl)
@test norm(Q'*[A B]*Z-M[1:n,:]) < sqrt(eps(1.)) &&
      norm([C D]*Z-M[n+1:n+p,:]) < sqrt(eps(1.)) &&
      norm(Q'*[E zeros(n,m)]*Z-N[1:n,:]) < sqrt(eps(1.)) &&
      dimsc == (4, 2, 2, 0) && ismissing(nmsz) && niz == 2 &&
      zer == Ty[] && zi == Int[]

M, N, Q, Z, dimsc, nmsz, niz = gsklf(A, E, B, C, D, jobopt = "unstable", fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
nr = dimsc[1]; nl = dimsc[2]; ml = dimsc[3]; nsinf = dimsc[4]; 
il = n-nsinf-nl+1:n-nsinf; jal = nr+1:nr+nl; jbl = nr+nl+1:nr+nl+ml; icl = n+1:n+p
Abl = M[il,jal]; Ebl = N[il,jal]; Bbl = M[il,jbl]; Cbl = M[icl,jal]; Dbl = M[icl,jbl];
zer, zi, = spzeros(Abl,Ebl,Bbl,Cbl,Dbl)
@test norm(Q'*[A B]*Z-M[1:n,:]) < sqrt(eps(1.)) &&
      norm([C D]*Z-M[n+1:n+p,:]) < sqrt(eps(1.)) &&
      norm(Q'*[E zeros(n,m)]*Z-N[1:n,:]) < sqrt(eps(1.)) &&
      dimsc == (3, 3, 2, 0) && nmsz == 0 && niz == 2 &&
      zer ≈ [2.0] && zi == Int[]

M, N, Q, Z, dimsc, nmsz, niz = gsklf(A, E, B, C, D, jobopt = "unstable", disc = true, fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
nr = dimsc[1]; nl = dimsc[2]; ml = dimsc[3]; nsinf = dimsc[4]; 
il = n-nsinf-nl+1:n-nsinf; jal = nr+1:nr+nl; jbl = nr+nl+1:nr+nl+ml; icl = n+1:n+p
Abl = M[il,jal]; Ebl = N[il,jal]; Bbl = M[il,jbl]; Cbl = M[icl,jal]; Dbl = M[icl,jbl];
zer, zi, = spzeros(Abl,Ebl,Bbl,Cbl,Dbl)
@test norm(Q'*[A B]*Z-M[1:n,:]) < sqrt(eps(1.)) &&
      norm([C D]*Z-M[n+1:n+p,:]) < sqrt(eps(1.)) &&
      norm(Q'*[E zeros(n,m)]*Z-N[1:n,:]) < sqrt(eps(1.)) &&
      dimsc == (1, 5, 2, 0) && nmsz == 1 && niz == 0 &&
      zer ≈ [2.0, Inf, Inf] && zi == [1, 1]

M, N, Q, Z, dimsc, nmsz, niz = gsklf(A, E, B, C, D, jobopt = "all", fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
nr = dimsc[1]; nl = dimsc[2]; ml = dimsc[3]; nsinf = dimsc[4]; 
il = n-nsinf-nl+1:n-nsinf; jal = nr+1:nr+nl; jbl = nr+nl+1:nr+nl+ml; icl = n+1:n+p
Abl = M[il,jal]; Ebl = N[il,jal]; Bbl = M[il,jbl]; Cbl = M[icl,jal]; Dbl = M[icl,jbl];
zer, zi, = spzeros(Abl,Ebl,Bbl,Cbl,Dbl)
@test norm(Q'*[A B]*Z-M[1:n,:]) < sqrt(eps(1.)) &&
      norm([C D]*Z-M[n+1:n+p,:]) < sqrt(eps(1.)) &&
      norm(Q'*[E zeros(n,m)]*Z-N[1:n,:]) < sqrt(eps(1.)) &&
      dimsc == (0, 6, 2, 0) && ismissing(nmsz) && ismissing(niz) &&
      sort(zer) ≈ sort(zerref) && zi == ziref


M, N, Q, Z, dimsc, nmsz, niz = gsklf(A, E, B, C, D, jobopt = "infinite", fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
nr = dimsc[1]; nl = dimsc[2]; ml = dimsc[3]; nsinf = dimsc[4]; 
il = n-nsinf-nl+1:n-nsinf; jal = nr+1:nr+nl; jbl = nr+nl+1:nr+nl+ml; icl = n+1:n+p
Abl = M[il,jal]; Ebl = N[il,jal]; Bbl = M[il,jbl]; Cbl = M[icl,jal]; Dbl = M[icl,jbl];
zer, zi, = spzeros(Abl,Ebl,Bbl,Cbl,Dbl)
@test norm(Q'*[A B]*Z-M[1:n,:]) < sqrt(eps(1.)) &&
      norm([C D]*Z-M[n+1:n+p,:]) < sqrt(eps(1.)) &&
      norm(Q'*[E zeros(n,m)]*Z-N[1:n,:]) < sqrt(eps(1.)) &&
      dimsc == (2, 4, 2, 0) && ismissing(nmsz) && niz == 2 &&
      zer == [Inf, Inf] && zi == [1,1]

M, N, Q, Z, dimsc, nmsz, niz = gsklf(A, E, B, C, D, jobopt = "stable", fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
nr = dimsc[1]; nl = dimsc[2]; ml = dimsc[3]; nsinf = dimsc[4]; 
il = n-nsinf-nl+1:n-nsinf; jal = nr+1:nr+nl; jbl = nr+nl+1:nr+nl+ml; icl = n+1:n+p
Abl = M[il,jal]; Ebl = N[il,jal]; Bbl = M[il,jbl]; Cbl = M[icl,jal]; Dbl = M[icl,jbl];
zer, zi, = spzeros(Abl,Ebl,Bbl,Cbl,Dbl)
@test norm(Q'*[A B]*Z-M[1:n,:]) < sqrt(eps(1.)) &&
      norm([C D]*Z-M[n+1:n+p,:]) < sqrt(eps(1.)) &&
      norm(Q'*[E zeros(n,m)]*Z-N[1:n,:]) < sqrt(eps(1.)) &&
      dimsc == (3, 3, 2, 0) && nmsz == 0 && niz == 2 &&
      zer ≈ [-1] && zi == Int[]

M, N, Q, Z, dimsc, nmsz, niz = gsklf(A, E, B, C, D, jobopt = "stable", disc = true, fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
nr = dimsc[1]; nl = dimsc[2]; ml = dimsc[3]; nsinf = dimsc[4]; 
il = n-nsinf-nl+1:n-nsinf; jal = nr+1:nr+nl; jbl = nr+nl+1:nr+nl+ml; icl = n+1:n+p
Abl = M[il,jal]; Ebl = N[il,jal]; Bbl = M[il,jbl]; Cbl = M[icl,jal]; Dbl = M[icl,jbl];
zer, zi, = spzeros(Abl,Ebl,Bbl,Cbl,Dbl)
@test norm(Q'*[A B]*Z-M[1:n,:]) < sqrt(eps(1.)) &&
      norm([C D]*Z-M[n+1:n+p,:]) < sqrt(eps(1.)) &&
      norm(Q'*[E zeros(n,m)]*Z-N[1:n,:]) < sqrt(eps(1.)) &&
      dimsc == (4, 2, 2, 0) && nmsz == 1 && niz == 0 &&
      zer == Ty[] && zi == Int[]

M, N, Q, Z, dimsc, nmsz, niz = gsklf(A, E, B, C, D, jobopt = "s-unstable", fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
nr = dimsc[1]; nl = dimsc[2]; ml = dimsc[3]; nsinf = dimsc[4]; 
il = n-nsinf-nl+1:n-nsinf; jal = nr+1:nr+nl; jbl = nr+nl+1:nr+nl+ml; icl = n+1:n+p
Abl = M[il,jal]; Ebl = N[il,jal]; Bbl = M[il,jbl]; Cbl = M[icl,jal]; Dbl = M[icl,jbl];
zer, zi, = spzeros(Abl,Ebl,Bbl,Cbl,Dbl)
@test norm(Q'*[A B]*Z-M[1:n,:]) < sqrt(eps(1.)) &&
      norm([C D]*Z-M[n+1:n+p,:]) < sqrt(eps(1.)) &&
      norm(Q'*[E zeros(n,m)]*Z-N[1:n,:]) < sqrt(eps(1.)) &&
      dimsc == (1, 5, 2, 0) && nmsz == 0 && niz == 2 &&
      zer ≈ [2, Inf, Inf] && zi == [1, 1]

M, N, Q, Z, dimsc, nmsz, niz = gsklf(A, E, B, C, D, jobopt = "s-unstable", disc = true, fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
nr = dimsc[1]; nl = dimsc[2]; ml = dimsc[3]; nsinf = dimsc[4]; 
il = n-nsinf-nl+1:n-nsinf; jal = nr+1:nr+nl; jbl = nr+nl+1:nr+nl+ml; icl = n+1:n+p
Abl = M[il,jal]; Ebl = N[il,jal]; Bbl = M[il,jbl]; Cbl = M[icl,jal]; Dbl = M[icl,jbl];
zer, zi, = spzeros(Abl,Ebl,Bbl,Cbl,Dbl)
@test norm(Q'*[A B]*Z-M[1:n,:]) < sqrt(eps(1.)) &&
      norm([C D]*Z-M[n+1:n+p,:]) < sqrt(eps(1.)) &&
      norm(Q'*[E zeros(n,m)]*Z-N[1:n,:]) < sqrt(eps(1.)) &&
      dimsc == (0, 6, 2, 0) && nmsz == 1 && niz == 0 &&
      sort(zer) ≈ sort(zerref) && zi == ziref


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
n = 9; m = 3; p = 3;
M2 = [A B; C D]; N2 = [E zeros(n,m); zeros(p,n+m)];
zerref, ziref, = pzeros(M2,N2)

M, N, Q, Z, dimsc, nmsz, niz = gsklf(A, E, B, C, D, jobopt = "none", fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
nr = dimsc[1]; nl = dimsc[2]; ml = dimsc[3]; nsinf = dimsc[4]; 
il = n-nsinf-nl+1:n-nsinf; jal = nr+1:nr+nl; jbl = nr+nl+1:nr+nl+ml; icl = n+1:n+p
Abl = M[il,jal]; Ebl = N[il,jal]; Bbl = M[il,jbl]; Cbl = M[icl,jal]; Dbl = M[icl,jbl];
zer, zi, = spzeros(Abl,Ebl,Bbl,Cbl,Dbl)
@test norm(Q'*[A B]*Z-M[1:n,:]) < sqrt(eps(1.)) &&
      norm([C D]*Z-M[n+1:n+p,:]) < sqrt(eps(1.)) &&
      norm(Q'*[E zeros(n,m)]*Z-N[1:n,:]) < sqrt(eps(1.)) &&
      dimsc == (6, 1, 2, 3) && ismissing(nmsz) && niz == 2 &&
      zer == Ty[] && zi == Int[]

M, N, Q, Z, dimsc, nmsz, niz = gsklf(A, E, B, C, D, jobopt = "unstable", fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
nr = dimsc[1]; nl = dimsc[2]; ml = dimsc[3]; nsinf = dimsc[4]; 
il = n-nsinf-nl+1:n-nsinf; jal = nr+1:nr+nl; jbl = nr+nl+1:nr+nl+ml; icl = n+1:n+p
Abl = M[il,jal]; Ebl = N[il,jal]; Bbl = M[il,jbl]; Cbl = M[icl,jal]; Dbl = M[icl,jbl];
zer, zi, = spzeros(Abl,Ebl,Bbl,Cbl,Dbl,atol1=1.e-7,atol2=1.e-7)
@test norm(Q'*[A B]*Z-M[1:n,:]) < sqrt(eps(1.)) &&
      norm([C D]*Z-M[n+1:n+p,:]) < sqrt(eps(1.)) &&
      norm(Q'*[E zeros(n,m)]*Z-N[1:n,:]) < sqrt(eps(1.)) &&
      dimsc == (5, 2, 2, 3) && nmsz == 0 && niz == 2 &&
      zer ≈ [1] && zi == Int[]

M, N, Q, Z, dimsc, nmsz, niz = gsklf(A, E, B, C, D, jobopt = "unstable", disc = true, fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
nr = dimsc[1]; nl = dimsc[2]; ml = dimsc[3]; nsinf = dimsc[4]; 
il = n-nsinf-nl+1:n-nsinf; jal = nr+1:nr+nl; jbl = nr+nl+1:nr+nl+ml; icl = n+1:n+p
Abl = M[il,jal]; Ebl = N[il,jal]; Bbl = M[il,jbl]; Cbl = M[icl,jal]; Dbl = M[icl,jbl];
zer, zi, = spzeros(Abl,Ebl,Bbl,Cbl,Dbl)
@test norm(Q'*[A B]*Z-M[1:n,:]) < sqrt(eps(1.)) &&
      norm([C D]*Z-M[n+1:n+p,:]) < sqrt(eps(1.)) &&
      norm(Q'*[E zeros(n,m)]*Z-N[1:n,:]) < sqrt(eps(1.)) &&
      dimsc == (4, 3, 2, 3) && nmsz == 1 && niz == 0 &&
      zer ≈ [Inf, Inf] && zi == [2]

M, N, Q, Z, dimsc, nmsz, niz = gsklf(A, E, B, C, D, jobopt = "all", fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
nr = dimsc[1]; nl = dimsc[2]; ml = dimsc[3]; nsinf = dimsc[4]; 
il = n-nsinf-nl+1:n-nsinf; jal = nr+1:nr+nl; jbl = nr+nl+1:nr+nl+ml; icl = n+1:n+p
Abl = M[il,jal]; Ebl = N[il,jal]; Bbl = M[il,jbl]; Cbl = M[icl,jal]; Dbl = M[icl,jbl];
zer, zi, = spzeros(Abl,Ebl,Bbl,Cbl,Dbl)
@test norm(Q'*[A B]*Z-M[1:n,:]) < sqrt(eps(1.)) &&
      norm([C D]*Z-M[n+1:n+p,:]) < sqrt(eps(1.)) &&
      norm(Q'*[E zeros(n,m)]*Z-N[1:n,:]) < sqrt(eps(1.)) &&
      dimsc == (3, 4, 2, 3) && ismissing(nmsz) && ismissing(niz) &&
      sort(zer) ≈ sort(zerref) && zi == ziref

M, N, Q, Z, dimsc, nmsz, niz = gsklf(A, E, B, C, D, jobopt = "infinite", fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
nr = dimsc[1]; nl = dimsc[2]; ml = dimsc[3]; nsinf = dimsc[4]; 
il = n-nsinf-nl+1:n-nsinf; jal = nr+1:nr+nl; jbl = nr+nl+1:nr+nl+ml; icl = n+1:n+p
Abl = M[il,jal]; Ebl = N[il,jal]; Bbl = M[il,jbl]; Cbl = M[icl,jal]; Dbl = M[icl,jbl];
zer, zi, = spzeros(Abl,Ebl,Bbl,Cbl,Dbl)
@test norm(Q'*[A B]*Z-M[1:n,:]) < sqrt(eps(1.)) &&
      norm([C D]*Z-M[n+1:n+p,:]) < sqrt(eps(1.)) &&
      norm(Q'*[E zeros(n,m)]*Z-N[1:n,:]) < sqrt(eps(1.)) &&
      dimsc == (4, 3, 2, 3) && ismissing(nmsz) && niz == 2 &&
      zer ≈ [Inf, Inf] && zi == [2]

M, N, Q, Z, dimsc, nmsz, niz = gsklf(A, E, B, C, D, jobopt = "finite", fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
nr = dimsc[1]; nl = dimsc[2]; ml = dimsc[3]; nsinf = dimsc[4]; 
il = n-nsinf-nl+1:n-nsinf; jal = nr+1:nr+nl; jbl = nr+nl+1:nr+nl+ml; icl = n+1:n+p
Abl = M[il,jal]; Ebl = N[il,jal]; Bbl = M[il,jbl]; Cbl = M[icl,jal]; Dbl = M[icl,jbl];
zer, zi, = spzeros(Abl,Ebl,Bbl,Cbl,Dbl)
@test norm(Q'*[A B]*Z-M[1:n,:]) < sqrt(eps(1.)) &&
      norm([C D]*Z-M[n+1:n+p,:]) < sqrt(eps(1.)) &&
      norm(Q'*[E zeros(n,m)]*Z-N[1:n,:]) < sqrt(eps(1.)) &&
      dimsc == (5, 2, 2, 3) && ismissing(nmsz) && ismissing(niz) &&
      zer ≈ [1] && zi == []

M, N, Q, Z, dimsc, nmsz, niz = gsklf(A, E, B, C, D, jobopt = "stable", fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
nr = dimsc[1]; nl = dimsc[2]; ml = dimsc[3]; nsinf = dimsc[4]; 
il = n-nsinf-nl+1:n-nsinf; jal = nr+1:nr+nl; jbl = nr+nl+1:nr+nl+ml; icl = n+1:n+p
Abl = M[il,jal]; Ebl = N[il,jal]; Bbl = M[il,jbl]; Cbl = M[icl,jal]; Dbl = M[icl,jbl];
zer, zi, = spzeros(Abl,Ebl,Bbl,Cbl,Dbl)
@test norm(Q'*[A B]*Z-M[1:n,:]) < sqrt(eps(1.)) &&
      norm([C D]*Z-M[n+1:n+p,:]) < sqrt(eps(1.)) &&
      norm(Q'*[E zeros(n,m)]*Z-N[1:n,:]) < sqrt(eps(1.)) &&
      dimsc == (6, 1, 2, 3) && nmsz == 0 && niz == 2 &&
      zer ≈ Ty[] && zi == Int[]

M, N, Q, Z, dimsc, nmsz, niz = gsklf(A, E, B, C, D, jobopt = "stable", disc = true, fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
nr = dimsc[1]; nl = dimsc[2]; ml = dimsc[3]; nsinf = dimsc[4]; 
il = n-nsinf-nl+1:n-nsinf; jal = nr+1:nr+nl; jbl = nr+nl+1:nr+nl+ml; icl = n+1:n+p
Abl = M[il,jal]; Ebl = N[il,jal]; Bbl = M[il,jbl]; Cbl = M[icl,jal]; Dbl = M[icl,jbl];
zer, zi, = spzeros(Abl,Ebl,Bbl,Cbl,Dbl)
@test norm(Q'*[A B]*Z-M[1:n,:]) < sqrt(eps(1.)) &&
      norm([C D]*Z-M[n+1:n+p,:]) < sqrt(eps(1.)) &&
      norm(Q'*[E zeros(n,m)]*Z-N[1:n,:]) < sqrt(eps(1.)) &&
      dimsc == (6, 1, 2, 3) && nmsz == 1 && niz == 0 &&
      zer == Ty[] && zi == Int[]

M, N, Q, Z, dimsc, nmsz, niz = gsklf(A, E, B, C, D, jobopt = "s-unstable", fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
nr = dimsc[1]; nl = dimsc[2]; ml = dimsc[3]; nsinf = dimsc[4]; 
il = n-nsinf-nl+1:n-nsinf; jal = nr+1:nr+nl; jbl = nr+nl+1:nr+nl+ml; icl = n+1:n+p
Abl = M[il,jal]; Ebl = N[il,jal]; Bbl = M[il,jbl]; Cbl = M[icl,jal]; Dbl = M[icl,jbl];
zer, zi, = spzeros(Abl,Ebl,Bbl,Cbl,Dbl)
@test norm(Q'*[A B]*Z-M[1:n,:]) < sqrt(eps(1.)) &&
      norm([C D]*Z-M[n+1:n+p,:]) < sqrt(eps(1.)) &&
      norm(Q'*[E zeros(n,m)]*Z-N[1:n,:]) < sqrt(eps(1.)) &&
      dimsc == (3, 4, 2, 3) && nmsz == 0 && niz == 2 &&
      zer ≈ [1, Inf, Inf] && zi == [2]

M, N, Q, Z, dimsc, nmsz, niz = gsklf(A, E, B, C, D, jobopt = "s-unstable", disc = true, fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
nr = dimsc[1]; nl = dimsc[2]; ml = dimsc[3]; nsinf = dimsc[4]; 
il = n-nsinf-nl+1:n-nsinf; jal = nr+1:nr+nl; jbl = nr+nl+1:nr+nl+ml; icl = n+1:n+p
Abl = M[il,jal]; Ebl = N[il,jal]; Bbl = M[il,jbl]; Cbl = M[icl,jal]; Dbl = M[icl,jbl];
zer, zi, = spzeros(Abl,Ebl,Bbl,Cbl,Dbl)
@test norm(Q'*[A B]*Z-M[1:n,:]) < sqrt(eps(1.)) &&
      norm([C D]*Z-M[n+1:n+p,:]) < sqrt(eps(1.)) &&
      norm(Q'*[E zeros(n,m)]*Z-N[1:n,:]) < sqrt(eps(1.)) &&
      dimsc == (3, 4, 2, 3) && nmsz == 1 && niz == 0 &&
      sort(zer) ≈ sort(zerref) && zi == ziref


Ty = Complex{Float64};  fast = false; 
for Ty in (Float64, Complex{Float64})
n = 5; m = 4; p = 3;  
A = rand(Ty,n,n); B = rand(Ty,n,m); C = rand(Ty,p,n); D = rand(Ty,p,m); 
E = I;
M2 = [A B; C D]; N2 = [E zeros(n,m); zeros(p,n+m)];
zerref, ziref, = pzeros(M2,N2)

M, N, Q, Z, dimsc, nmsz, niz = gsklf(A, E, B, C, D, jobopt = "none", fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
nr = dimsc[1]; nl = dimsc[2]; ml = dimsc[3]; nsinf = dimsc[4]; 
il = n-nsinf-nl+1:n-nsinf; jal = nr+1:nr+nl; jbl = nr+nl+1:nr+nl+ml; icl = n+1:n+p
Abl = M[il,jal]; Ebl = N[il,jal]; Bbl = M[il,jbl]; Cbl = M[icl,jal]; Dbl = M[icl,jbl];
zer, zi, = spzeros(Abl,Ebl,Bbl,Cbl,Dbl)

@test norm(Q'*[A B]*Z-M[1:n,:]) < sqrt(eps(1.)) &&
      norm([C D]*Z-M[n+1:n+p,:]) < sqrt(eps(1.)) &&
      norm(Q'*[E zeros(n,m)]*Z-N[1:n,:]) < sqrt(eps(1.)) &&
      dimsc == (6, 0, 3, 0) && ismissing(nmsz) && niz == 0 &&
      zer ≈ Ty[] && zi == []

n = 9; m = 3; p = 3; rE = 6; 
A = rand(Ty,n,n); B = rand(Ty,n,m); C = rand(Ty,p,n); D = rand(Ty,p,m); 
E = [rand(Ty,n,rE)  zeros(Ty,n,n-rE)]*qr(rand(Ty,n,n)).Q
M2 = [A B; C D]; N2 = [E zeros(n,m); zeros(p,n+m)];
zerref, ziref, = pzeros(M2,N2)

M, N, Q, Z, dimsc, nmsz, niz = gsklf(A, E, B, C, D, jobopt = "none", fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
nr = dimsc[1]; nl = dimsc[2]; ml = dimsc[3]; nsinf = dimsc[4]; 
il = n-nsinf-nl+1:n-nsinf; jal = nr+1:nr+nl; jbl = nr+nl+1:nr+nl+ml; icl = n+1:n+p
Abl = M[il,jal]; Ebl = N[il,jal]; Bbl = M[il,jbl]; Cbl = M[icl,jal]; Dbl = M[icl,jbl];
zer, zi, = spzeros(Abl,Ebl,Bbl,Cbl,Dbl)

@test norm(Q'*[A B]*Z-M[1:n,:]) < sqrt(eps(1.)) &&
      norm([C D]*Z-M[n+1:n+p,:]) < sqrt(eps(1.)) &&
      norm(Q'*[E zeros(n,m)]*Z-N[1:n,:]) < sqrt(eps(1.)) &&
      dimsc == (6, 0, 3, 3) && ismissing(nmsz) && niz == 0 &&
      zer ≈ Ty[] && zi == []

M, N, Q, Z, dimsc, nmsz, niz = gsklf(A, E, B, C, D, jobopt = "unstable", fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
nr = dimsc[1]; nl = dimsc[2]; ml = dimsc[3]; nsinf = dimsc[4]; 
il = n-nsinf-nl+1:n-nsinf; jal = nr+1:nr+nl; jbl = nr+nl+1:nr+nl+ml; icl = n+1:n+p
Abl = M[il,jal]; Ebl = N[il,jal]; Bbl = M[il,jbl]; Cbl = M[icl,jal]; Dbl = M[icl,jbl];
zer, zi, = spzeros(Abl,Ebl,Bbl,Cbl,Dbl,atol1=1.e-7,atol2=1.e-7)
@test norm(Q'*[A B]*Z-M[1:n,:]) < sqrt(eps(1.)) &&
      norm([C D]*Z-M[n+1:n+p,:]) < sqrt(eps(1.)) &&
      norm(Q'*[E zeros(n,m)]*Z-N[1:n,:]) < sqrt(eps(1.)) 

M, N, Q, Z, dimsc, nmsz, niz = gsklf(A, E, B, C, D, jobopt = "unstable", disc = true, fast = fast, atol1 = 1.e-7, atol2 = 1.e-7)
nr = dimsc[1]; nl = dimsc[2]; ml = dimsc[3]; nsinf = dimsc[4]; 
il = n-nsinf-nl+1:n-nsinf; jal = nr+1:nr+nl; jbl = nr+nl+1:nr+nl+ml; icl = n+1:n+p
Abl = M[il,jal]; Ebl = N[il,jal]; Bbl = M[il,jbl]; Cbl = M[icl,jal]; Dbl = M[icl,jbl];
zer, zi, = spzeros(Abl,Ebl,Bbl,Cbl,Dbl)
@test norm(Q'*[A B]*Z-M[1:n,:]) < sqrt(eps(1.)) &&
      norm([C D]*Z-M[n+1:n+p,:]) < sqrt(eps(1.)) &&
      norm(Q'*[E zeros(n,m)]*Z-N[1:n,:]) < sqrt(eps(1.)) 
end      

end
end #fast

end #module