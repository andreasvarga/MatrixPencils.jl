module Test_rmtools

using LinearAlgebra
using MatrixPencils
using Polynomials 
using Test

println("Test_rmtools")

@testset "Rational Matrix Tools" begin


@testset "Tests polgcdvw, pollcm, conv, poldivrem, poldiv" begin


# test example for Polynomials functions
num = Polynomial([0.8581454436924945, 0.249671302254737, 0.8048498901050951, 0.1922713965697087])
den = Polynomial([0.9261520696359462, 0.07141031902098072, 0.378071465860349])

@test poldeg(coeffs(num)) == 3 && poldeg1(coeffs(num)) == 4 

q1, r1 = poldivrem(coeffs(num),coeffs(den))
@test coeffs(num) ≈ conv(coeffs(den),q1) + [r1;zeros(2) ]

d, = polgcdvw(coeffs(num),coeffs(den))
@test d == [1]

m = pollcm(coeffs(num),coeffs(den))
p = conv(coeffs(num),coeffs(den))
@test m*p[end] ≈ m[end]*p


# test example which fails for polgcd but not for polgcdvw
num = [ 0.34137237259273556
0.1815967953870914
0.005829782404846551
0.7934778818753858
0.5308749468599383
0.14488329929437715
0.488795838965558
0.806095325402598
0.29135959590683647
0.22853582789218474];
den = [ 0.7371989151075657
0.3525356298341029
0.7811847379591219 
0.6173718776940151
0.32725779892403306
];
d = [ 0.015852680715187084
0.3602221067929683
0.7487570819767699
0.5882036947609526
0.011233093587261544
];

d2, = polgcdvw(conv(num,d),conv(den,d),atol=1.e-10);  # OK
@test norm(d*d2[end] - d2*d[end]) / norm(d) < 1.e-7


# Example Table 7: Zeng Math. Comp. 2005

num = reverse([1.00000000 
23.35360257 
29.89831582 
10.75803809 
15.57240922 
18.76038493 
13.73079603 
30.45600101 
46.21275197 
44.89871211 
30.17981700
8.33455813]);
den = reverse([1.00000000 
23.01829201
22.05776405]);
sol=reverse([1.00000000 
0.33531056 
0.12227539 
0.54726624 
0.27815340 
0.28629915 
1.00523653 
1.00205392 
0.97391204 
0.37785145 ]);

# this test fails
q, r = poldivrem(num,den)
@test !(q ≈ sol)

# long division has small residual 
@test conv(q,den) ≈ num - [r;zeros(length(num)-length(r))] # OK

@test conv(sol,den)≈ num # OK

q = poldiv(num,den)  # accurate deconvolution for approximate quotient
@test q ≈ sol

# long division has small residual
q, r = poldivrem(conv(sol,den),den)
@test conv(q,den) ≈ conv(sol,den) - [r;zeros(length(num)-length(r))] # OK


q = poldiv(conv(sol,den),den)  # accurate deconvolution for exact quotient
@test q ≈ sol

# testing conv and poldiv for random data
num = rand(100);
d = rand(25); #d = Polynomial([100])
q = poldiv(conv(num,d),d);
@test q ≈ num

@test conv(num,d) ≈ convmtx(d,length(num))*num 
@test conv(d,num) ≈ convmtx(d,length(num))*num 

@test conv(d,num) ≈ convmtx(num,length(d))*d 
@test conv(num,d) ≈ convmtx(num,length(d))*d 

Ty = Complex{Float64}
num = rand(Ty,100);
d = rand(Ty,25); #d = Polynomial([100])
q = poldiv(conv(num,d),d)
@test q ≈ num

@test conv(num,d) ≈ convmtx(d,length(num))*num 
@test conv(d,num) ≈ convmtx(d,length(num))*num 

@test conv(d,num) ≈ convmtx(num,length(d))*d 
@test conv(num,d) ≈ convmtx(num,length(d))*d 


Ty = Complex{Float64}
num = rand(Ty,100);
d = rand(25); #d = Polynomial([100])
q = poldiv(conv(num,d),d)
@test q ≈ num

@test conv(num,d) ≈ convmtx(d,length(num))*num 
@test conv(d,num) ≈ convmtx(d,length(num))*num 

@test conv(d,num) ≈ convmtx(num,length(d))*d 
@test conv(num,d) ≈ convmtx(num,length(d))*d 

# tests polgcdvw, pollcm, conv, poldiv
p1 = [-6, 11, -6, 1]
p2 = [4, -5, 1]
d1, v1, w1, δ1 = polgcdvw(p1,p2)
@test poldiv(p1,d1) ≈ v1*norm(p1) && poldiv(p2,d1) ≈ w1*norm(p2)
m = pollcm(p1,p2)
mc = poldiv(conv(p1,p2),d1)
@test  mc*m[end]≈ m*mc[end]  


p1 = [-6, 11, -6, 1]*rand(Complex{Float64})
p2 = [4, -5, 1]*rand(Complex{Float64})
d1, v1, w1, δ1 = polgcdvw(p1,p2)
@test poldiv(p1,d1) ≈ v1*norm(p1) && poldiv(p2,d1) ≈ w1*norm(p2)
m = pollcm(p1,p2)
mc = poldiv(conv(p1,p2),d1)
@test  mc*m[end]≈ m*mc[end]  

# Example 2.1 Zeng 2011
p1 = [10, 1, 0, 0,  0, 0, 0, 0, 10/3, 31/3, 1];
p2 = [-60/7, -6/7, 0, 0,  0, 0, 0, 0, 10/7, 71/7, 1];
d = [10, 1];

d2, v2, w2 = polgcdvw(p1,p2,atol=1.e-5);
@test norm(d*d2[end] - d2*d[end]) / norm(d) < 1.e-7
@test p1/norm(p1) ≈ conv(v2,d2) && p2/norm(p2) ≈ conv(w2,d2)

m = pollcm(p1,p2);
mc = poldiv(conv(p1,p2),d);
@test  mc*m[end]≈ m*mc[end]  


# Example 2.2 Zeng 2011 "exact"
p2 = [10, 1, 0, 0,  0, 0, 0, 0, 10/3, 31/3, 1]
p1 = [10, 1]
d = [10, 1]

d2, v2, w2 = polgcdvw(p1,p2,atol=1.e-5);
@test norm(d*d2[end] - d2*d[end]) / norm(d) < 1.e-7
@test p1/norm(p1) ≈ conv(v2,d2) && p2/norm(p2) ≈ conv(w2,d2)

m = pollcm(p1,p2);
mc = poldiv(conv(p1,p2),d);
@test  mc*m[end]≈ m*mc[end]  


# Example 2.2 Zeng 2011 "approximate"
p1 = [10, 1, 0, 0,  0, 0, 0, 0,  3.333333333,  10.33333333, 1]
p2 = [10, 1]
d = [10, 1]

d2, v2, w2 = polgcdvw(p1,p2,atol=1.e-5);
@test norm(d*d2[end] - d2*d[end]) / norm(d) < 1.e-7
@test p1/norm(p1) ≈ conv(v2,d2) && p2/norm(p2) ≈ conv(w2,d2)

m = pollcm(p1,p2);
mc = poldiv(conv(p1,p2),d);
@test  mc*m[end]≈ m*mc[end]  


# q, r = poldivrem(p1,p3) # inexact remainder of order 3!!

# Example of Zeng 2009, with corrected p1
p1 = [-.999999999999, 1, -.999999999999, 1, -.999999999999, 1, -.999999999999, 1];
p2 = [2,-2, 2, -3, 1, -1, 1];
d = [1, -1, 1, -1];   # approximate gcd

d2, v2, w2 = polgcdvw(p1,p2,atol=1.e-9);
@test norm(d*d2[end] - d2*d[end]) / norm(d) < 1.e-7
@test poldiv(p1,d2) ≈ v2*norm(p1) && poldiv(p2,d2) ≈ w2*norm(p2)

m = pollcm(p1,p2);
mc = poldiv(conv(p1,p2),d);
@test  mc*m[end]≈ m*mc[end]  


# Example of Zeng 2009, with corrected p1
p1 = [-1, 1, -1, 1, -1, 1, -1, 1]
p2 = [2,-2, 2, -3, 1, -1, 1]
d = [1, -1, 1, -1];   # exact gcd

d2, v2, w2 = polgcdvw(p1,p2,atol=1.e-9);
@test norm(d*d2[end] - d2*d[end]) / norm(d) < 1.e-7
@test poldiv(p1,d2) ≈ v2*norm(p1) && poldiv(p2,d2) ≈ w2*norm(p2)

m = pollcm(p1,p2);
mc = poldiv(conv(p1,p2),d);
@test  mc*m[end]≈ m*mc[end]  


# Example in Test 3 of Zeng 2011
v = [1,1,1,1];
w = [1,-1,1,-1,1];
d = rand(20);
p1 = conv(v,d);
p2 = conv(w,d);

d2, v2, w2 = polgcdvw(p1,p2,atol=1.e-9);
@test norm(d*d2[end] - d2*d[end]) / norm(d) < 1.e-7
@test poldiv(p1,d2) ≈ v2*norm(p1) && poldiv(p2,d2) ≈ w2*norm(p2)

m = pollcm(p1,p2);
mc = poldiv(conv(p1,p2),d);
@test  mc*m[end]≈ m*mc[end]  

# testing polgcdvw for random data and fixed d 
v = rand(5); w = rand(13); d = ones(20); 
#d = Polynomial([100])
p1 = conv(v,d); p2 = conv(w,d);

d2, v2, w2 = polgcdvw(p1,p2,atol=1.e-9);
@test norm(d*d2[end] - d2*d[end]) / norm(d) < 1.e-7
@test poldiv(p1,d2) ≈ v2*norm(p1) && poldiv(p2,d2) ≈ w2*norm(p2)

m = pollcm(p1,p2);
mc = poldiv(conv(p1,p2),d);
@test  mc*m[end]≈ m*mc[end]  


# testing polgcdvw for random data
v = rand(5); w = rand(13); d = rand(20); 
#d = Polynomial([100])
p1 = conv(v,d); p2 = conv(w,d);

d2, v2, w2 = polgcdvw(p1,p2,atol=1.e-9);
@test norm(d*d2[end] - d2*d[end]) / norm(d) < 1.e-7
q1 = poldiv(p1,d2); q2 = poldiv(p2,d2);
@test q1*v2[end] ≈ v2*q1[end] && q2*w2[end] ≈ w2*q2[end]

#
p=.01
d3, v3, w3, δ3 = gcdvwupd(p1,p2,d2+p*rand(length(d2)),v2+p*rand(length(v2)),w2+p*rand(length(w2)),maxnit=100);
@test norm(d*d3[end] - d3*d[end]) / norm(d) < 1.e-7
q1 = poldiv(p1,d3); q2 = poldiv(p2,d3);
@test q1*v3[end] ≈ v3*q1[end] && q2*w3[end] ≈ w3*q2[end]

m = pollcm(p1,p2);
mc = poldiv(conv(p1,p2),d);
@test  mc*m[end]≈ m*mc[end]  


# testing polgcdvw for complex random data
Ty = Complex{Float64}
v = rand(Ty,5); w = rand(Ty,13); d = rand(Ty,20); 
#d = Polynomial([100])
p1 = conv(v,d); p2 = conv(w,d);

d2, v2, w2, δ2 = polgcdvw(p1,p2,atol=1.e-9);
@test norm(d*d2[end] - d2*d[end]) / norm(d) < 1.e-7
q1 = poldiv(p1,d2); q2 = poldiv(p2,d2);
@test q1*v2[end] ≈ v2*q1[end] && q2*w2[end] ≈ w2*q2[end]


d2, v2, w2, δ2 = polgcdvw(p1,p2,atol=1.e-9,maxnit=10);
@test norm(d*d2[end] - d2*d[end]) / norm(d) < 1.e-7
q1 = poldiv(p1,d2); q2 = poldiv(p2,d2);
@test q1*v2[end] ≈ v2*q1[end] && q2*w2[end] ≈ w2*q2[end]


p = 0.01
d3, v3, w3, δ3 = gcdvwupd(p1,p2,d2+p*rand(Ty,length(d2)),v2+p*rand(Ty,length(v2)),w2+p*rand(Ty,length(w2)),maxnit=10);
@test norm(d*d3[end] - d3*d[end]) / norm(d) < 1.e-7
q1 = poldiv(p1,d3); q2 = poldiv(p2,d3);
@test q1*v3[end] ≈ v3*q1[end] && q2*w3[end] ≈ w3*q2[end]

m = pollcm(p1,p2);
mc = poldiv(conv(p1,p2),d);
@test  mc*m[end]≈ m*mc[end]  



# some particular cases

num = Polynomial([2, 1]);
d, v, w, δ = polgcdvw(coeffs(num),[0])
@test d == [1.] && v == [2.0, 1.0] && w == [0.] && δ == 0.

d, v, w, δ = polgcdvw([0],coeffs(num))
@test d == [1.] && w == [2.0, 1.0] && v == [0.] && δ == 0.

d, v, w, δ = polgcdvw(coeffs(num),[5])
@test d == [1.] && v*norm(num) ≈ [2.0, 1.0] && w*5 == [5.] && δ == 0.

d, v, w, δ = polgcdvw([5],coeffs(num))
@test d == [1.] && w*norm(num) ≈ [2.0, 1.0] && v*5 == [5.] && δ == 0.


end 





@testset "Rational Matrix Tools" begin

# Example Dopico et al. 2020
e1 = Polynomial(rand(6))
e2 = Polynomial(rand(2))
P = [e1 0; 0 e2]
vals, info = pmeigvals(P,atol=1.e-7)
@test info.id  == [4] # infinite eigenvalue of multiplicity 4 

zer, iz, info = pmzeros(P,atol=1.e-7)
@test iz == []

sys = rm2ls(P,atol = 1.e-7,minimal=true)
zer, iz, info = spzeros(sys[1:5]...,atol1=1.e-7,atol2 = 1.e-7)
@test iz == []


c1 = [0.06155268911372547, 0.006545378521362721, 0.40445990039119284, 0.8829892254580274, 0.7573496766341161, 0.4094804382958148]; 
c2 = [0.4059208742974014, 0.48094667571705574]; 
e1 = Polynomial(c1)
e2 = Polynomial(c2)

NUM = [e1 0; 1 e2]
DEN = [1 1; Polynomial([0, 1]) 1]

A, E, B, C, D, bl = rm2ls(NUM,DEN,atol = 1.e-7,minimal=true)

z1, iz1, info1 = spzeros(A,E,B,C,D)
@test iz1 == [] && any(abs.(z1) .< 1.e-10)

p1, ip1, info = pzeros(A,E)
@test ip1 == [1, 5]  && any(abs.(p1) .< 1.e-10)


# Example Vlad Ionescu
s = Polynomial([0, 1],:s);
NUM = [s^2+3*s+3 1; -1 2*s^2+7*s+4];
DEN = [(s+1)^2 s+2; (s+1)^3 (s+1)*(s+2)]

sys = rm2ls(NUM,DEN,atol = 1.e-7, minimal=true) 
val = float(pi)
@test pmeval(NUM,val) ./ pmeval(DEN,val) ≈ lseval(sys[1:5]...,val)

num1, den1 = ls2rm(sys[1:5]...,atol1 = 1.e-7)
@test all(NUM .* pm2poly(den1,:s) .≈ DEN .* pm2poly(num1,:s))

num1, den1 = ls2rm(sys[1],sys[3:5]...,atol1 = 1.e-7)
@test all(NUM .* pm2poly(den1,:s) .≈ DEN .* pm2poly(num1,:s))


psys = rm2lps(NUM,DEN,atol = 1.e-3, minimal=true) 
val = float(pi)
@test pmeval(NUM,val) ./ pmeval(DEN,val) ≈ lpseval(psys[1:8]...,val)

λ = Polynomial([0,1],:λ)
syst=rm2lps(λ)
info = pkstruct(syst[7],syst[8])
@test (info.rki, info.lki,info.id, info.nf, info.nrank) == ([], [], [], 1, 1)


s = Polynomial([0, 1],:s);
NUM1 = [(s+1)*(s+2) s+1; -1 s];
DEN1 = [(s+1)^3 (s+1)*(s+2); (s+1)^3 (s+1)*(s+2)]
sys1 = rm2ls(NUM1,DEN1,atol = 1.e-5, minimal=true); 
val = float(pi)
@test pmeval(NUM1,val) ./ pmeval(DEN1,val) ≈ lseval(sys1[1:5]...,val)

@test lsequal(sys[1:5]...,sys1[1:4]...,sys[5])

s = Polynomial([0, 1],:s);
N = [(s+1)*(s+2) s+1; -1 s];
D = [(s+1)^3 0; 0 (s+1)*(s+2)];

sys2 = rpmfd2ls(D,N,atol=1.e-7,minimal = true);
@test lsequal(sys[1:5]...,sys2[1:4]...,sys[5],atol1=1.e-5)
val = float(pi)
@test pmeval(N,val) / pmeval(D,val) ≈ lseval(sys2...,val)

psys2 = rpmfd2lps(D,N,atol=1.e-7,minimal = true);
@test lpsequal(psys[1:8]...,psys2[1:6]...,psys2[7]+psys[7],psys[8],atol1=1.e-5)
val = float(pi)
@test pmeval(N,val) / pmeval(D,val) ≈ lpseval(psys2...,val)

s = Polynomial([0, 1],:s);
N = [(s+1)*(s+2) s+1; -1 s];
D = [(s+1)^3 0; 0 (s+1)*(s+2)];

sys2 = lpmfd2ls(poly2pm(D),poly2pm(N),atol=1.e-7,minimal = true);
val = float(pi)
@test  pmeval(D,val) \ pmeval(N,val)  ≈ lseval(sys2...,val)

sys2 = lpmfd2ls(D,N,atol=1.e-7,minimal = true);
val = float(pi)
@test  pmeval(D,val) \ pmeval(N,val)  ≈ lseval(sys2...,val)


psys2 = lpmfd2lps(poly2pm(D),poly2pm(N),atol=1.e-7,minimal = true);
val = float(pi)
@test  pmeval(D,val) \ pmeval(N,val)  ≈ lpseval(psys2...,val)

psys2 = lpmfd2lps(D,N,atol=1.e-7,minimal = true);
val = float(pi)
@test  pmeval(D,val) \ pmeval(N,val)  ≈ lpseval(psys2...,val)


s = Polynomial([0, 1],:s);
D = [(s+1)^3 0; 0 (s+1)*(s+2)];
sys = pminv2ls(D,atol=1.e-7,minimal = true);
val = float(pi)
@test  pmeval(D,val) * lseval(sys...,val)  ≈ I


NUM, DEN = ls2rm(sys...)
@test  rmeval(NUM,DEN,val) * pmeval(D,val)  ≈ I

sys1 = pminv2lps(poly2pm(D),atol=1.e-7,minimal = true)
val = float(pi)
@test  pmeval(D,val) * lpseval(sys1...,val)  ≈ I

sys1 = pminv2lps(D,atol=1.e-7,minimal = true)
val = float(pi)
@test  pmeval(D,val) * lpseval(sys1...,val)  ≈ I

NUM1, DEN1 = lps2rm(sys1...)
@test  rmeval(NUM1,DEN1,val) * pmeval(D,val)  ≈ I


s = Polynomial([0, 1],:x)
U = [-s  0   s^2 + 2*s + 1
s^2 - s + 1  0 -s^3 - s^2 - 1
0 -1 s];
V = [3*s + 1 3*s^3 + s^2 - 3*s - 4  6
-s -s^3 + s + 1  -2
0  0  1];
@test ispmunimodular(U)
@test ispmunimodular(V)

sys1 = pminv2ls(U,atol=1.e-7,minimal = true)
val = float(pi)
@test  pmeval(U,val) * lseval(sys1...,val)  ≈ I

NUM1, DEN1 = ls2rm(sys1...)
@test  rmeval(NUM1,DEN1,val) * pmeval(U,val)  ≈ I

@test ispmunimodular(NUM1,atol=1.e-7) && DEN1[:,:,1] ≈ ones(3,3)

sys2 = pminv2ls(poly2pm(V),atol=1.e-7,minimal = true)
val = float(pi)
@test  pmeval(V,val) * lseval(sys2...,val)  ≈ I

NUM2, DEN2 = ls2rm(sys2...)
@test  rmeval(NUM2,DEN2,val) * pmeval(V,val)  ≈ I

@test ispmunimodular(NUM2,atol=1.e-7) && DEN2[:,:,1] ≈ ones(3,3)


sys1 = pminv2lps(poly2pm(U),atol=1.e-7,minimal = true)
val = float(pi)
@test  pmeval(U,val) * lpseval(sys1...,val)  ≈ I

NUM1, DEN1 = lps2rm(sys1...,atol1=1.e-7,atol2=1.e-7)
@test  rmeval(NUM1,DEN1,val) * pmeval(U,val)  ≈ I

@test ispmunimodular(NUM1,atol=1.e-7) && DEN1[:,:,1] ≈ ones(3,3)

sys2 = pminv2lps(poly2pm(V),atol=1.e-7,minimal = true)
val = float(pi)
@test  pmeval(V,val) * lpseval(sys2...,val)  ≈ I

NUM2, DEN2 = lps2rm(sys2...,atol1=1.e-7,atol2=1.e-7)
@test  rmeval(NUM2,DEN2,val) * pmeval(V,val)  ≈ I

@test ispmunimodular(NUM2,atol=1.e-7) && DEN2[:,:,1] ≈ ones(3,3) 

end



@testset "Rational matrix realizations" begin
s = Polynomial([0, 1],:s)
num = Polynomial([4:-1:1...],:s)
den = Polynomial([7:-1:4...,1],:s)
A, B, C, D, blkdims = rm2lspm(num,den)
@test rmeval(num,den,1) ≈ lseval(A,I,B,C,D,1)

A, B, C, D, blkdims = rm2lspm(num,den,contr=true)
@test rmeval(num,den,1) ≈ lseval(A,I,B,C,D,1)



num1, den1 = ls2rm(A,Matrix{Int}(I,4,4),B,C,D)
@test num*pm2poly(den1,:s)[1,1] ≈ pm2poly(num1,:s)[1,1]*den

A1, B1, C1, D1 = rm2lspm(den,num)
@test rmeval(den,num,1) ≈ lseval(A1,I,B1,C1,zeros(1,1),1) + pmeval(D1,1)

num1, den1 = ls2rm(A1,Matrix{Int}(I,3,3),B1,C1,zeros(1,1))
@test den*pm2poly(den1,:s)[1,1] ≈ (pm2poly(num1,:s)[1,1]+pm2poly(D1,:s)[1,1]*pm2poly(den1,:s)[1,1])*num


A, E, B, C, D = rm2ls(num,den,minimal = true)
@test rmeval(num,den,1) ≈ lseval(A,E,B,C,D,1)

num1, den1 = ls2rm(A, E, B, C, D)
@test num*pm2poly(den1,:s)[1,1] ≈ pm2poly(num1,:s)[1,1]*den

A, E, B, C, D = rm2ls(den,num,minimal = true)
@test rmeval(den,num,1) ≈ lseval(A,E,B,C,D,1)

den1, num1 = ls2rm(A, E, B, C, D)
@test num*pm2poly(den1,:s)[1,1] ≈ pm2poly(num1,:s)[1,1]*den


A, E, B, C, D = rm2ls(num;minimal = true)
@test rmeval(num,1) ≈ lseval(A,E,B,C,D,1)

num1, den1 = ls2rm(A, E, B, C, D)
@test num*pm2poly(den1,:s)[1,1] ≈ pm2poly(num1,:s)[1,1]


num = rand(2,3,4);
den = rand(2,3,2) .+ 0.5;
# num = rand(1,1,4)
# den = rand(1,1,2)
A, E, B, C, D = rm2ls(num,den,minimal = true,atol = 1.e-7);
@test rmeval(num,den,1) ≈ lseval(A,E,B,C,D,1)

num1, den1 = ls2rm(A, E, B, C, D, atol1 = 1.e-7, atol2 = 1.e-7);
@test rmeval(num1,den1,1) ≈ lseval(A,E,B,C,D,1)

@test all(pm2poly(num) .* pm2poly(den1) .≈ pm2poly(num1) .* pm2poly(den))

num = rand(2,3,2)
den = rand(2,3,4)
# num = rand(1,1,4)
# den = rand(1,1,2)
A, E, B, C, D = rm2ls(num,den,minimal = true,atol = 1.e-7);
@test rmeval(num,den,1) ≈ lseval(A,E,B,C,D,1)

num1, den1 = ls2rm(A, E, B, C, D, atol1 = 1.e-7, atol2 = 1.e-7);
@test rmeval(num1,den1,1) ≈ lseval(A,E,B,C,D,1)

@test all(pm2poly(num) .* pm2poly(den1) .≈ pm2poly(num1) .* pm2poly(den))

# Example Varga, Sima 1997
s = Polynomial([0, 1],:s)
num = [s 2; 1 s]
den = [s+1 (s+1)*(s+3); s+4 (s+2)*(s+4)]

A, B, C, D, blkdims = rm2lspm(num,den,contr=true)
@test rmeval(num,den,1) ≈ lseval(A,I,B,C,D,1)

A, B, C, D, blkdims = rm2lspm(num,den,obs=true)
@test rmeval(num,den,1) ≈ lseval(A,I,B,C,D,1)

A, E, B, C, D, blkdims = rm2ls(num,den,minimal = true, atol = 1.e-7);
@test rmeval(num,den,1) ≈ lseval(A,E,B,C,D,1)

A, E, B, C, D, blkdims = rm2ls(num,den,minimal = true, contr = true, atol = 1.e-7);
@test rmeval(num,den,1) ≈ lseval(A,E,B,C,D,1)

A, E, B, C, D, blkdims = rm2ls(num,den,minimal = true, obs = true, atol = 1.e-7);
@test rmeval(num,den,1) ≈ lseval(A,E,B,C,D,1)


num1, den1 = ls2rm(A, E, B, C, D, atol1 = 1.e-7, atol2 = 1.e-7);
@test rmeval(num1,den1,1) ≈ lseval(A,E,B,C,D,1)

@test all(num .* pm2poly(den1,:s) .≈ pm2poly(num1,:s) .* den)

# Example 4.3 Antsaklis, Michel 2006
s = Polynomial([0, 1],:s)
num = [s^2+1 s+1]
den = [s^2 s^3]

A, B, C, D, blkdims = rm2lspm(num,den,contr=true)
@test rmeval(num,den,1) ≈ lseval(A,I,B,C,D,1)

A, B, C, D, blkdims = rm2lspm(num,den,obs=true)
@test rmeval(num,den,1) ≈ lseval(A,I,B,C,D,1)

# Example 4.3 (modified) Antsaklis, Michel 2006
s = Polynomial([0, 1],:s)
num = [s^3+1 s+1]
den = [s^2 s^3]

A, B, C, D, blkdims = rm2lspm(num,den,contr=true)
@test rmeval(num,den,1) ≈ lseval(A,I,B,C,zeros(1,2),1) + pmeval(D,1)

A, B, C, D, blkdims = rm2lspm(num,den,obs=true)
@test rmeval(num,den,1) ≈ lseval(A,I,B,C,zeros(1,2),1) + pmeval(D,1)


# Example 4.4 Antsaklis, Michel 2006
s = Polynomial([0, 1],:s)
num = [2 1; 1 0];
den = [s+1 1; s 1];

A, B, C, D, blkdims = rm2lspm(num,den)
@test rmeval(num,den,1) ≈ lseval(A,I,B,C,zeros(2,2),1) + pmeval(D,1)

A, B, C, D, blkdims = rm2lspm(num,den,contr=true)
@test rmeval(num,den,1) ≈ lseval(A,I,B,C,zeros(2,2),1) + pmeval(D,1)

A, B, C, D, blkdims = rm2lspm(num,den,obs=true)
@test rmeval(num,den,1) ≈ lseval(A,I,B,C,zeros(2,2),1) + pmeval(D,1)


s = Polynomial([0, 1],:s)
num = [2+s 1; 1 0];
den = [0.5 1; 0.5 1];

A, B, C, D, blkdims = rm2lspm(num,den)
@test rmeval(num,den,1) ≈ lseval(A,I,B,C,zeros(2,2),1) + pmeval(D,1)

A, B, C, D, blkdims = rm2lspm(num,den,contr=true)
@test rmeval(num,den,1) ≈ lseval(A,I,B,C,zeros(2,2),1) + pmeval(D,1)

A, B, C, D, blkdims = rm2lspm(num,den,obs=true)
@test rmeval(num,den,1) ≈ lseval(A,I,B,C,zeros(2,2),1) + pmeval(D,1)

A, B, C, D, blkdims = rm2lspm(num,obs=true)
@test rmeval(num,1) ≈ lseval(A,I,B,C,zeros(2,2),1) + pmeval(D,1)
@test rmeval(poly2pm(num),1) ≈ lseval(A,I,B,C,zeros(2,2),1) + pmeval(D,1)

# Example 4.4 (transposed) Antsaklis, Michel 2006
s = Polynomial([0, 1],:s)*im
num = [2 1; 1 0]
den = [s+1 s; 1 1]

A, B, C, D, blkdims = rm2lspm(num,den)
@test rmeval(num,den,1) ≈ lseval(A,I,B,C,zeros(2,2),1) + pmeval(D,1)

A, B, C, D, blkdims = rm2lspm(num,den,contr=true)
@test rmeval(num,den,1) ≈ lseval(A,I,B,C,zeros(2,2),1) + pmeval(D,1)

A, B, C, D, blkdims = rm2lspm(num,den,obs=true)
@test rmeval(num,den,1) ≈ lseval(A,I,B,C,zeros(2,2),1) + pmeval(D,1)




P = zeros(Int,3,3,3);
P[:,:,1] = [1 2 -2; 0 -1 -2; 0 0 0];
P[:,:,2] = [1 3 0; 1 4 2; 0 -1 -2];
P[:,:,3] = [1 4 2; 0 0 0; 1 4 2];

A, B, C, D, blkdims = rm2lspm(P)
@test D == P

A, E, B, C, D, blkdims = rm2ls(P,contr=true)
@test pmeval(P,1) ≈ lseval(A,E,B,C,D,1)



@test P ≈ ls2pm(rm2ls(P)[1:5]...)
@test P ≈ ls2rm(rm2ls(P)[1:5]...)[1]
N, D = ls2rm(rm2ls(P)[1:5]...);
@test all(pm2poly(P) .* pm2poly(D) .≈ pm2poly(N))
@test P ≈ lps2pm(rm2lps(P)[1:8]...;atol1=1.e-7,atol2=1.e-7)
@test P ≈ lps2rm(rm2lps(P)[1:8]...;atol1=1.e-7,atol2=1.e-7)[1]
N, D = lps2rm(rm2lps(P)[1:8]...);
@test all(pm2poly(P) .* pm2poly(D) .≈ pm2poly(N))


end

end  # testset

end # module