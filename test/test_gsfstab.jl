module Test_gsfstab


using LinearAlgebra
using MatrixPencils
using Test



function evsym!(ev) 
   ev[imag.(ev) .> 0] = conj(ev[imag.(ev) .< 0])
   return ev
end


@testset "Spectrum allocation functions" begin

@testset "saloc for standard pair (A,B)" begin


## simple cases: B = []; 
a = ones(1,1);  b = zeros(1,0);  evals = [-1]; 
@time f, SF, blkdims = saloc(a,b, evals =  evals)
@test SF.Z*SF.T*SF.Z' ≈ a+b*f && blkdims == [0, 0, 1]
 evals,eigvals(a+b*f)

## simple cases
a = ones(1,1);  b = zeros(1,5);  evals = [-1]; 
@time f, SF, blkdims = saloc(a,b, evals =  evals)
@test SF.Z*SF.T*SF.Z' ≈ a+b*f && blkdims == [0, 0, 1]
evals,eigvals(a+b*f)

## simple cases  
a = ones(1,1);  b = ones(1,1);  evals = [-1];
@time f, SF, blkdims = saloc(a,b, evals =  evals)
@test SF.Z*SF.T*SF.Z' ≈ a+b*f && blkdims == [0, 1, 0] &&
sort(real( evals)) ≈ sort(real(SF.values)) && sort(imag( evals)) ≈ sort(imag(SF.values)) 
evals,eigvals(a+b*f)

## simple cases
a = complex(ones(1,1));  b = ones(1,1);  evals = [-im];
@time f, SF, blkdims = saloc(a,b, evals =  evals)
@test SF.Z*SF.T*SF.Z' ≈ a+b*f && blkdims == [0, 1, 0] && 
      sort(real( evals)) ≈ sort(real(SF.values)) && sort(imag( evals)) ≈ sort(imag(SF.values)) 
evals,eigvals(a+b*f)

## simple cases (no real eigenvalue) 
a = ones(1,1);  b = ones(1,1);  evals = [-1+im*0.5;-1-im*0.5]; 
@time f, SF, blkdims = saloc(a,b, evals =  evals)
@test SF.Z*SF.T*SF.Z' ≈ a+b*f && blkdims == [0, 1, 0]
evals,eigvals(a+b*f)

## simple cases (no real eigenvalue, use sdeg)
a = ones(1,1);  b = ones(1,1);  evals = [-1+im*0.5;-1-im*0.5]; sdeg = -0.5;
@time f, SF, blkdims = saloc(a,b, evals =  evals, sdeg = sdeg)
@test SF.Z*SF.T*SF.Z' ≈ a+b*f && blkdims == [0, 1, 0]
evals,eigvals(a+b*f)

## simple cases
a = ones(1,1);  b = ones(1,1);  evals = [-1;-0.5]; 
@time f, SF, blkdims = saloc(a,b, evals =  evals)
@test SF.Z*SF.T*SF.Z' ≈ a+b*f && blkdims == [0, 1, 0]
evals,eigvals(a+b*f)


## simple cases
a = [1 0;0 1];  b = rand(2,5);  evals = [-1;-0.5]; 
@time f, SF, blkdims = saloc(a,b, evals =  evals)
@test SF.Z*SF.T*SF.Z' ≈ a+b*f  && blkdims == [0, 2, 0]
@test sort(real( evals)) ≈ sort(real(SF.values)) && sort(imag( evals)) ≈ sort(imag(SF.values)) 
evals,eigvals(a+b*f)

 ## simple cases  uncontrollable pair
a = [1 0;0 1];  b = rand(2,1);  evals = [-1;-0.5]; 
@time f, SF, blkdims = saloc(a,b, evals =  evals)
@test SF.Z*SF.T*SF.Z' ≈ a+b*f && blkdims == [0, 1, 1]
evals,eigvals(a+b*f)

 ## simple cases  controllable pair
 a = [1 0;0 2];  b = rand(2,1);  evals = [-1;-0.5]; 
 @time f, SF, blkdims = saloc(a,b, evals =  evals)
 @test SF.Z*SF.T*SF.Z' ≈ a+b*f && blkdims == [0, 2, 0] &&
       sort(real( evals)) ≈ sort(real(SF.values)) && sort(imag( evals)) ≈ sort(imag(SF.values)) 
evals,eigvals(a+b*f)

 ## simple cases  uncontrollable pair 
 a = [1 0;0 1];  b = rand(2,1);  evals = [-1+im*0.5;-1-im*0.5]; 
 @time f, SF, blkdims = saloc(a,b, evals =  evals)
 @test SF.Z*SF.T*SF.Z' ≈ a+b*f  && blkdims == [0, 1, 1]
 evals,eigvals(a+b*f)

## simple cases  controllable pair 
a = [1 0;0 2];  b = rand(2,1);  evals = [-1+im*0.5;-1-im*0.5]; 
@time f, SF, blkdims = saloc(a,b, evals =  evals)
@test SF.Z*SF.T*SF.Z' ≈ a+b*f && blkdims == [0, 2, 0] &&
      sort(real( evals)) ≈ sort(real(SF.values)) && sort(imag( evals)) ≈ sort(imag(SF.values)) 
evals,eigvals(a+b*f)

 
## simple cases  
a = [1 1;0 2];  b = rand(2,5);  evals = [-1+im*0.5;-1-im*0.5]; 
@time f, SF, blkdims = saloc(a,b, evals =  evals)
@test SF.Z*SF.T*SF.Z' ≈ a+b*f && blkdims == [0, 2, 0] &&
      sort(real( evals)) ≈ sort(real(SF.values)) && sort(imag( evals)) ≈ sort(imag(SF.values)) 
evals,eigvals(a+b*f)


## simple cases
a = [1 1 1;-1 1 0;0 0 2];  b = rand(3,5);  evals = [-1+im*0.5;-1-im*0.5;-2]; 
@time f, SF, blkdims = saloc(a,b, evals =  evals)
@test SF.Z*SF.T*SF.Z' ≈ a+b*f && blkdims == [0, 3, 0] && 
      sort(real( evals)) ≈ sort(real(SF.values)) && sort(imag( evals)) ≈ sort(imag(SF.values)) 
evals,eigvals(a+b*f)

## simple cases 
a = [1 1 0;-1 1 0;0 0 2];  b = [0 0; 0 0; 1 1];  evals = [-1+im*0.5;-1-im*0.5]; 
@time f, SF, blkdims = saloc(a,b, evals =  evals)
@test SF.Z*SF.T*SF.Z' ≈ a+b*f && blkdims == [0, 1, 2] 
evals,eigvals(a+b*f)

## simple cases 
a = [1 1; 0 1 ];  b = [0 0; 1 1];  evals = [-1+im*0.5;-1-im*0.5]; 
@time f, SF, blkdims = saloc(a,b, evals =  evals)
@test SF.Z*SF.T*SF.Z' ≈ a+b*f && blkdims == [0, 2, 0] 
evals,eigvals(a+b*f)

## simple cases
a = [1 1 1;-1 1 0;0 0 2];  b = rand(3,5);  evals = [-1;-0.5;-2]; 
@time f, SF, blkdims = saloc(a,b, evals =  evals)
@test SF.Z*SF.T*SF.Z' ≈ a+b*f && blkdims == [0, 3, 0] &&
      sort(real( evals)) ≈ sort(real(SF.values)) && sort(imag( evals)) ≈ sort(imag(SF.values)) 
evals,eigvals(a+b*f)


## simple cases
a = [1 1 1;-1 1 0;0 0 2];  b = rand(3,5); sdeg = -0.5;
@time f, SF, blkdims = saloc(a,b,sdeg = sdeg)
@test SF.Z*SF.T*SF.Z' ≈ a+b*f && blkdims == [0, 3, 0]  &&
      all(sdeg .≈ (real(SF.values)))
sdeg,eigvals(a+b*f)

## simple cases
a = [1 1 1;-1 1 0;0 0 2];  b = rand(3,5);  evals = [-1,-0.5]; sdeg = -0.5;
@time f, SF, blkdims = saloc(a,b, evals =  evals, sdeg = sdeg)
@test SF.Z*SF.T*SF.Z' ≈ a+b*f && blkdims == [0, 3, 0]
sdeg, evals, eigvals(a+b*f)

## simple cases
a = [1 1 1;-1 1 0;0 0 2];  b = rand(3,5);  evals = [-3]; sdeg = -2;
@time f, SF, blkdims = saloc(a,b, evals =  evals, sdeg = sdeg)
@test SF.Z*SF.T*SF.Z' ≈ a+b*f && blkdims == [0, 3, 0]
sdeg, evals, eigvals(a+b*f)


## SB01BD EXAMPLE PROGRAM DATA
a = [
  -6.8000   0.0000  -207.0000   0.0000
   1.0000   0.0000     0.0000   0.0000
  43.2000   0.0000     0.0000  -4.2000
   0.0000   0.0000     1.0000   0.0000];
b = [
   5.6400   0.0000
   0.0000   0.0000
   0.0000   1.1800
   0.0000   0.0000]; 
 evals = [
  -0.5000+im*0.1500
  -0.5000-im*0.1500
  -2.0000
  -0.4000];
sdeg = -0.4;
e = I;
@time f, SF, blkdims = saloc(a,b, evals =  evals, sdeg = sdeg)
@test SF.Z*SF.T*SF.Z' ≈ a+b*f && blkdims == [2, 2, 0]
sdeg, evals, eigvals(a+b*f)

sdeg = missing;
@time f, SF, blkdims = saloc(a,b, evals =  evals, sdeg = sdeg)
@test SF.Z*SF.T*SF.Z' ≈ a+b*f && blkdims == [0, 4, 0] && 
      sort(real( evals)) ≈ sort(real(SF.values)) && sort(imag( evals)) ≈ sort(imag(SF.values)) 
sdeg, evals, eigvals(a+b*f)


##  SB01DD EXAMPLE PROGRAM DATA
#   for this example PLACE fails
a = [
  -1.0  0.0  2.0 -3.0
   1.0 -4.0  3.0 -1.0
   0.0  2.0  4.0 -5.0
   0.0  0.0 -1.0 -2.0];
b = [
   1.0  0.0
   0.0  0.0
   0.0  0.0
   0.0  1.0];
 evals = [
  -1.0, -1.0, -1.0, -1.0];
sdeg = [];
@time f, SF, blkdims = saloc(a,b, evals =  evals)
@test SF.Z*SF.T*SF.Z' ≈ a+b*f && blkdims == [0, 4, 0] && 
      sort(real( evals)) ≈ sort(real(SF.values)) && sort(imag( evals)) ≈ sort(imag(SF.values)) 
 evals, eigvals(a+b*f)

# random examples
Ty = Float64      
for Ty in (Float64, Complex{Float64})

a = rand(Ty,6,6); b = rand(Ty,6,3);   evals = eigvals(a); 
@time f, SF, blkdims = saloc(a,b, evals =  evals)
@test SF.Z*SF.T*SF.Z' ≈ a+b*f  && norm(f) < 1.e-10 && blkdims == [0, 6, 0] &&
      sort(real( evals)) ≈ sort(real(SF.values)) && sort(imag( evals)) ≈ sort(imag(SF.values)) 

# random example  
a = rand(Ty,6,6); b = [zeros(Ty,2,3);rand(Ty,2,3);zeros(Ty,2,3)];   evals = eigvals(a); evals = evals[sortperm(rand(6))]; 
@time f, SF, blkdims = saloc(a,b, evals =  evals)
@test SF.Z*SF.T*SF.Z' ≈ a+b*f  && norm(f) < 1.e-10 && blkdims == [0, 6, 0] &&
      sort(real( evals)) ≈ sort(real(SF.values)) && sort(imag( evals)) ≈ sort(imag(SF.values)) 

# random example  
a = rand(Ty,6,6); c = [zeros(Ty,3,2) rand(Ty,3,2) zeros(Ty,3,2)];   evals = eigvals(a); 
@time k, SF, blkdims = salocd(a, c, evals =  evals)
@test SF.Z*SF.T*SF.Z' ≈ a+k*c  && norm(k) < 1.e-10 && blkdims == [0, 6, 0] &&
      sort(real( evals)) ≈ sort(real(SF.values)) && sort(imag( evals)) ≈ sort(imag(SF.values))     

# random example  - stabilization
a = randn(Ty,6,6); b = rand(Ty,6,3);   sdeg = -0.2;  
@time f, SF, blkdims = saloc(a,b, sdeg = sdeg)
@test SF.Z*SF.T*SF.Z' ≈ a+b*f  && sum(blkdims[1:2]) == 6 &&
      all(real(SF.values) .<= sdeg+sqrt(eps()))  

# random example  - stabilization
a = randn(Ty,6,6); b = rand(Ty,6,3);    
@time f, SF, blkdims = saloc(a,b)
@test SF.Z*SF.T*SF.Z' ≈ a+b*f  && sum(blkdims[1:2]) == 6 &&
      all(real(SF.values) .<= 0)  

# random example  - stabilization
a = rand(Ty,6,6); b = rand(Ty,6,3);   sdeg = 0.2;  
@time f, SF, blkdims = saloc(a,b, sdeg = sdeg, disc = true)
@test SF.Z*SF.T*SF.Z' ≈ a+b*f  && sum(blkdims[1:2]) == 6 &&
      all(abs.(SF.values) .<= sdeg+sqrt(eps()))  

# random example  - stabilization
a = randn(Ty,6,6); b = rand(Ty,6,3);    
@time f, SF, blkdims = saloc(a,b, disc = true)
@test SF.Z*SF.T*SF.Z' ≈ a+b*f  && sum(blkdims[1:2]) == 6 &&
      all(abs.(SF.values) .<= 1)  

# random example - stabilization
a = randn(Ty,6,6); b = rand(Ty,6,3);   sdeg = -0.2; 
@time f, SF, blkdims = saloc(a,b, sdeg = sdeg)
@test SF.Z*SF.T*SF.Z' ≈ a+b*f  && sum(blkdims[1:2]) == 6 &&
      all(real(SF.values) .<= sdeg+sqrt(eps()))

# random example - stabilization
a = randn(Ty,6,6); b = rand(Ty,6,3);    
@time f, SF, blkdims = saloc(a,b)
@test SF.Z*SF.T*SF.Z' ≈ a+b*f  && sum(blkdims[1:2]) == 6 &&
      all(real(SF.values) .<= 0)

# random example - stabilization  
a = randn(Ty,6,6); b = rand(Ty,6,3);    
@time f, SF, blkdims = saloc(a,b,evals = [-1, -2])
@test SF.Z*SF.T*SF.Z' ≈ a+b*f  && sum(blkdims[1:2]) == 6 &&
      all(real(SF.values) .<= 0)

# random example - stabilization 
a = randn(Ty,6,6); b = rand(Ty,6,3);    
@time f, SF, blkdims = saloc(a,b,evals = [-1])
@test SF.Z*SF.T*SF.Z' ≈ a+b*f  && sum(blkdims[1:2]) == 6 &&
      all(real(SF.values) .<= 0)


# random example - stabilization 
a = rand(Ty,6,6); b = rand(Ty,6,3);  sdeg = 0.2 
@time f, SF, blkdims = saloc(a,b, sdeg = sdeg, disc = true)
@test SF.Z*SF.T*SF.Z' ≈ a+b*f  && sum(blkdims[1:2]) == 6 &&
      all(abs.(SF.values) .<= sdeg+sqrt(eps()))

a = rand(Ty,6,6); b = rand(Ty,6,3);   
@time f, SF, blkdims = saloc(a,b, disc = true)
@test SF.Z*SF.T*SF.Z' ≈ a+b*f  && sum(blkdims[1:2]) == 6 &&
      all(abs.(SF.values) .<= 1)
      
##  uncontrollable system   
n = 10; nc = 6; nu = n-nc; m = 4; 
au = rand(Ty,nu,nu); ac = rand(Ty,nc,nc);  
a = [ac rand(Ty,nc,nu); zeros(Ty,nu,nc) au]; b = [rand(Ty,nc,m);zeros(Ty,nu,m)];
q = qr(rand(Ty,n,n)).Q; 
poluc = eigvals(au);
a = q'*a*q; b = q'*b;
#evals = complex(rand(nc))
evals = eigvals(rand(Ty,nc,nc))
@time f, SF, blkdims = saloc(a,b, evals =  evals, atol1 = 1.e-7, atol2 = 1.e-7)
@test SF.Z*SF.T*SF.Z' ≈ a+b*f  && blkdims == [0, nc, nu] && 
      sort(real(poluc)) ≈ sort(real(SF.values[nc+1:n])) && sort(imag(poluc)) ≈ sort(imag(SF.values[nc+1:n])) &&
      sort(real(evals)) ≈ sort(real(SF.values[1:nc])) && sort(imag( evals)) ≈ sort(imag(SF.values[1:nc])) 
end # Ty loop

end #begin

@testset "saloc for generalized pair (A-λE,B)" begin

## simple cases: B = []; 
a = ones(1,1); e = I; b = zeros(1,0); evals = [-1]; 
@time f, SF, blkdims = saloc(a,e,b,evals = evals)
@test SF.Q*SF.S*SF.Z' ≈ a+b*f && SF.Q*SF.T*SF.Z' ≈ e  && blkdims == [0, 0, 0, 1]
evals,eigvals(a+b*f)

## simple cases: B = []; 
a = ones(1,1); e = 2*ones(1,1); b = zeros(1,0); evals = [-1]; 
@time f, SF, blkdims = saloc(a,e,b,evals = evals)
@test SF.Q*SF.S*SF.Z' ≈ a+b*f && SF.Q*SF.T*SF.Z' ≈ e  && blkdims == [0, 0, 0, 1]
evals,eigvals(a+b*f,e)

## simple cases
a = ones(1,1); e = 2*ones(1,1); b = zeros(1,5); evals = [-1]; 
@time f, SF, blkdims = saloc(a,e,b,evals = evals)
@test SF.Q*SF.S*SF.Z' ≈ a+b*f && SF.Q*SF.T*SF.Z' ≈ e  && blkdims == [0, 0, 0, 1]
evals,eigvals(a+b*f,e)

## simple cases
a = ones(1,1); e = 2*ones(1,1); b = ones(1,1); evals = [-1];
@time f, SF, blkdims = saloc(a,e,b,evals = evals)
@test SF.Q*SF.S*SF.Z' ≈ a+b*f && SF.Q*SF.T*SF.Z' ≈ e  && blkdims == [0, 0, 1, 0]
@test sort(real(evals)) ≈ sort(real(SF.α ./SF.β)) && sort(imag(evals)) ≈ sort(imag(SF.α ./SF.β)) 
evals,eigvals(a+b*f,e)

## simple cases
a = ones(1,1); e = 2*ones(1,1); b = ones(1,1); evals = [-1];
@time f, SF, blkdims = saloc(a,e,b,evals = evals,sepinf = false)
@test SF.Q*SF.S*SF.Z' ≈ a+b*f && SF.Q*SF.T*SF.Z' ≈ e  && blkdims == [0, 0, 1, 0] &&
    sort(real(evals)) ≈ sort(real(SF.α ./SF.β)) && sort(imag(evals)) ≈ sort(imag(SF.α ./SF.β)) 
evals,eigvals(a+b*f,e)


## simple cases
a = complex(ones(1,1)); e = 2*ones(1,1); b = ones(1,1); evals = [-im];
@time f, SF, blkdims = saloc(a,e,b,evals = evals)
@test SF.Q*SF.S*SF.Z' ≈ a+b*f && SF.Q*SF.T*SF.Z' ≈ e  && blkdims == [0, 0, 1, 0] &&
      sort(real(evals)) ≈ sort(real(SF.α ./SF.β)) && sort(imag(evals)) ≈ sort(imag(SF.α ./SF.β)) 
evals,eigvals(a+b*f,e)

## simple cases
a = complex(ones(1,1)); e = 2*ones(1,1); b = ones(1,1); evals = [-im];
@time f, SF, blkdims = saloc(a,e,b,evals = evals,sepinf = false)
@test SF.Q*SF.S*SF.Z' ≈ a+b*f && SF.Q*SF.T*SF.Z' ≈ e  && blkdims == [0, 0, 1, 0] && 
      sort(real(evals)) ≈ sort(real(SF.α ./SF.β)) && sort(imag(evals)) ≈ sort(imag(SF.α ./SF.β)) 
evals,eigvals(a+b*f,e)

## simple cases (no real eigenvalue)
a = ones(1,1); e = 2*ones(1,1); b = ones(1,1); evals = [-1+im*0.5;-1-im*0.5]; 
@time f, SF, blkdims = saloc(a,e,b,evals = evals)
@test SF.Q*SF.S*SF.Z' ≈ a+b*f && SF.Q*SF.T*SF.Z' ≈ e  && blkdims == [0, 0, 1, 0]
evals,eigvals(a+b*f,e)

## simple cases (no real eigenvalue, use sdeg)
a = ones(1,1); e = 2*ones(1,1);  b = ones(1,1); evals = [-1+im*0.5;-1-im*0.5]; sdeg = -0.5;
@time f, SF, blkdims = saloc(a,e,b,evals = evals, sdeg = sdeg)
@test SF.Q*SF.S*SF.Z' ≈ a+b*f && SF.Q*SF.T*SF.Z' ≈ e  && blkdims == [0, 0, 1, 0]
evals,eigvals(a+b*f,e)

## simple cases
a = ones(1,1); e = 2*ones(1,1);  b = ones(1,1); evals = [-1;-0.5]; 
@time f, SF, blkdims = saloc(a,e,b,evals = evals)
@test SF.Q*SF.S*SF.Z' ≈ a+b*f && SF.Q*SF.T*SF.Z' ≈ e  && blkdims == [0, 0, 1, 0]
evals,eigvals(a+b*f,e)


## simple cases  
a = [1 0;0 1]; e = rand(2,2); b = rand(2,5); evals = [-1;-0.5]; 
e =  [0.0568026  0.988833
0.0651635  0.862077];
b = [0.338582  0.74765  0.314715  0.265301  0.749282 
0.221894  0.44414  0.910163  0.994525  0.0763453]
@time f, SF, blkdims = saloc(a,e,b,evals = evals)
@test SF.Q*SF.S*SF.Z' ≈ a+b*f && SF.Q*SF.T*SF.Z' ≈ e && blkdims == [0, 0, 2, 0]
@test sort(real(evals)) ≈ sort(real(SF.α ./SF.β)) && sort(imag(evals)) ≈ sort(imag(SF.α ./SF.β)) 
evals,eigvals(a+b*f,e)

## simple cases  
a = [1 0;0 1]; e = rand(2,2); b = rand(2,5); evals = [-1;-0.5]; 
e =  [0.0568026  0.988833
0.0651635  0.862077];
b = [0.338582  0.74765  0.314715  0.265301  0.749282 
0.221894  0.44414  0.910163  0.994525  0.0763453]
@time f, SF, blkdims = saloc(a,e,b,evals = evals, sepinf = false)
@test SF.Q*SF.S*SF.Z' ≈ a+b*f && SF.Q*SF.T*SF.Z' ≈ e && blkdims == [0, 0, 2, 0] &&
      sort(real(evals)) ≈ sort(real(SF.α ./SF.β)) && sort(imag(evals)) ≈ sort(imag(SF.α ./SF.β)) 

## simple cases  
a = [1 0;0 1]; e = rand(2,2); b = rand(2,5); evals = [-1;-0.5]; 
e =  [0.0568026  0.988833
0  0.862077];
b = [0.338582  0.74765  0.314715  0.265301  0.749282 
0.221894  0.44414  0.910163  0.994525  0.0763453]
@time f, SF, blkdims = saloc(a,e,b,evals = evals, sepinf = false)
@test SF.Q*SF.S*SF.Z' ≈ a+b*f && SF.Q*SF.T*SF.Z' ≈ e && blkdims == [0, 0, 2, 0] &&
      sort(real(evals)) ≈ sort(real(SF.α ./SF.β)) && sort(imag(evals)) ≈ sort(imag(SF.α ./SF.β)) 

      
## simple cases  uncontrollable pair 
a = rand(2,2); e = a;  b = rand(2,1);  evals = [-1+im*0.5;-1-im*0.5]; 
@time f, SF, blkdims = saloc(a, e, b, evals =  evals)
@test SF.Q*SF.S*SF.Z' ≈ a+b*f && SF.Q*SF.T*SF.Z' ≈ e && blkdims == [0, 0, 1, 1]
evals,eigvals(a+b*f,e)


## simple cases  
a = [1 1;0 2]; e = rand(2,2); b = rand(2,5); evals = [-1+im*0.5;-1-im*0.5]; 
@time f, SF, blkdims = saloc(a,e,b,evals = evals)
@test SF.Q*SF.S*SF.Z' ≈ a+b*f && SF.Q*SF.T*SF.Z' ≈ e && blkdims == [0, 0, 2, 0] &&
      sort(real(evals)) ≈ sort(real(SF.α ./SF.β)) && sort(imag(evals)) ≈ sort(imag(SF.α ./SF.β)) 
evals,eigvals(a+b*f,e)

## simple cases 
e = rand(3,3); a = e*[1 1 0;-1 1 0;0 0 2];  b = e*[0 0; 0 0; 1 1];  evals = [-1+im*0.5;-1-im*0.5]; 
@time f, SF, blkdims = saloc(a, e, b, evals =  evals)
@test SF.Q*SF.S*SF.Z' ≈ a+b*f && SF.Q*SF.T*SF.Z' ≈ e && blkdims == [0, 0, 1, 2] 
evals,eigvals(a+b*f,e)


## simple cases
e = rand(3,3); a = e*[1 1 1;-1 1 0;0 0 2]; b = rand(3,5); evals = [-1+im*0.5;-1-im*0.5;-2]; 
@time f, SF, blkdims = saloc(a,e,b,evals = evals)
@test SF.Q*SF.S*SF.Z' ≈ a+b*f && SF.Q*SF.T*SF.Z' ≈ e && blkdims == [0, 0, 3, 0] &&
      sort(real(evals)) ≈ sort(real(SF.α ./SF.β)) && sort(imag(evals)) ≈ sort(imag(SF.α ./SF.β)) 
evals,eigvals(a+b*f,e)

## simple cases
e = [1 1 1;-1 1 0;0 0 0]; a = [1 1 1;-1 1 0;0 0 2]; b = rand(3,5); evals = [-1+im*0.5;-1-im*0.5;-2]; 
@time f, SF, blkdims = saloc(a,e,b,evals = evals)
@test SF.Q*SF.S*SF.Z' ≈ a+b*f && SF.Q*SF.T*SF.Z' ≈ e && blkdims == [1, 0, 2, 0]
evals,eigvals(a+b*f,e)

## simple cases
e = [1 1 1;-1 1 0;0 0 0]; a = [1 1 1;-1 1 0;0 0 2]; b = rand(3,5); evals = [-1+im*0.5;-1-im*0.5;-2]; 
@time f, SF, blkdims = saloc(a,e,b,evals = evals, sepinf = false) 
@test SF.Q*SF.S*SF.Z' ≈ a+b*f && SF.Q*SF.T*SF.Z' ≈ e && blkdims == [1, 0, 2, 0]
evals,eigvals(a+b*f,e)


## simple cases  
#e = rand(3,3); 
e = [
 0.521866  0.369606  0.456723
 0.740103  0.357063  0.280376
 0.397995  0.362346  0.885047];
a = e*[1 1 1;-1 1 0;0 0 2]; b = rand(3,5); evals = [-1;-0.5;-2]; 
@time f, SF, blkdims = saloc(a,e,b,evals = evals)
#@time f, SF, blkdims = saloc(a,e,b,evals = evals, sepinf = false)

@test SF.Q*SF.S*SF.Z' ≈ a+b*f && SF.Q*SF.T*SF.Z' ≈ e && blkdims == [0, 0, 3, 0] &&
      sort(real(evals)) ≈ sort(real(SF.α ./SF.β)) && sort(imag(evals)) ≈ sort(imag(SF.α ./SF.β)) 


## simple cases
e = [1 1 1;-1 1 0;0 0 0]; a = [1 1 1;-1 1 0;0 0 2]; b = rand(3,5); evals = [-1;-0.5;-2]; 
@time f, SF, blkdims = saloc(a,e,b,evals = evals, sepinf = false) 
@test SF.Q*SF.S*SF.Z' ≈ a+b*f && SF.Q*SF.T*SF.Z' ≈ e && blkdims == [1, 0, 2, 0]
evals,eigvals(a+b*f,e)


## simple cases
e = rand(3,3); a = e*[1 1 1;-1 1 0;0 0 2]; b = rand(3,5); sdeg = -0.5;
@time f, SF, blkdims = saloc(a,e,b,sdeg = sdeg)
@test SF.Q*SF.S*SF.Z' ≈ a+b*f && SF.Q*SF.T*SF.Z' ≈ e  && blkdims == [0, 0, 3, 0] &&
      all(sdeg .≈ (real(SF.values)))
sdeg, eigvals(a+b*f,e) 

## simple cases
e = rand(3,3); a = e*[1 1 1;-1 1 0;0 0 2];  b = rand(3,5); evals = [-1,-0.5]; sdeg = -0.5;
@time f, SF, blkdims = saloc(a,e,b,evals = evals, sdeg = sdeg)
@test SF.Q*SF.S*SF.Z' ≈ a+b*f && SF.Q*SF.T*SF.Z' ≈ e  && blkdims == [0, 0, 3, 0]
sdeg, eigvals(a+b*f,e) 

## simple cases
e = rand(3,3); a = e*[1 1 1;-1 1 0;0 0 2]; b = rand(3,5); evals = [-3]; sdeg = -2;
@time f, SF, blkdims = saloc(a,e,b,evals = evals, sdeg = sdeg)
@test SF.Q*SF.S*SF.Z' ≈ a+b*f && SF.Q*SF.T*SF.Z' ≈ e && blkdims == [0, 0, 3, 0]
sdeg, eigvals(a+b*f,e) 

## simple cases 
e = rand(3,3); a = e*[1 1 0;-1 1 0;0 0 2];  b = e*[0 0; 0 0; 1 1];  evals = [-1+im*0.5;-1-im*0.5]; 
@time f, SF, blkdims = saloc(a, e, b, atol1 = 1.e-7, atol2 = 1.e-7, atol3 = 1.e-7, evals =  evals)
@test SF.Q*SF.S*SF.Z' ≈ a+b*f && SF.Q*SF.T*SF.Z' ≈ e  && blkdims == [0, 0, 1, 2] 
evals,eigvals(a+b*f,e)


## SB01BD EXAMPLE PROGRAM DATA
a = [
  -6.8000   0.0000  -207.0000   0.0000
   1.0000   0.0000     0.0000   0.0000
  43.2000   0.0000     0.0000  -4.2000
   0.0000   0.0000     1.0000   0.0000];
b = [
   5.6400   0.0000
   0.0000   0.0000
   0.0000   1.1800
   0.0000   0.0000]; 
evals = [
  -0.5000+im*0.1500
  -0.5000-im*0.1500
  -2.0000
  -0.4000];
sdeg = -0.4;
e = rand(4,4); a = e*a; b = e*b;
@time f, SF, blkdims = saloc(a,e,b,evals = evals, sdeg = sdeg)
@test SF.Q*SF.S*SF.Z' ≈ a+b*f && SF.Q*SF.T*SF.Z' ≈ e  && blkdims == [0, 2, 2, 0]
sdeg,evals, eigvals(a+b*f,e)

sdeg = missing;
@time f, SF, blkdims = saloc(a,e,b,evals = evals, sdeg = sdeg)
@test SF.Q*SF.S*SF.Z' ≈ a+b*f && SF.Q*SF.T*SF.Z' ≈ e && blkdims == [0, 0, 4, 0] && 
      sort(real(evals)) ≈ sort(real(SF.values)) && sort(imag(evals)) ≈ sort(imag(SF.values)) 
sdeg, evals, eigvals(a+b*f,e)


##  SB01DD EXAMPLE PROGRAM DATA
#   for this example PLACE fails
a = [
  -1.0  0.0  2.0 -3.0
   1.0 -4.0  3.0 -1.0
   0.0  2.0  4.0 -5.0
   0.0  0.0 -1.0 -2.0];
b = [
   1.0  0.0
   0.0  0.0
   0.0  0.0
   0.0  1.0];
evals = [ -1.0, -1.0, -1.0, -1.0];
sdeg = missing;
e = rand(4,4); a = e*a; b = e*b;
@time f, SF, blkdims = saloc(a,e,b,evals = evals)
@test SF.Q*SF.S*SF.Z' ≈ a+b*f && SF.Q*SF.T*SF.Z' ≈ e  && blkdims == [0, 0, 4, 0]
@test sort(real(evals)) ≈ sort(real(SF.values)) && sort(imag(evals)) ≈ sort(imag(SF.values)) 
sdeg, evals, eigvals(a+b*f,e)

# random example
fast = true
for fast in (true,false)

Ty = Float64      
for Ty in (Float64, Complex{Float64})
complx = Ty <: Complex

a = rand(Ty,6,6); b = rand(Ty,6,3); e = rand(Ty,6,6); 
evals = complx ? eigvals(a,e) : evsym!(eigvals(a,e)); 
@time f, SF, blkdims = saloc(a,e,b,evals = evals, fast = fast)
@test SF.Q*SF.S*SF.Z' ≈ a+b*f && SF.Q*SF.T*SF.Z' ≈ e  && norm(f) < 1.e-10  && blkdims == [0, 0, 6, 0] && 
      sort(real(evals)) ≈ sort(real(SF.values)) && sort(imag(evals)) ≈ sort(imag(SF.values)) 

# random example  
a = rand(Ty,6,6); b = [zeros(Ty,2,3);rand(Ty,2,3);zeros(Ty,2,3)]; e = rand(Ty,6,6); 
evals = complx ? eigvals(a,e) : evsym!(eigvals(a,e)); 
@time f, SF, blkdims = saloc(a,e,b,evals = evals, fast = fast)
@test SF.Q*SF.S*SF.Z' ≈ a+b*f && SF.Q*SF.T*SF.Z' ≈ e  && norm(f) < 1.e-10  && blkdims == [0, 0, 6, 0] && 
      sort(real(evals)) ≈ sort(real(SF.values)) && sort(imag(evals)) ≈ sort(imag(SF.values)) 

# random example  - stabilization 
a = randn(Ty,6,6); e = randn(Ty,6,6); b = rand(Ty,6,3);    
@time f, SF, blkdims = saloc(a,e,b,evals = [-1, -2])
@test SF.Q*SF.S*SF.Z' ≈ a+b*f && SF.Q*SF.T*SF.Z' ≈ e  && sum(blkdims[2:3]) == 6  &&
      all(real(SF.values) .<= 0)  

@time f, SF, blkdims = saloc(a,b,evals = [-1, -2])


# random example  - stabilization 
a = randn(Ty,6,6); e = randn(Ty,6,6); b = rand(Ty,6,3);    
@time f, SF, blkdims = saloc(a,e,b,disc = true, sepinf = false)
@test SF.Q*SF.S*SF.Z' ≈ a+b*f && SF.Q*SF.T*SF.Z' ≈ e  && sum(blkdims[2:3]) == 6  &&
      all(abs.(SF.values) .<= 1)  

# random example  - stabilization 
a = randn(Ty,6,6); e = randn(Ty,6,6); b = rand(Ty,6,3);    
@time f, SF, blkdims = saloc(a,e,b,disc = true, sepinf = true)
@test SF.Q*SF.S*SF.Z' ≈ a+b*f && SF.Q*SF.T*SF.Z' ≈ e  && sum(blkdims[2:3]) == 6  &&
      all(abs.(SF.values) .<= 1)  


# random example  
a = rand(Ty,6,6); c = [zeros(Ty,3,2) rand(Ty,3,2) zeros(Ty,3,2)]; e = rand(Ty,6,6); 
evals = complx ? eigvals(a,e) : evsym!(eigvals(a,e)); 
@time k, SF, blkdims = salocd(a,e,c,evals = evals, fast = fast)
@test SF.Q*SF.S*SF.Z' ≈ a+k*c && SF.Q*SF.T*SF.Z' ≈ e  && norm(k) < 1.e-10  && blkdims == [0, 6, 0, 0] && 
      sort(real(evals)) ≈ sort(real(SF.values)) && sort(imag(evals)) ≈ sort(imag(SF.values)) 

##  uncontrollable system 
n = 10; nc = 6; nu = n-nc; m = 4; 
au = rand(Ty,nu,nu); ac = rand(Ty,nc,nc); eu = rand(Ty,nu,nu); ec = rand(Ty,nc,nc);
a = [ac rand(Ty,nc,nu); zeros(Ty,nu,nc) au]; b = [rand(Ty,nc,m);zeros(Ty,nu,m)];
e = [ec rand(Ty,nc,nu); zeros(Ty,nu,nc) eu];
q = qr(rand(Ty,n,n)).Q; z = qr(rand(Ty,n,n)).Q; 
poluc = eigvals(au,eu);
a = q'*a*z; e = q'*e*z; b = q'*b;
evals = eigvals(rand(Ty,nc,nc));
@time f, SF, blkdims = saloc(a,e,b,evals = evals, fast = fast, atol1 = 1.e-7, atol2 = 1.e-7, atol3 = 1.e-7)
@test SF.Q*SF.S*SF.Z' ≈ a+b*f && SF.Q*SF.T*SF.Z' ≈ e  && blkdims == [0, 0, nc, nu] &&
      sort(real(poluc)) ≈ sort(real(SF.values[nc+1:n])) && sort(imag(poluc)) ≈ sort(imag(SF.values[nc+1:n])) &&
      sort(real(evals)) ≈ sort(real(SF.values[1:nc])) && sort(imag(evals)) ≈ sort(imag(SF.values[1:nc])) 

end # Ty loop

for sepinf in (true,false)


## proper system  
n = 10;  ns = 3; m = 4; 
e = [zeros(n,ns) randn(n,n-ns)]'; a = randn(n,n); b = randn(n,m);
evals = complex(randn(n-ns))
nc = 2*floor(Int,floor((n-ns+2)/2)*rand())
for ii = 1 : 2 : nc,
   evals[ii]   = evals[ii] + im*evals[ii+1];
   evals[ii+1] = conj( evals[ii] );
end
@time f, SF, blkdims = saloc(a,e,b,evals = evals, sepinf = sepinf, fast = fast)
@test SF.Q*SF.S*SF.Z' ≈ a+b*f && SF.Q*SF.T*SF.Z' ≈ e  && blkdims == [ns, 0, n-ns, 0] &&
      sort(real(evals)) ≈ sort(real(SF.values[ns+1:n])) && sort(imag(evals)) ≈ sort(imag(SF.values[ns+1:n])) 

## Simas' paper (standard system)  
n = 10; m = 4; 
a = randn(n,n); b = randn(n,m);
evals = complex(randn(n));
nc = 2*floor(Int,floor((n+2)/2)*rand())
for ii = 1 : 2 : nc,
   evals[ii]   = evals[ii] + im*evals[ii+1];
   evals[ii+1] = conj( evals[ii] );
end
@time f, SF, blkdims = saloc(a,I,b,evals = evals, sepinf = sepinf, fast = fast);
@test SF.Q*SF.S*SF.Z' ≈ a+b*f && SF.Q*SF.T*SF.Z' ≈ I  && blkdims == [0, 0, n, 0] &&
      sort(real(evals)) ≈ sort(real(SF.values[1:n])) && sort(imag(evals)) ≈ sort(imag(SF.values[1:n])) 

## Sima's paper (descriptor system)
n = 10;  m = 4; 
a = randn(n,n);  e = randn(n,n);  b = randn(n,m);
evals = complex(randn(n));
nc = 2*floor(Int,floor((n+2)/2)*rand())
for ii = 1 : 2 : nc,
   evals[ii]   = evals[ii] + im*evals[ii+1];
   evals[ii+1] = conj( evals[ii] );
end
@time f, SF, blkdims = saloc(a,e,b,evals = evals, sepinf = sepinf, fast = fast)
@test SF.Q*SF.S*SF.Z' ≈ a+b*f && SF.Q*SF.T*SF.Z' ≈ e  && blkdims == [0, 0, n, 0] &&
      sort(real(evals)) ≈ sort(real(SF.values[1:n])) && sort(imag(evals)) ≈ sort(imag(SF.values[1:n])) 


## improper system R-stabilization  
ni = [3, 2, 1, 1]; nf = 5; ninf = sum(ni); n = ninf+nf; m = 4;
a = [triu(rand(ninf,n)); zeros(nf,ninf) rand(nf,nf)]; 
ei = rand(ninf,n); b = randn(n,m);
k1 = 0; 
k2 = k1 + ni[1]
ei[k1+1:ninf,1:k2] .= 0
k1 = k2
k2 = k1 + ni[2]
ei[k1+1:ninf,1:k2] .= 0
k1 = k2
k2 = k1 + ni[3]
ei[k1+1:ninf,1:k2] .= 0
k1 = k2
k2 = k1 + ni[4]
ei[k1+1:ninf,1:k2] .= 0

e = [ei; zeros(nf,ninf) rand(nf,nf)]; 
q = qr(rand(n,n)).Q; z = qr(rand(n,n)).Q; 
a = q*a*z; e = q*e*z;

evals = complex(-rand(nf) .- 1);
nc = 2*floor(Int,floor((n-ninf+2)/2)*rand())
for ii = 1 : 2 : nc,
   evals[ii]   = evals[ii] + im*evals[ii+1];
   evals[ii+1] = conj( evals[ii] );
end
@time f, SF, blkdims = saloc(a,e,b,evals = evals, fast = fast, sepinf = sepinf, atol1 = 1.e-7, atol2 = 1.e-7, atol3 = 1.e-7)
@test SF.Q*SF.S*SF.Z' ≈ a+b*f && SF.Q*SF.T*SF.Z' ≈ e  && blkdims == [ninf, 0, nf, 0] && 
      sort(real(evals)) ≈ sort(real(SF.values[ninf+1:n])) && sort(imag(evals)) ≈ sort(imag(SF.values[ninf+1:n])) 

end # sepinf loop
end # fast loop

end #begin

end #test

end #module





