module Test_sklfapps

using Random
using LinearAlgebra
using MatrixPencils
using Test


@testset "Structured Matrix Pencils Applications" begin

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

# Kronecker structure
kinfo  = spkstruct(A, E, B, C, D, atol1 = 1.e-7, atol2 = 1.e-7)
@test (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == (Int64[], [2], [2, 2], 2, 8)

kinfo  = spkstruct(A, E, view(B,:,1), C, view(D,:,1), atol1 = 1.e-7, atol2 = 1.e-7)
@test (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == (Int64[], [1,2], [2], 2, 7)

# zeros
@time val, iz, kinfo  = spzeros(A, E, B, C, D)
@test sort(real(val)) ≈ [-1, 2, Inf,Inf] && iz == [1, 1] && 
      (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == (Int64[], [2], [2, 2], 2, 8)

@time val, iz, kinfo  = spzeros(A, E, view(B,:,1), C, view(D,:,1))
@test sort(real(val)) ≈ [-1, 2, Inf] && iz == [1] && 
      (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == (Int64[], [1, 2], [2], 2, 7)

@time val, kinfo  = speigvals(A, E, B, C, D)
@test sort(real(val)) ≈ [-1, 2, Inf, Inf, Inf, Inf] && 
(kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == (Int64[], [2], [2, 2], 2, 8) 

@time val, kinfo  = speigvals(A, E, view(B,:,1), C, view(D,:,1))
@test sort(real(val)) ≈ [-1, 2, Inf, Inf] && 
(kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == (Int64[], [1, 2], [2], 2, 7) 

@test sprank(A, E, B, C, D, fastrank = true) == 8 && sprank(A, E, B, C, D, fastrank = false) == 8

@test sprank(A, E, view(B,:,1), C, view(D,:,1), fastrank = true) == 7 && sprank(A, E, view(B,:,1), C, view(D,:,1), fastrank = false) == 7

# output decoupling zeros
kinfo  = spkstruct(A, E, missing, C, missing, atol1 = 1.e-7, atol2 = 1.e-7)
@test (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == (Int64[], [1, 2, 2], [], 1, 6)

@time val, iz, kinfo  = spzeros(A, E, missing, C, missing)
@test val ≈ [-1] && iz == [] && 
      (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == (Int64[], [1, 2, 2], [], 1, 6)

@time val, kinfo  = speigvals(A, E, missing, C, missing, atol1 = 1.e-7, atol2 = 1.e-7)
@test val ≈ [-1] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == (Int64[], [1, 2, 2], [], 1, 6)

@test sprank(A, E, missing, C, missing, fastrank = true) == 6 && 
      sprank(A, E, missing, C, missing, fastrank = false) == 6

# input decoupling zeros
kinfo  = spkstruct(A, E, B, missing, missing, atol1 = 1.e-7, atol2 = 1.e-7)
@test (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == ([2,3], [], [], 1, 6)

@time val, iz, kinfo  = spzeros(A, E, B, missing, missing)
@test val ≈ [-4] && iz == [] && 
      (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == ([2,3], [], [], 1, 6)

@time val, kinfo  = speigvals(A, E, B, missing, missing)
@test val ≈ [-4] && (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == ([2,3], [], [], 1, 6)
   
@test sprank(A, E, B, missing, missing, fastrank = true) == 6 && 
      sprank(A, E, B, missing, missing, fastrank = false) == 6

@test sprank(A, E, missing, missing, missing, fastrank = true) == 6 && 
      sprank(A, E, missing, missing, missing, fastrank = false) == 6

@test sprank(missing, missing, missing, missing, D, fastrank = true) == 0 && 
      sprank(missing, missing, missing, missing, D, fastrank = false) == 0


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
M2 = [A B; C D]
N2 = [E zeros(9,3); zeros(3,12)]

# poles
kinfo  = spkstruct(A, E, missing, missing, missing, atol1 = 1.e-7, atol2 = 1.e-7)
@test (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == (Int64[], [], [3, 3, 3], 0, 9)

@time val, iz, kinfo  = spzeros(A, E, missing, missing, missing, atol1 = 1.e-7, atol2 = 1.e-7)
@test val ≈ [Inf, Inf, Inf, Inf, Inf, Inf] && iz == [2, 2, 2] && 
      (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == (Int64[], [], [3, 3, 3], 0, 9)

@time val, kinfo  = speigvals(A, E, missing, missing, missing, atol1 = 1.e-7, atol2 = 1.e-7)
@test val ≈ [Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf] && 
      (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == (Int64[], [], [3, 3, 3], 0, 9)

@test sprank(A, E, missing, missing, missing, fastrank = true) == 9 && 
      sprank(A, E, missing, missing, missing, fastrank = false) == 9

@test sprank(missing, missing, missing, missing, missing, fastrank = true) == 0 && 
      sprank(missing, missing, missing, missing, missing, fastrank = false) == 0
      
# output decoupling zeros 

kinfo  = spkstruct(A, E, missing, C, missing, atol1 = 1.e-7, atol2 = 1.e-7)
@test (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == (Int64[], [0, 1, 1], [1, 3, 3], 0, 9)

@time val, iz, kinfo  = spzeros(A, E, missing, C, missing)
@test val ≈ [Inf, Inf, Inf, Inf] && iz == [2, 2] && 
      (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == (Int64[], [0, 1, 1], [1, 3, 3], 0, 9)

@time val, kinfo  = speigvals(A, E, missing, C, missing, atol1 = 1.e-7, atol2 = 1.e-7)
@test val ≈ [Inf, Inf, Inf, Inf, Inf, Inf, Inf] && 
      (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == (Int64[], [0, 1, 1], [1, 3, 3], 0, 9)

@test sprank(A, E, missing, C, missing, fastrank = true) == 9 && 
      sprank(A, E, missing, C, missing, fastrank = false) == 9


# input decoupling zeros
kinfo  = spkstruct(A, E, B, missing, missing, atol1 = 1.e-7, atol2 = 1.e-7)
@test (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == ([2, 2, 2], Int64[], [1, 1, 1], 0, 9)

@time val, iz, kinfo  = spzeros(A, E, B, missing, missing)
@test val ≈ Float64[] && iz == [] && 
      (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == ([2, 2, 2], Int64[], [1, 1, 1], 0, 9)

@time val, kinfo  = speigvals(A, E, B, missing, missing, atol1 = 1.e-7, atol2 = 1.e-7)
@test val ≈ [Inf, Inf, Inf] && 
      (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == ([2, 2, 2], Int64[], [1, 1, 1], 0, 9)
     
@test sprank(A, E, B, missing, missing, fastrank = true) == 9 && 
      sprank(A, E, missing, C, missing, fastrank = false) == 9



# zeros
kinfo  = spkstruct(A, E, B, C, D, atol1 = 1.e-7, atol2 = 1.e-7)
@test (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == ([2], [1], [1, 1, 1, 1, 3], 1, 11)

@time val, iz, kinfo  = spzeros(A, E, B, C, D)
@test val ≈ [1, Inf,Inf] && iz == [2] && 
      (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == ([2], [1], [1, 1, 1, 1, 3], 1, 11)
     
@time val, kinfo  = speigvals(A, E, B, C, D, atol1 = 1.e-7, atol2 = 1.e-7)
@test val ≈ [1, Inf, Inf, Inf, Inf, Inf, Inf, Inf] && 
      (kinfo.rki,kinfo.lki,kinfo.id,kinfo.nf,kinfo.nrank) == ([2], [1], [1, 1, 1, 1, 3], 1, 11)

@test sprank(A, E, B, C, D, fastrank = true) == 11 && 
      sprank(A, E, B, C, D, fastrank = false) == 11

@test sprank(missing, missing, missing, missing, D, fastrank = true) == 2 && 
      sprank(missing, missing, missing, missing, D, fastrank = false) == 2


end  # applications testset

end  # module