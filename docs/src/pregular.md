# Computations involving regular matrix pencils
| Function | Description |
| :--- | :--- |
| **[`regbalance!`](@ref)** | Balancing regular matrix pencils.  |
| **[`isregular`](@ref)** | Checking the regularity of a pencil |
| **[`isunimodular`](@ref)** | Checking the unimodularity of a pencil |  
| **[`fisplit`](@ref)** | Finite-infinite eigenvalue splitting |
| **[`sfisplit`](@ref)** | Special finite-infinite eigenvalue splitting |
| **[`fihess`](@ref)** | Finite-infinite eigenvalue splitting in a generalized Hessenberg form|
| **[`fischur`](@ref)** | Finite-infinite eigenvalue splitting in a generalized Schur form|
| **[`fischursep`](@ref)** | Finite-infinite eigenvalue splitting in an ordered generalized Schur form|
| **[`sfischursep`](@ref)** | Special finite-infinite eigenvalue splitting in an ordered generalized Schur form|
| **[`fiblkdiag`](@ref)** | Finite-infinite eigenvalue splitting based block diagonalization |
| **[`gsblkdiag`](@ref)** | Finite-infinite and stable-unstable eigenvalue splitting based block diagonalization |
| **[`ssblkdiag`](@ref)** | Stable-unstable eigenvalue splitting based block diagonalization |
| **[`saloc`](@ref)** | Spectrum alocation for the pairs `(A,B)` and `(A-λE,B)` |
| **[`salocd`](@ref)** | Spectrum alocation for the dual pairs `(A,C)` and `(A-λE,C)`  |
| **[`salocinf`](@ref)** | Infinite spectrum alocation for the pair `(A-λE,B)` |
| **[`salocinfd`](@ref)** | Infinite spectrum alocation for the dual pair `(A-λE,C)` |
| **[`ordeigvals`](@ref)** | Order-preserving computation of eigenvalues of a Schur matrix or a generalized Schur pair.   |

```@docs
regbalance!
ordeigvals
isregular
isunimodular
fisplit
MatrixPencils.fisplit!
sfisplit
MatrixPencils.sfisplit!
fihess
fischur
fischursep
sfischursep
fiblkdiag
gsblkdiag
ssblkdiag
saloc
salocd
salocinf
salocinfd
MatrixPencils._qrE!
MatrixPencils._svdlikeAE!
```
