# MatrixPencils.jl

#[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3556377.svg)](https://doi.org/10.5281/zenodo.3556867)
[![Travis](https://travis-ci.com/andreasvarga/MatrixPencils.jl.svg?branch=master)](https://travis-ci.com/andreasvarga/MatrixPencils.jl)
[![codecov.io](https://codecov.io/gh/andreasvarga/MatrixPencils.jl/coverage.svg?branch=master)](https://codecov.io/gh/andreasvarga/MatrixPencils.jl?branch=master)
[![Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://andreasvarga.github.io/MatrixPencils.jl/dev/)
[![The MIT License](https://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat-square)](https://github.com/andreasvarga/MatrixPencils.jl/blob/master/LICENSE.md)

**Matrix pencil manipulation using [Julia](http://julialang.org)**

## Compatibility

Julia 1.1 and higher.

## How to Install

````JULIA
pkg> add MatrixPencils
pkg> test MatrixPencils
````
## About

The Kronecker-canonical form of a linear pencil `M − λN` basically characterizes the right and left singular structure and the eigenvalue structure of the pencil. The computation of the Kronecker-canonical form may involve the use of ill-conditioned similarity transformations and, therefore, is potentially numerically unstable. Fortunately, alternative staircase forms, called `Kronecker-like forms` (KLFs), can be determined by employing exclusively orthogonal or unitary similarity transformations and allow to obtain basically the same (or only a part of) structural information on the pencil `M − λN`. Various KLFs can serve to address, in a numerically reliable way, the main applications of the Kronecker form,
such as the computation of minimal left or right nullspace bases, the computation of eigenvalues and zeros, the determination of the normal rank of polynomial and rational matrices, the computation of various factorizations of rational matrices, as well as the solution of linear equations with polynomial or rational matrices. The KLFs are also instrumental for solving computational problems in the analysis of generalized systems described by linear differential- or difference-algebraic equations (also known as descriptor systems).

This collection of Julia functions is an attemp to implement high performance numerical software to compute a range of
KLFs which reveal the full or partial Kronecker structure of a linear pencil. The KLFs are computed by performing several pencil reduction operations on a reduced basic form of the initial pencil. These operations efficiently compress the rows or columns of certain submatrices to full rank matrices and simultaneously maintain the reduced basic form. The rank decisions involve the use of rank revealing QR-decompositions with colum pivoting or, the more reliable, SVD-decompositions. The overall computational complexity of all reduction algorithms is ``O(n^3)``, where ``n`` is the largest dimension of the pencil.

Many of the implemented pencil manipulation algorithms are extensions of computational procedures proposed by Professor Paul Van Dooren (Université catholique de Louvain, Belgium) in several seminal contributions in the field of linear algebra and its applications in control systems theory. The author expresses his gratitude to Paul Van Dooren for his friendly support during the implementation of functions for manipulation of polynomial matrices. Therefore, the release v1.0 of the **MatrixPencils** package is dedicated in his honor on the occasion of his 70th birthday in 2020.

The current version of the package includes the following functions:

**Manipulation of general linear matrix pencils**

* **sreduceBF**  Reduction to the basic condensed form  `[B A-λE; D C]` with `E` upper triangular and nonsingular.
* **klf**   Computation of the Kronecker-like form exhibiting the full Kronecker structure.
* **klf_left**  Computation of the Kronecker-like form exhibiting the left Kronecker structure.
* **klf_right**  Computation of the Kronecker-like form exhibiting the right Kronecker structure.
* **klf_rlsplit**  Computation of the Kronecker-like form exhibiting the separation of right and left Kronecker structures.

**Manipulation of structured linear matrix pencils of the form `[A-λE B; C D]`**

* **sreduceBF** Reduction to the basic condensed form  `[B A-λE; D C]` with `E` upper triangular and nonsingular.
* **sklf**  Computation of the Kronecker-like form exhibiting the full Kronecker structure.
* **sklf_left** Computation of the Kronecker-like form exhibiting the left Kronecker structure.
* **sklf_right**  Computation of the Kronecker-like form exhibiting the right Kronecker structure.

**Manipulation of regular linear matrix pencils**

* **isregular**   Checking the regularity of a pencil.
* **isunimodular** Checking the unimodularity of a pencil.
* **fisplit**  Finite-infinite eigenvalue splitting.

**Some applications of matrix pencil computations**

* **pkstruct** Determination of the complete Kronecker structure.  
* **prank** Determination of the normal rank.
* **peigvals** Computation of the finite and infinite eigenvalues.
* **pzeros** Computation of the finite and infinite zeros.

**Some applications to structured linear matrix pencils of the form `[A-λE B; C D]`**

* **spkstruct**  Determination of the complete Kronecker structure.
* **sprank**  Determination of the normal rank.
* **speigvals**  Computation of the finite and infinite eigenvalues.
* **spzeros**  Computation of the finite and infinite zeros.

**Manipulation of linearizations of the form `[A-λE B; C D]` and `[A-λE B-λF; C-λG D-λH]` of polynomial or rational matrices**

* **lsminreal** Computation of minimal order liniarizations `[A-λE B; C D]` of rational matrices.
* **lsminreal2** Computation of minimal order liniarizations `[A-λE B; C D]` of rational matrices (potentially more efficient).
* **lpsminreal**  Computation of strong minimal pencil based liniarizations `[A-λE B-λF; C-λG D-λH]` of rational matrices.
* **lsequal**  Check the equivalence of two linearizations.
* **lpsequal**  Check the equivalence of two pencil based liniarizations.  

**Manipulation of polynomial matrices** 

* **poly2pm**  Conversion of a polynomial matrix from the **[`Polynomials`](https://github.com/JuliaMath/Polynomials.jl)** package format to a 3D matrix.
* **pm2poly**  Conversion of a polynomial matrix from a 3D matrix to the **[`Polynomials`](https://github.com/JuliaMath/Polynomials.jl)** package format.
* **pmdeg**  Determination of the degree of a polynomial matrix.
* **pmeval**  Evaluation of a polynomial matrix for a given value of its argument.
* **pmreverse**  Building the reversal of a polynomial matrix.  
* **pm2lpCF1**  Building a linearization in the first companion Frobenius form.
* **pm2lpCF2**  Building a linearization in the second companion Frobenius form.  
* **pm2ls**  Building a structured linearization of a polynomial matrix.
* **ls2pm**  Computation of the polynomial matrix from its structured linearization.
* **pm2lps**  Building a linear pencil based structured linearization of a polynomial matrix.
* **lps2pm**  Computation of the polynomial matrix from its linear pencil based structured linearization.
* **spm2ls** Building a structured linearization `[A-λE B; C D]` of a structured polynomial matrix `[T(λ) U(λ); V(λ) W(λ)]`.
* **spm2lps** Building a linear pencil based structured linearization `[A-λE B-λF; C-λG D-λH]` of a structured polynomial matrix `[T(λ) U(λ); V(λ) W(λ)]`.

**Some applications to polynomial matrices** 

* **pmkstruct**  Determination of the complete Kronecker structure.
* **pmeigvals**  Computation of the finite and infinite eigenvalues.
* **pmzeros**  Computation of the finite and infinite zeros.
* **pmzeros1**  Computation of the finite and infinite zeros using linear pencil based structured linearization.
* **pmzeros2**  Computation of the finite and infinite zeros using structured pencil based linearization.
* **pmroots**  Computation of the roots of the determinant of a regular polynomial matrix.
* **pmpoles**  Computation of the infinite poles.
* **pmpoles1**  Computation of the infinite poles using linear pencil based structured linearization.
* **pmpoles2**  Computation of the infinite poles using structured pencil based linearization.
* **pmrank**  Determination of the normal rank.
* **ispmregular**  Checking the regularity of a polynomial matrix.
* **ispmunimodular**  Checking the unimodularity of a polynomial matrix.

A complete list of implemented functions is available [here](https://sites.google.com/site/andreasvargacontact/home/software/matrix-pencils-in-julia).

## Future plans

The collection of tools will be extended by adding new functionality, such as new tools for the manipulation of regular pencils (e.g., reduction to a block-diagonal structure, eigenvalue assignment), building linearizations of polynomial matrices in other bases (e.g., orthogonal polynomial bases), applications of structured linear pencils manipulations to rational matrix problems, etc.
