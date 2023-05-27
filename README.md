# MatrixPencils.jl
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3894504.svg)](https://doi.org/10.5281/zenodo.3894503)
[![DocBuild](https://github.com/andreasvarga/MatrixPencils.jl/workflows/CI/badge.svg)](https://github.com/andreasvarga/MatrixPencils.jl/actions)
[![codecov.io](https://codecov.io/gh/andreasvarga/MatrixPencils.jl/coverage.svg?branch=master)](https://codecov.io/gh/andreasvarga/MatrixPencils.jl?branch=master)
[![Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://andreasvarga.github.io/MatrixPencils.jl/dev/)
[![The MIT License](https://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat-square)](https://github.com/andreasvarga/MatrixPencils.jl/blob/master/LICENSE.md)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

**Matrix pencil manipulation using [Julia](http://julialang.org)**

## Compatibility

Julia 1.6 and higher.

## How to Install

````JULIA
pkg> add MatrixPencils
pkg> test MatrixPencils
````

## About

The Kronecker-canonical form of a linear matrix pencil `M − λN` basically characterizes the right and left singular structure and the eigenvalue structure of the pencil. The computation of the Kronecker-canonical form may involve the use of ill-conditioned similarity transformations and, therefore, is potentially numerically unstable. Fortunately, alternative staircase forms, called *Kronecker-like forms* (KLFs), can be determined by employing exclusively orthogonal or unitary similarity transformations and allow to obtain basically the same (or only a part of) structural information on the pencil `M − λN`. Various KLFs can serve to address, in a numerically reliable way, the main applications of the Kronecker form,
such as the computation of minimal left or right nullspace bases, the computation of eigenvalues and zeros, the determination of the normal rank of polynomial and rational matrices, the computation of various factorizations of rational matrices, as well as the solution of linear equations with polynomial or rational matrices. The KLFs are also instrumental for solving computational problems in the analysis of generalized systems described by linear differential- or difference-algebraic equations (also known as descriptor systems).

This collection of Julia functions is an attemp to implement high performance numerical software to compute a range of
KLFs which reveal the full or partial Kronecker structure of a linear pencil. The KLFs are computed by performing several pencil reduction operations on a reduced basic form of the initial pencil. These operations efficiently compress the rows or columns of certain submatrices to full rank matrices and simultaneously maintain the reduced basic form. The rank decisions involve the use of rank revealing QR-decompositions with column pivoting or, the more reliable, SVD-decompositions. The overall computational complexity of all reduction algorithms is ``O(n^3)``, where ``n`` is the largest dimension of the pencil.

Many of the implemented pencil manipulation algorithms are extensions of computational procedures proposed by Professor Paul Van Dooren (Université catholique de Louvain, Belgium) in several seminal contributions in the field of linear algebra and its applications in control systems theory. The author expresses his gratitude to Paul Van Dooren for his friendly support during the implementation of functions for manipulation of polynomial matrices. Therefore, the release v1.0 of the **MatrixPencils** package is dedicated in his honor on the occasion of his 70th birthday in 2020.

The current version of the package includes the following functions:

**Manipulation of general linear matrix pencils**

* **pbalance!** Balancing linear matrix pencils.  
* **pbalqual**  Balancing quality of a matrix pencils.  
* **preduceBF**  Reduction to the basic condensed form  `[B A-λE; D C]` with `E` upper triangular and nonsingular.
* **klf**   Computation of the Kronecker-like form exhibiting the full Kronecker structure.
* **klf_left**  Computation of the Kronecker-like form exhibiting the left and finite Kronecker structures.
* **klf_leftinf**  Computation of the Kronecker-like form exhibiting the left and infinite Kronecker structures.
* **klf_right**  Computation of the Kronecker-like form exhibiting the right Kronecker structure.
* **klf_rlsplit**  Computation of the Kronecker-like form exhibiting the separation of right and left Kronecker structures.

**Manipulation of structured linear matrix pencils of the form `[A-λE B; C D]`**

* **sreduceBF** Reduction to the basic condensed form  `[B A-λE; D C]` with `E` upper triangular and nonsingular.
* **sklf**  Computation of the Kronecker-like form exhibiting the full Kronecker structure.
* **sklf_left** Computation of the Kronecker-like form exhibiting the left Kronecker structure.
* **sklf_right**  Computation of the Kronecker-like form exhibiting the right Kronecker structure.
* **gsklf**  Computation of several row partition preserving special Kronecker-like forms.

**Manipulation of regular linear matrix pencils**

* **regbalance!** Balancing regular matrix pencils.  
* **isregular**   Checking the regularity of a pencil.
* **isunimodular** Checking the unimodularity of a pencil.
* **fisplit**  Finite-infinite eigenvalue splitting.
* **sfisplit**  Special finite-infinite eigenvalue splitting.
* **fihess**  Finite-infinite eigenvalue splitting in a generalized Hessenberg form.
* **fischur**  Finite-infinite eigenvalue splitting in a generalized Schur form.
* **fischursep**  Finite-infinite eigenvalue splitting in an ordered generalized Schur form.
* **sfischursep**  Special finite-infinite eigenvalue splitting in an ordered generalized Schur form.
* **fiblkdiag**  Finite-infinite eigenvalue splitting based block diagonalization.
* **gsblkdiag**  Finite-infinite and stable-unstable eigenvalue splitting based block diagonalization.
* **saloc**  Spectrum alocation for the pairs `(A,B)` and `(A-λE,B)`.
* **salocd**  Spectrum alocation for the dual pairs `(A,C)` and `(A-λE,C)`.
* **ordeigvals**  Order-preserving computation of eigenvalues of a Schur matrix or a generalized Schur pair.

**Some applications of matrix pencil computations**

* **pkstruct** Determination of the Kronecker structure information.  
* **peigvals** Computation of the finite and infinite eigenvalues.
* **pzeros** Computation of the finite and infinite zeros.
* **prank** Determination of the normal rank.

**Some applications to structured linear matrix pencils of the form `[A-λE B; C D]`**

* **spkstruct**  Determination of the Kronecker structure information.
* **speigvals**  Computation of the finite and infinite eigenvalues.
* **spzeros**  Computation of the finite and infinite zeros.
* **sprank**  Determination of the normal rank.

**Manipulation of linearizations of polynomial or rational matrices**

* **lsbalance!** Scaling of a descriptor system based linearization. 
* **lsminreal** Computation of minimal order linearizations of the form `[A-λE B; C D]`.
* **lsminreal2** Computation of minimal order linearizations of the form `[A-λE B; C D]` (potentially more efficient).
* **lpsminreal**  Computation of strong minimal pencil based linearizations of the form `[A-λE B-λF; C-λG D-λH]`.
* **lsequal**  Check the equivalence of two linearizations.
* **lpsequal**  Check the equivalence of two pencil based linearizations.  
* **lseval**   Evaluation of the value of the rational matrix corresponding to a descriptor system based linearization.
* **lpseval**  Evaluation of the value of the rational matrix corresponding to a pencil based linearization.
* **lps2ls**  Conversion of a pencil based linearization into a descriptor system based linearization.
* **lsbalqual** Evaluation of the scaling quality of descriptor system based linearizations.

**Manipulation of polynomial matrices** 

* **poly2pm**  Conversion of a polynomial matrix from the **[`Polynomials`](https://github.com/JuliaMath/Polynomials.jl)** package format to a 3D matrix.
* **pm2poly**  Conversion of a polynomial matrix from a 3D matrix to the **[`Polynomials`](https://github.com/JuliaMath/Polynomials.jl)** package format.
* **pmdeg**  Determination of the degree of a polynomial matrix.
* **pmeval**  Evaluation of a polynomial matrix for a given value of its argument.
* **pmreverse**  Building the reversal of a polynomial matrix.  
* **pmdivrem**  Quotients and remainders of elementwise divisions of two polynomial matrices.  
* **pm2lpCF1**  Building a linearization in the first companion Frobenius form.
* **pm2lpCF2**  Building a linearization in the second companion Frobenius form.  
* **pm2ls**  Building a structured linearization of a polynomial matrix.
* **ls2pm**  Computation of the polynomial matrix from its structured linearization.
* **pm2lps**  Building a linear pencil based structured linearization of a polynomial matrix.
* **lps2pm**  Computation of the polynomial matrix from its linear pencil based structured linearization.
* **spm2ls** Building a structured linearization `[A-λE B; C D]` of a structured polynomial matrix `[T(λ) U(λ); V(λ) W(λ)]`.
* **spm2lps** Building a linear pencil based structured linearization `[A-λE B-λF; C-λG D-λH]` of a structured polynomial matrix `[T(λ) U(λ); V(λ) W(λ)]`.

**Some applications to polynomial matrices** 

* **pmkstruct**  Determination of the Kronecker structure and infinite pole-zero structure.
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

**Manipulation of rational matrices** 

* **rm2lspm** Representation of a rational matrix as a linearization of its strictly proper part plus its polynomial part.
* **rmeval** Evaluation of a rational matrix for a given value of its argument.
* **rm2ls** Building a descriptor system based structured linearization of a rational matrix.
* **ls2rm** Computation of the rational matrix from its descriptor system based structured linearization.
* **rm2lps** Building a pencil based structured linearization of a rational matrix.
* **lps2rm** Computation of the rational matrix from its pencil based structured linearization.
* **lpmfd2ls** Building a descriptor system based structured linearization of a left polynomial matrix fractional description.  
* **rpmfd2ls** Building a descriptor system based structured linearization of a right polynomial matrix fractional description.
* **lpmfd2lps** Building a pencil based structured linearization of a left polynomial matrix fractional description.
* **rpmfd2lps** Building a pencil based structured linearization of a right polynomial matrix fractional description.
* **pminv2ls** Building a descriptor system based structured linearization of the inverse of a polynomial matrix.
* **pminv2lps** Building a pencil based structured linearization of the inverse of a polynomial matrix.

**Some applications to rational matrices**

* **rmkstruct**  Determination of the Kronecker structure and infinite pole-zero structure.
* **rmzeros**  Computation of the finite and infinite zeros using structured pencil based linearization.
* **rmzeros1**  Computation of the finite and infinite zeros using linear pencil based structured linearization.
* **rmpoles**  Computation of the finite and infinite poles using structured pencil based linearization.
* **rmpoles1**  Computation of the finite and infinite poles using linear pencil based structured linearization.
* **rmrank**  Determination of the normal rank.

A complete list of implemented functions is available
[here](https://sites.google.com/view/andreasvarga/home/software/matrix-pencils-in-julia).

## Future plans

Functional extensions and performance enhancements of some functions will be performed as needs arise.

## Supplementary information

The mathematical background and the computational aspects which underly the implementation of functions for polynomial and rational matrices are presented in the eprint [arXiv:2006.06825](https://arxiv.org/pdf/2006.06825).
