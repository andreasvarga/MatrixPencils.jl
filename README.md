# MatrixPencils.jl

[![Travis](https://travis-ci.com/andreasvarga/MatrixPencils.jl.svg?branch=master)](https://travis-ci.com/andreasvarga/MatrixEquations.jl)
[![codecov.io](https://codecov.io/gh/andreasvarga/MatrixPencils.jl/coverage.svg?branch=master)](https://codecov.io/gh/andreasvarga/MatrixEquations.jl?branch=master)
[![Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://andreasvarga.github.io/MatrixPencils.jl/dev/)
[![The MIT License](https://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat-square)](https://github.com/andreasvarga/MatrixPencils.jl/blob/master/LICENSE.md)

**Matrix pencil manipulation using [Julia](http://julialang.org)**

## About

The Kronecker-canonical form of a linear pencil `M − λN` basically characterizes the right and left singular structure and the eigenvalue structure of the pencil. The computation of the Kronecker-canonical form may involve the use of ill-conditioned similarity transformations and, therefore, is potentially numerically unstable. Fortunately, alternative staircase forms, called `Kronecker-like forms` (KLFs), can be determined by employing exclusively orthogonal or unitary similarity transformations and allow to obtain basically the same (or only a part of) structural information on the pencil `M − λN`. Various KLFs can serve to address, in a numerically reliable way, the main applications of the Kronecker form,
such as the computation of minimal left or right nullspace bases, the computation of eigenvalues and (Smith) zeros, the determination of the normal rank of polynomial and rational matrices, the computation of various factorizations of rational matrices, as well as the solution of linear equations with polynomial or rational matrices. The KLFs are also instrumental for solving computational problems in the analysis of generalized systems described by linear differential- or difference-algebraic equations (also known as descriptor systems).

This collection of Julia functions is an attemp to implement high performance numerical software to compute a range of
KLFs which reveal the full or partial Kronecker structure of a linear pencil. The KLFs are computed by performing several pencil reduction operations on a reduced basic form of the initial pencil. These operations efficiently compress the rows or columns of certain submatrices to full rank matrices and simultaneously maintain the reduced basic form. The rank decisions involve the use of rank revealing QR-decompositions with colum pivoting or, the more reliable, SVD-decompositions. The overall computational complexity of all reduction algorithms is ``O(n^3)``, where ``n`` is the largest dimension of the pencil.

The current version of the package includes the following functions:

**Manipulation of general linear matrix pencils**

* **klf**   Computation of the Kronecker-like form exhibiting the full Kronecker structure.
* **klf_left**  Computation of the Kronecker-like form exhibiting the left Kronecker structure.
* **klf_right**  Computation of the Kronecker-like form exhibiting the right Kronecker structure.
* **klf_rlsplit**  Computation of the Kronecker-like form exhibiting the separation of right and left Kronecker structures.

**Manipulation of regular linear matrix pencils**

* **isregular**   Checking the regularity of a pencil.

**Some applications of matrix pencil computations**

* **pkstruct** Determination of the complete Kronecker structure.  
* **prank** Determination of the normal rank.
* **peigvals** Computation of the finite and infinite eigenvalues.
* **pzeros** Computation of the finite and infinite (Smith) zeros.

## Future plans

The collection of tools will be extended by adding new functionality, such as tools for the manipulation of regular pencils (e.g., reduction to forms with separated finite and infinite eigenvalues or to a block-diagonal structure), manipulation of structured linear pencils with application to polynomial and rational matrices, etc.