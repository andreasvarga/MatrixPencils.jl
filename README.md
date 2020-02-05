# MatrixPencils.jl

[![Travis](https://travis-ci.com/andreasvarga/MatrixPencils.jl.svg?branch=master)](https://travis-ci.com/andreasvarga/MatrixEquations.jl)
[![codecov.io](https://codecov.io/gh/andreasvarga/MatrixPencils.jl/coverage.svg?branch=master)](https://codecov.io/gh/andreasvarga/MatrixEquations.jl?branch=master)
[![Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://andreasvarga.github.io/MatrixPencils.jl/dev/)
[![The MIT License](https://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat-square)](https://github.com/andreasvarga/MatrixPencils.jl/blob/master/LICENSE.md)

**Matrix pencil manipulation using [Julia](http://julialang.org)**

## About

This collection of Julia functions is an attemp to implement high performance
numerical software to efficiently perform manipulations of matrix pencils. The main goal is to provide a comprehensive set
of computational tools which allow the solution of various structural problems arising in the study of linear systems described by linear differential- or difference-algebraic equations (also know as descriptor systems). The provided tools
will be also instrumental for the analysis of generalized systems described by rational or polynomial matrices.

The available functions in the current version of the `MatrixPencils.jl` package cover several basic reductions of linear
matrix pencils `M - λN` to block upper-triangular forms (also known as Kronecker-like forms) using orthogonal similarity
transformations. The resulting condensed forms reveal a part or the full Kronecker structure of the pencil `M - λN` and allow to address several applications as the determination of the rank, finite and infinite eigenvalues or zeros (also known as roots) of `M - λN`. The implementation of basic functions rely on a set of flexible pencil manipulation tools, which allow to perform elementary row and column compresions which preserve the underlying basic form of the reduced pencil.

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

The collection of tools will be extended by adding new functionality, such as tools for the manipulation of regular pencils (e.g., reduction to forms with finite-infinite separated eigenvalues or to a block-diagonal structure), manipulation of structured linear pencils with application to polynomial and rational matrices, etc.
