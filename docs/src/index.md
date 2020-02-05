```@meta
CurrentModule = MatrixPencils
DocTestSetup = quote
    using MatrixPencils
end
```

# MatrixPencils.jl

[![Build Status](https://travis-ci.com/andreasvarga/MatrixPencils.jl.svg?branch=master)](https://travis-ci.com/andreasvarga/MatrixPencils.jl)
[![Code on Github.](https://img.shields.io/badge/code%20on-github-blue.svg)](https://github.com/andreasvarga/MatrixPencils.jl)

The Kronecker-canonical form of a linear pencil `M − λN` basically characterizes the right and left singular structure and the eigenvalue structure of the pencil. The computation of the Kronecker-canonical form may involve the use of ill-conditioned similarity transformations and, therefore, is potentially numerically unstable. Fortunately, alternative staircase forms, called `Kronecker-like forms` (KLFs), allow to obtain basically the same (or only a
part of) structural information on the pencil `M − λN` by employing exclusively orthogonal similarity transformations.
Various KLFs can serve to address, in a numerically reliable way, the main applications of the Kronecker form,
such as the computation of minimal left or right nullspace bases, the computation of eigenvalues and (Smith) zeros, the determination of the normal rank of polynomial and rational matrices, computation of various factorizations of rational matrices, as well as the solution of linear equations with polynomial or rational matrices.

This collection of Julia functions is an attemp to implement high performance numerical software to compute a range of
KLFs which reveal the full or partial Kronecker structure of a linear pencil. The KLFs are computed by performing several pencil reduction operations on a reduced basic form of the initial pencil. These operations efficiently compress the rows or columns of certain submatrices to full rank matrices and simultaneously maintain the reduced basic form. The rank decisions involve the use of rank revealing QR-decompositions with colum pivoting or the, more reliable, SVD-decompositions. The overall computational complexity of all reduction algorithms is ``O(n^3)``, where ``n`` is the largest dimension of the pencil.  

The implemented basic pencil reduction operations are described in [1] and [2], and form the basis of the implemented **PREDUCE** procedure described in [3].  

The available functions in the `MatrixPencils.jl` package cover both real and complex numerical data.
The current version of the package includes the following functions:

**Manipulation of general linear matrix pencils**

| Function | Description |
| :--- | :--- |
| **klf** |   Computation of the Kronecker-like form exhibiting the full Kronecker structure |
| **klf_left** |  Computation of the Kronecker-like form exhibiting the left Kronecker structure |
| **klf_right** |   Computation of the Kronecker-like form exhibiting the right Kronecker structure |
| **klf_rlsplit** | Computation of the Kronecker-like form exhibiting the separation of right and left Kronecker structures |

**Manipulation of regular linear matrix pencils**

| Function | Description |
| :--- | :--- |
| **isregular** | Checking the regularity of a pencil |

**Some applications of matrix pencil computations**

| Function | Description |
| :--- | :--- |
| **pkstruct** | Determination of the complete Kronecker structure |
| **prank** | Determination of the normal rank |
| **peigvals** | Computation of the finite and infinite eigenvalues |
| **pzeros** | Computation of the finite and infinite (Smith) zeros |

A complete list of implemented functions is available [here](https://sites.google.com/site/andreasvargacontact/home/software/matrix-pencils-in-julia).

## Future plans

The collection of tools will be extended by adding new functionality, such as tools for the manipulation of regular pencils (e.g., reduction to forms with finite-infinite separated eigenvalues or to a block-diagonal structure), manipulation of structured linear pencils with application to polynomial and rational matrices, etc.

## Release Notes

### Version 0.1.0

This is the initial release covering prototype implementations of several pencil reduction algorithms to various KLFs and some straightforward applications such as the computation of finite and infinite eigenvalues or, the determination of the normal rank, Kronecker indices and infinite eigenvalue structure.

## Main developer

[Andreas Varga](https://sites.google.com/site/andreasvargacontact/home)

License: MIT (expat)

## References

[1]   C. Oara and P. Van Dooren. An improved algorithm for the computation of structural invariants of a system pencil and related geometric aspects. Syst. Control Lett., 30:39–48, 1997.

[2]   A. Varga. Computation of irreducible generalized state-space realizations. Kybernetika, 26:89–106, 1990.

[3]   A. Varga, Solving Fault Diagnosis Problems – Linear Synthesis Techniques, Vol. 84 of
Studies in Systems, Decision and Control, Springer International Publishing, 2017.