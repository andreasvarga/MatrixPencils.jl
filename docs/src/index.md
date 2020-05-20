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
such as the computation of minimal left or right nullspace bases, the computation of eigenvalues and zeros, the determination of the normal rank of polynomial and rational matrices, computation of various factorizations of rational matrices, as well as the solution of linear equations with polynomial or rational matrices.

This collection of Julia functions is an attemp to implement high performance numerical software to compute a range of
KLFs which reveal the full or partial Kronecker structure of a linear pencil. The KLFs are computed by performing several pencil reduction operations on a reduced basic form of the initial pencil. These operations efficiently compress the rows or columns of certain submatrices to full rank matrices and simultaneously maintain the reduced basic form. The rank decisions involve the use of rank revealing QR-decompositions with colum pivoting or the, more reliable, SVD-decompositions. The overall computational complexity of all reduction algorithms is ``O(n^3)``, where ``n`` is the largest dimension of the pencil.  

The implemented basic pencil reduction operations are described in [1] and [2], and form the basis of the implemented **PREDUCE** procedure described in [3].

A set of functions is provided to address pencil manipulation problems for structured linear pencils `M − λN` of the forms
`[A - λE B; C D]` or `[A-λE B-λF; C-λG D-λH]`. Linear pencils with these structure frequently arise from the linearization of polynomial or rational matrices. The computation of least order linearizations is based on algorithms described in [3]-[9].  

A set of functions is available for the manipulation of polynomial matrices specified via their coefficient matrices in a monomial basis. All funtions also support matrix, vector or scalar of elements of the `Polynomial` type
provided by the [Polynomials](https://github.com/JuliaMath/Polynomials.jl) package. Several linearization functions are available which allow the extension of pencil manipulation techniques to matrix polynomials. Some straightforward applications are covered such as the computation of finite and infinite eigenvalues, zeros and poles, the determination of the normal rank, the determination of Kronecker indices and finite and infinite eigenvalue structure, checks of regularity and unimodularity. The implementations follow the computational framework and results developed in [10].

The available functions in the `MatrixPencils.jl` package cover both real and complex numerical data.
The current version of the package includes the following functions:

**Manipulation of general linear matrix pencils**

| Function | Description |
| :--- | :--- |
| **[`preduceBF`](@ref)** | Reduction to the basic condensed form  `[B A-λE; D C]` with `E` upper triangular and nonsingular. |
| **[`klf`](@ref)** |   Computation of the Kronecker-like form exhibiting the full Kronecker structure |
| **[`klf_left`](@ref)** |  Computation of the Kronecker-like form exhibiting the left Kronecker structure |
| **[`klf_right`](@ref)** |   Computation of the Kronecker-like form exhibiting the right Kronecker structure |
| **[`klf_rlsplit`](@ref)** | Computation of the Kronecker-like form exhibiting the separation of right and left Kronecker structures |

**Manipulation of structured linear matrix pencils of the form [A-λE B; C D]**

| Function | Description |
| :--- | :--- |
| **[`sreduceBF`](@ref)** | Reduction to the basic condensed form  `[B A-λE; D C]` with `E` upper triangular and nonsingular. |
| **[`sklf`](@ref)** |   Computation of the Kronecker-like form exhibiting the full Kronecker structure |
| **[`sklf_left`](@ref)** |  Computation of the Kronecker-like form exhibiting the left Kronecker structure |
| **[`sklf_right`](@ref)** |   Computation of the Kronecker-like form exhibiting the right Kronecker structure |

**Manipulation of regular linear matrix pencils**

| Function | Description |
| :--- | :--- |
| **[`isregular`](@ref)** | Checking the regularity of a pencil |
| **[`isunimodular`](@ref)** | Checking the unimodularity of a pencil |  
| **[`fisplit`](@ref)** | Finite-infinite eigenvalue splitting |

**Some applications of matrix pencil computations**

| Function | Description |
| :--- | :--- |
| **[`pkstruct`](@ref)** | Determination of the complete Kronecker structure |
| **[`prank`](@ref)** | Determination of the normal rank |
| **[`peigvals`](@ref)** | Computation of the finite and infinite eigenvalues |
| **[`pzeros`](@ref)** | Computation of the finite and infinite zeros |

**Some applications to structured linear matrix pencils of the form `[A-λE B; C D]`**

| Function | Description |
| :--- | :--- |
| **[`spkstruct`](@ref)** | Determination of the complete Kronecker structure |
| **[`sprank`](@ref)** | Determination of the normal rank |
| **[`speigvals`](@ref)** | Computation of the finite and infinite eigenvalues |
| **[`spzeros`](@ref)** | Computation of the finite and infinite zeros |

**Manipulation of linearizations of the form `[A-λE B; C D]` and `[A-λE B-λF; C-λG D-λH]` of polynomial or rational matrices**

| Function | Description |
| :--- | :--- |
| **[`lsminreal`](@ref)** | Computation of minimal order linearizations `[A-λE B; C D]` of rational matrices |
| **[`lsminreal2`](@ref)** | Computation of minimal order linearizations `[A-λE B; C D]` of rational matrices (potentially more efficient)|
| **[`lpsminreal`](@ref)** | Computation of strong minimal pencil based linearizations `[A-λE B-λF; C-λG D-λH]` of rational matrices |
| **[`lsequal`](@ref)** | Check the equivalence of two linearizations |
| **[`lpsequal`](@ref)** | Check the equivalence of two pencil based linearizations |

**Manipulation of polynomial matrices** 

| Function | Description |
| :--- | :--- |
| **[`poly2pm`](@ref)** | Conversion of a polynomial matrix used in **Polynomials** package to a polynomial matrix represented as a 3-dimensional matrix |
| **[`pm2poly`](@ref)** | Conversion of a polynomial matrix represented as a 3-dimensional matrix to a polynomial matrix used in **Polynomials** package |
| **[`pmdeg`](@ref)** | Determination of the degree of a polynomial matrix |
| **[`pmeval`](@ref)** | Evaluation of a polynomial matrix for a given value of its argument.|
| **[`pmreverse`](@ref)** | Building the reversal of a polynomial matrix  |
| **[`pm2lpCF1`](@ref)** | Building a linearization in the first companion Frobenius form  |
| **[`pm2lpCF2`](@ref)** | Building a linearization in the second companion Frobenius form  |
| **[`pm2ls`](@ref)** | Building a structured linearization `[A-λE B; C D]` of a polynomial matrix |
| **[`ls2pm`](@ref)** | Computation of the polynomial matrix from its structured linearization |
| **[`pm2lps`](@ref)** | Building a linear pencil based structured linearization `[A-λE B-λF; C-λG D-λH]` of a polynomial matrix |
| **[`lps2pm`](@ref)** | Computation of the polynomial matrix from its linear pencil based structured linearization |
| **[`spm2ls`](@ref)** | Building a structured linearization `[A-λE B; C D]` of a structured polynomial matrix `[T(λ) U(λ); V(λ) W(λ)]`|
| **[`spm2lps`](@ref)** | Building a linear pencil based structured linearization `[A-λE B-λF; C-λG D-λH]` of a structured polynomial matrix `[T(λ) U(λ); V(λ) W(λ)]`|

**Some applications to polynomial matrices** 

| Function | Description |
| :--- | :--- |
| **[`pmkstruct`](@ref)** | Determination of the complete Kronecker structure using companion form based linearizations  |
| **[`pmeigvals`](@ref)** | Computation of the finite and infinite eigenvalues using companion form based linearizations |
| **[`pmzeros`](@ref)** | Computation of the finite and infinite zeros using companion form based linearizations |
| **[`pmzeros1`](@ref)** | Computation of the finite and infinite zeros using linear pencil based structured linearization |
| **[`pmzeros2`](@ref)** | Computation of the finite and infinite zeros using structured pencil based linearization |
| **[`pmroots`](@ref)** | Computation of the roots of the determinant of a regular polynomial matrix |
| **[`pmpoles`](@ref)** | Computation of the (infinite) poles using companion form based linearizations |
| **[`pmpoles1`](@ref)** | Computation of the (infinite) poles using linear pencil based structured linearization |
| **[`pmpoles2`](@ref)** | Computation of the (infinite) poles using structured pencil linearization |
| **[`pmrank`](@ref)** | Determination of the normal rank |
| **[`ispmregular`](@ref)** | Checking the regularity of a polynomial matrix |
| **[`ispmunimodular`](@ref)** | Checking the unimodularity of a polynomial matrix |

A complete list of implemented functions is available [here](https://sites.google.com/site/andreasvargacontact/home/software/matrix-pencils-in-julia).

## Future plans

The collection of tools will be extended by adding new functionality, such as tools for the manipulation of regular pencils (e.g., reduction to a block-diagonal structure, eigenvalue assignment), building linearizations of polynomial matrices in other bases (e.g., orthogonal polynomial bases) or in sparse polynomial representations, applications of structured linear pencils manipulations to rational matrix problems, etc.

## Release Notes

### Version 1.0.0

This release includes implementations of computational procedures to manipulate polynomial matrices specified via their coefficient matrices in a monomial basis. All funtions also support matrix, vector or scalar of elements of the `Polynomial` type
provided by the [Polynomials](https://github.com/JuliaMath/Polynomials.jl) package.  Several linearization functions are available which allow the extension of pencil manipulation techniques to matrix polynomials. Some straightforward applications are covered such as the computation of finite and infinite eigenvalues, zeros and poles, the determination of the normal rank, the determination of Kronecker indices and finite and infinite eigenvalue structure, checks of regularity and unimodularity.  A new function is provided to check the unimodularity of a linear pencil.

### Version 0.5.0

This release includes implementations of computational procedures to determine least order linear pencil based structured linerizations, as those which may arise from the linearization of polynomial and rational matrices. A new function to check the equivalence of two linear pencil based linearizations is also provided. An enhanced modularization of functions for basic pencil and structured pencil reduction has been performed to eliminate possible modifications of input arguments.

### Version 0.4.0

This release includes implementations of computational procedures to determine least order linerizations, as those which may arise from the linearization of polynomial and rational matrices. The elimination of simple eigenvalues is based on a new function to compute SVD-like forms of regular matrix pencils. A new function to check the equivalence of two linearizations is also provided.

### Version 0.3.0

This release includes implementations of several reductions of structured linear pencils to various KLFs and covers some straightforward applications such as the computation of finite and infinite eigenvalues and zeros, the determination of the normal rank, the determination of Kronecker indices and infinite eigenvalue structure.

### Version 0.2.0

This release includes a new tool for finite-infinite eigenvalue splitting of a regular pencil and substantial improvements of the modularization and interfaces of implemented basic pencil operations.

### Version 0.1.0

This is the initial release providing prototype implementations of several pencil reduction algorithms to various KLFs and covering some straightforward applications such as the computation of finite and infinite eigenvalues and zeros, the determination of the normal rank, Kronecker indices and infinite eigenvalue structure.

## Main developer

[Andreas Varga](https://sites.google.com/site/andreasvargacontact/home)

License: MIT (expat)

## References

[1]   C. Oara and P. Van Dooren. An improved algorithm for the computation of structural invariants of a system pencil and related geometric aspects. Syst. Control Lett., 30:39–48, 1997.

[2]   A. Varga. Computation of irreducible generalized state-space realizations. Kybernetika, 26:89–106, 1990.

[3]   A. Varga, Solving Fault Diagnosis Problems – Linear Synthesis Techniques, Vol. 84 of
Studies in Systems, Decision and Control, Springer International Publishing, 2017.

[4] P. Van Dooreen, The generalized eigenstructure problem in linear system theory,
IEEE Transactions on Automatic Control, vol. AC-26, pp. 111-129, 1981.

[5] P. Van Dooren and P. Dewilde, The eigenstructure of an arbitrary polynomial matrix: computational
aspects, Linear Algebra Appl., 50 (1983), pp. 545–579.

[6] F.M. Dopico, M.C. Quintana and P. Van Dooren, Linear system matrices of rational transfer functions, to appear in "Realization and Model Reduction of Dynamical Systems", A Festschrift to honor the 70th birthday of Thanos Antoulas", Springer-Verlag. [arXiv: 1903.05016](https://arxiv.org/pdf/1903.05016.pdf)

[7] G. Verghese, B. Lévy, and T. Kailath, A generalized state-space for singular systems, IEEE Trans.
Automat. Control, 26 (1981), pp. 811–831.

[8] G. Verghese, P. Van Dooren, and T. Kailath, Properties of the system matrix of a generalized state-
space system, Int. J. Control, 30 (1979), pp. 235–243.

[9] G. Verghese, Comments on ‘Properties of the system matrix of a generalized state-space system’,
Int. J. Control, Vol.31(5) (1980) 1007–1009.

[10] F. De Terán, F. M. Dopico, D. S. Mackey, Spectral equivalence of polynomial matrices and
the Index Sum Theorem, Linear Algebra and Its Applications, vol. 459, pp. 264-333, 2014.
