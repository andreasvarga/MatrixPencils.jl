# Manipulation of polynomial matrices
| Function | Description |
| :--- | :--- |
| **[`poly2pm`](@ref)** | Conversion of a polynomial matrix used in **Polynomials** package to a polynomial matrix represented as a 3-dimensional matrix |
| **[`pm2poly`](@ref)** | Conversion of a polynomial matrix represented as a 3-dimensional matrix to a polynomial matrix used in **Polynomials** package |
| **[`pmdeg`](@ref)** | Determination of the degree of a polynomial matrix |
| **[`pmeval`](@ref)** | Evaluation of a polynomial matrix for a given value of its argument.|
| **[`pmreverse`](@ref)** | Building the reversal of a polynomial matrix  |
| **[`pmdivrem`](@ref)** | Quotients and remainders of elementwise divisions of two polynomial matrices  |
| **[`pm2lpCF1`](@ref)** | Building a linearization in the first companion Frobenius form  |
| **[`pm2lpCF2`](@ref)** | Building a linearization in the second companion Frobenius form  |
| **[`pm2ls`](@ref)** | Building a descriptor system based structured linearization `[A-λE B; C D]` of a polynomial matrix |
| **[`ls2pm`](@ref)** | Computation of the polynomial matrix from its descriptor system based structured linearization |
| **[`pm2lps`](@ref)** | Building a pencil based structured linearization `[A-λE B-λF; C-λG D-λH]` of a polynomial matrix |
| **[`lps2pm`](@ref)** | Computation of the polynomial matrix from its pencil based structured linearization |
| **[`spm2ls`](@ref)** | Building a descriptor system based structured linearization `[A-λE B; C D]` of a structured polynomial matrix `[T(λ) U(λ); V(λ) W(λ)]`|
| **[`spm2lps`](@ref)** | Building a pencil based structured linearization `[A-λE B-λF; C-λG D-λH]` of a structured polynomial matrix `[T(λ) U(λ); V(λ) W(λ)]`|

```@docs
poly2pm
pm2poly
pmdeg
pmeval
pmreverse
pmdivrem
pm2lpCF1
pm2lpCF2
pm2ls
pm2lps
spm2ls
spm2lps
ls2pm
lps2pm
```
