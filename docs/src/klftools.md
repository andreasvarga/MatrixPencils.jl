# Manipulation of general linear matrix pencils
| Function | Description |
| :--- | :--- |
| **[`pbalance!`](@ref)** |Balancing arbitrary matrix pencils.  |
| **[`pbalqual`](@ref)** |Balancing quality of a matrix pencils.  |
| **[`klf`](@ref)** |   Computation of the Kronecker-like form exhibiting the full Kronecker structure |
| **[`klf_right`](@ref)** |   Computation of the Kronecker-like form exhibiting the right and finite Kronecker structures |
| **[`klf_rightinf`](@ref)** |   Computation of the Kronecker-like form exhibiting the right and infinite Kronecker structures |
| **[`klf_left`](@ref)** |  Computation of the Kronecker-like form exhibiting the left and finite Kronecker structures |
| **[`klf_leftinf`](@ref)** |  Computation of the Kronecker-like form exhibiting the left and infinite Kronecker structures |
| **[`klf_rlsplit`](@ref)** | Computation of the Kronecker-like form exhibiting the separation of right and left Kronecker structures |
| **[`preduceBF`](@ref)** | Reduction to the basic condensed form  `[B A-λE; D C]` with `E` upper triangular and nonsingular. |
| **[`klf_right!`](@ref)** |   Computation of the Kronecker-like form exhibiting the right and finite Kronecker structures |
| **[`klf_right_refine!`](@ref)** |   Update the Kronecker-like form by splitting the right and infinite Kronecker structures |
| **[`klf_right_refineut!`](@ref)** |   Refine the Kronecker-like form by enforcing upper triangular shapes of blocks in the leading full row rank subpencil |
| **[`klf_right_refineinf!`](@ref)** |   Refine the Kronecker-like form by enforcing upper triangular shapes of blocks in its infinite regular part |
| **[`klf_left!`](@ref)** |   Computation of the Kronecker-like form exhibiting the left and finite Kronecker structures |
| **[`klf_left_refine!`](@ref)** |   Update the Kronecker-like form by splitting the left and infinite Kronecker structures |
| **[`klf_left_refineut!`](@ref)** |   Refine the Kronecker-like form by enforcing upper triangular shapes of blocks in the leading full row rank subpencil |
| **[`klf_left_refineinf!`](@ref)** |   Refine the Kronecker-like form by enforcing upper triangular shapes of blocks in its infinite regular part |

```@docs
pbalance!
pbalqual
klf
klf_right
klf_rightinf
klf_rlsplit
preduceBF
klf_right!
MatrixPencils.klf_right_refine!
MatrixPencils.klf_right_refineut!
MatrixPencils.klf_right_refineinf!
klf_left
klf_leftinf
MatrixPencils.klf_left!
MatrixPencils.klf_left_refine!
MatrixPencils.klf_left_refineut!
MatrixPencils.klf_left_refineinf!
```
