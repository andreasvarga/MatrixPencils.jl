# Manipulation of structured linear matrix pencils
| Function | Description |
| :--- | :--- |
| **[`sreduceBF`](@ref)** | Reduction to the basic condensed form  `[B A-λE; D C]` with `E` upper triangular and nonsingular. |
| **[`sklf`](@ref)** |   Computation of the Kronecker-like form exhibiting the full Kronecker structure |
| **[`sklf_right`](@ref)** |   Computation of the Kronecker-like form exhibiting the right Kronecker structure |
| **[`sklf_left`](@ref)** |  Computation of the Kronecker-like form exhibiting the left Kronecker structure |
| **[`gsklf`](@ref)** | Computation of several row partition preserving special Kronecker-like forms |

```@docs
sklf
sklf_right
sklf_left
sreduceBF
gsklf
sklf_right!
sklf_right2!
sklf_left!
sklf_rightfin!
sklf_rightfin2!
sklf_leftfin!
```
