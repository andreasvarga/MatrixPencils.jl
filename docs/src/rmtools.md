# Manipulation of rational matrices
| Function | Description |
| :--- | :--- |
| **[`rm2lspm`](@ref)** | Representation of a rational matrix as a linearization of its strictly proper part plus its polynomial part |
| **[`rmeval`](@ref)** | Evaluation of a rational matrix for a given value of its argument.|
| **[`rm2ls`](@ref)** | Building a descriptor system based structured linearization `[A-λE B; C D]` of a rational matrix |
| **[`ls2rm`](@ref)** | Computation of the rational matrix from its descriptor system based structured linearization |
| **[`rm2lps`](@ref)** | Building a pencil based structured linearization `[A-λE B-λF; C-λG D-λH]` of a rational matrix |
| **[`lps2rm`](@ref)** | Computation of the rational matrix from its pencil based structured linearization |
| **[`lpmfd2ls`](@ref)** | Building a descriptor system based structured linearization `[A-λE B; C D]` of a left polynomial matrix fractional description |
| **[`rpmfd2ls`](@ref)** | Building a descriptor system based structured linearization `[A-λE B; C D]` of a right polynomial matrix fractional description |
| **[`lpmfd2lps`](@ref)** | Building a pencil based structured linearization `[A-λE B-λF; C-λG D-λH]` of a left polynomial matrix fractional description |
| **[`rpmfd2lps`](@ref)** | Building a pencil based structured linearization `[A-λE B-λF; C-λG D-λH]` of a right polynomial matrix fractional description |
| **[`pminv2ls`](@ref)** | Building a descriptor system based structured linearization `[A-λE B; C D]` of the inverse of a polynomial matrix |
| **[`pminv2lps`](@ref)** | Building a pencil based structured linearization `[A-λE B-λF; C-λG D-λH]` of the inverse of a polynomial matrix |

```@docs
rm2lspm
rmeval
rm2ls
ls2rm
rm2lps
lps2rm
lpmfd2ls
rpmfd2ls
lpmfd2lps
rpmfd2lps
pminv2ls
pminv2lps
```
