module MatrixPencils

const BlasFloat = Union{Float64,Float32,ComplexF64,ComplexF32}
const BlasReal = Union{Float64,Float32}
const BlasComplex = Union{ComplexF64,ComplexF32}

using LinearAlgebra
using Polynomials
import LinearAlgebra: copy_oftype, BlasInt

include("lapackutil2.jl")
import .LapackUtil2: larfg!, larfgl!, larf!, gghrd!, hgeqz!, tgexc!, tgsen!, lanv2, lag2, safemin, tgsyl!

export preduceBF, klf, klf_left, klf_leftinf, klf_right, klf_rightinf, klf_rlsplit, 
       klf_right!, klf_right_refine!, klf_right_refineut!, klf_right_refineinf!, 
       klf_left!, klf_left_refine!, klf_left_refineut!, klf_left_refineinf!
export prank, pkstruct, peigvals, pzeros, KRInfo
export isregular, isunimodular, fisplit, _svdlikeAE!, sfisplit
export sreduceBF, sklf, gsklf, sklf_right, sklf_left, sklf_right!, sklf_right2!, sklf_left!, sklf_rightfin!, sklf_rightfin2!, sklf_leftfin! 
export sprank, spkstruct, speigvals, spzeros
export lsminreal, lsminreal2, lsequal, lseval, lps2ls
export lpsminreal, lpsequal, lpseval
export poldivrem, polgcdvw, pollcm, conv, poldiv, convmtx, gcdvwupd, qrsolve!, poldeg, poldeg1
export poly2pm, pm2poly, pmdeg, pmreverse, pmeval, pmdivrem
export pm2lpCF1, pm2lpCF2, pm2lps, pm2ls, spm2ls, spm2lps, ls2pm, lps2pm
export pmkstruct, pmeigvals, pmzeros, pmzeros1, pmzeros2, pmroots, pmpoles, pmpoles1, pmpoles2, pmrank, ispmregular, ispmunimodular 
export rmeval, rm2lspm, rm2ls, ls2rm, rm2lps, lps2rm
export lpmfd2ls, rpmfd2ls, lpmfd2lps, rpmfd2lps, pminv2ls, pminv2lps
export rmkstruct, rmzeros, rmzeros1, rmpoles, rmpoles1, rmrank 
export saloc, salocd, salocinf, salocinfd, ordeigvals, isqtriu, eigselect1, eigselect2, saloc2
export fihess, fischur, fischursep, sfischursep, fiblkdiag, gsblkdiag, ssblkdiag
export pbalance!, regbalance!, lsbalance!, lsbalqual, pbalqual, qS1, rcsumsbal!

include("klftools.jl")
include("regtools.jl")
include("klfapps.jl")
include("sklftools.jl")
include("sklfapps.jl")
include("lstools.jl")
include("pmtools.jl")
include("pmapps.jl")
include("poltools.jl")
include("rmtools.jl")
include("rmapps.jl")
include("lputil.jl")
include("slputil.jl")
include("gsfstab.jl")
include("gsep.jl")
end
