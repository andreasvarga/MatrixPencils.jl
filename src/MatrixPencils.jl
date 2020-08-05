module MatrixPencils
# Release V0.6

const BlasFloat = Union{Float64,Float32,ComplexF64,ComplexF32}
const BlasReal = Union{Float64,Float32}
const BlasComplex = Union{ComplexF64,ComplexF32}

using LinearAlgebra
using Polynomials
using Random

include("lapackutil2.jl")
import .LapackUtil2: larfg!, larfgl!, larf!

export preduceBF, klf, klf_left, klf_right, klf_rlsplit
export prank, pkstruct, peigvals, pzeros, KRInfo
export isregular, isunimodular, fisplit, _svdlikeAE!
export sreduceBF, sklf, sklf_right, sklf_left, sklf_right!, sklf_left!, sklf_rightfin!, sklf_leftfin! 
export sprank, spkstruct, speigvals, spzeros
export lsminreal, lsminreal2, lsequal, lseval
export lpsminreal, lpsequal, lpseval
export poldivrem, polgcdvw, pollcm, conv, poldiv, convmtx, gcdvwupd, qrsolve!, poldeg, poldeg1
export poly2pm, pm2poly, pmdeg, pmreverse, pmeval, pmdivrem
export pm2lpCF1, pm2lpCF2, pm2lps, pm2ls, spm2ls, spm2lps, ls2pm, lps2pm
export pmkstruct, pmeigvals, pmzeros, pmzeros1, pmzeros2, pmroots, pmpoles, pmpoles1, pmpoles2, pmrank, ispmregular, ispmunimodular 
export rmeval, rm2lspm, rm2ls, ls2rm, rm2lps, lps2rm
export lpmfd2ls, rpmfd2ls, lpmfd2lps, rpmfd2lps, pminv2ls, pminv2lps
export rmkstruct, rmzeros, rmzeros1, rmpoles, rmpoles1, rmrank 
#export polbascoeffs, polbasparams, polbas2mon, polmon2bas
#export dss, ss, gnrank, gzero, gpole
#export LTISystem, AbstractDescriptorStateSpace
#export _preduceBF!, _preduce1!, _preduce2!, _preduce3!, _preduce4!

#import Base: +, -, *
#import LinearAlgebra: copy_oftype, require_one_based_indexing
import LinearAlgebra: copy_oftype

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
#include("dstools.jl")
include("lputil.jl")
include("slputil.jl")
end
