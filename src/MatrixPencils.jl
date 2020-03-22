module MatrixPencils
# Release V0.2

const BlasFloat = Union{Float64,Float32,ComplexF64,ComplexF32}
const BlasReal = Union{Float64,Float32}
const BlasComplex = Union{ComplexF64,ComplexF32}

using LinearAlgebra
using Random

include("lapackutil2.jl")
using .LapackUtil2: larfg!, larfgl!, larf!


export klf, klf_left, klf_right, klf_rlsplit
export prank, pkstruct, peigvals, pzeros, KRInfo
export isregular, fisplit, _svdlikeAE!
export sreduceBF, sklf, sklf_right, sklf_left, sklf_right!, sklf_left!, sklf_rightfin!, sklf_leftfin! 
export sprank, spkstruct, speigvals, spzeros
export lsminreal, lsminreal2, lsequal
export lsbuild, lpbuild, lpbuildCF1, lpbuildCF2
export polbascoeffs, polbasparams, polbas2mon, polmon2bas, poldeg, polreverse
#export dss, ss, gnrank, gzero, gpole
#export LTISystem, AbstractDescriptorStateSpace
#export _preduceBF!, _preduce1!, _preduce2!, _preduce3!, _preduce4!

#import Base: +, -, *

include("klftools.jl")
include("regtools.jl")
include("klfapps.jl")
include("sklftools.jl")
include("sklfapps.jl")
include("lstools.jl")
#include("poltools.jl")
#include("dstools.jl")
include("lputil.jl")
include("slputil.jl")
end
