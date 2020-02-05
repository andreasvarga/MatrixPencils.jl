module MatrixPencils
# Release V0.1

const BlasFloat = Union{Float64,Float32,ComplexF64,ComplexF32}
const BlasReal = Union{Float64,Float32}
const BlasComplex = Union{ComplexF64,ComplexF32}

using LinearAlgebra

export klf, klf_left, klf_right, klf_rlsplit, klf_left!, klf_right!, klf_left_refine!, klf_right_refine!
export prank, pkstruct, peigvals, pzeros, isregular, KRInfo
#export dss, ss, gnrank, gzero, gpole
#export LTISystem, AbstractDescriptorStateSpace
#export _preduceBF!, _preduce1!, _preduce2!, _preduce3!, _preduce4!

#import Base: +, -, *

include("klftools.jl")
#include("dstools.jl")
include("klfapps.jl")
include("lputil.jl")
end
