module Runtests

using Test, MatrixPencils

@testset "Test MatrixPencils.jl" begin
    include("test_klf.jl")
    include("test_klfapps.jl")
    include("test_regular.jl")
    include("test_sklf.jl")
    include("test_sklfapps.jl")
    include("test_lstools.jl")
    include("test_pmtools.jl")
    include("test_pmapps.jl")
end

end
