using Documenter, MatrixPencils
DocMeta.setdocmeta!(MatrixPencils, :DocTestSetup, :(using MatrixPencils); recursive=true)

makedocs(
  modules  = [MatrixPencils],
  sitename = "MatrixPencils.jl",
  authors  = "Andreas Varga",
  format   = Documenter.HTML(prettyurls = false),
  pages    = [
     "Home"   => "index.md",
     "Library" => [
        "klftools.md",
        "klfapps.md",
        "pregular.md"
     ],
     "Internal" => [
        "klftools_int.md"
     ],
     "Index" => "makeindex.md"
  ]
)

# deploydocs(deps = nothing, make = nothing,
#   repo = "github.com/andreasvarga/MatrixPencils.jl.git",
#   target = "build",
#   devbranch = "master"
# )