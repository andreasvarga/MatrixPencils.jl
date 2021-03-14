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
        "sklftools.md",
        "klfapps.md",
        "sklfapps.md",
        "pregular.md",
        "lstools.md",
        "pmtools.md",
        "pmapps.md",
        "rmtools.md",
        "rmapps.md"
     ],
     "Internal" => [
        "klftools_int.md"
     ],
     "Index" => "makeindex.md"
  ]
)

deploydocs(
  repo = "github.com/andreasvarga/MatrixPencils.jl.git",
  target = "build",
)
