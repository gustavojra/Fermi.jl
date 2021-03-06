using Documenter, Fermi

makedocs(
    sitename="Fermi.jl",
    authors = "Gustavo Aroeira",
    format = Documenter.HTML(
        sidebar_sitename = false
    ),
    pages = [
        "Home" => "index.md",
        "Modules" => "modules.md",
        "Core" => "core.md",
        "Backend" => "backend.md",
        "Methods" => "methods.md",
        "Contributing" => "contributing.md",
        "Index" => "indice.md"
    ]
)

deploydocs(
    repo = "github.com/FermiQC/Fermi.jl.git",
)
