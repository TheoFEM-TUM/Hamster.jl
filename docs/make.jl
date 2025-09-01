using Documenter
using Hamster

makedocs(
    sitename = "Hamster",
    modules = [Hamster],
    format = Documenter.HTML(;
    # Use clean URLs, unless built as a "local" build
    prettyurls=get(ENV, "CI", nothing) == "true",
    edit_link="master",
    size_threshold=nothing,  # do not fail build if large HTML outputs
    size_threshold_warn=nothing,
    ),
    pages = [
        "Home" => "index.md"
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/TheoFEM-TUM/Hamster.jl"
)
