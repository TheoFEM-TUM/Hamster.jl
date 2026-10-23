using Documenter
using Hamster

# The pages of the input tags are generated from the tag definitions (CONFIG_TAGS)
tag_pages = ["conf/tags/" * f for f in Hamster.write_tag_docs(joinpath(@__DIR__, "src", "conf", "tags"))]

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
        "Home" => "index.md",
        "Command-Line Interface" => "cli.md",
        "Config File" => [
            "conf/config.md",
            "Input Tags" => tag_pages
        ],
        "Calculation Types" => [
            "calc/standard.md",
            "calc/optim.md",
            "calc/hyperopt.md",
            "calc/gpu.md",
            "calc/dos.md",
            "calc/helper_functions.md"
        ],
        "Hamiltonian Models" => [
            "models/tb.md",
            "models/soc.md",
            "models/kernel.md",
            "models/ewald.md"
        ],
        "Examples" => [
            "examples/examples.md",
            "examples/model_eval.md",
            "examples/gaas_tb_hyperopt.md",
            "examples/gaas_ml_fit.md"
        ],
        "Library" => "library/library.md"
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/TheoFEM-TUM/Hamster.jl"
)
