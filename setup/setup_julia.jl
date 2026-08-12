# Instantiate the pinned Julia environment for the study harness.
#
# The environment is at evaluation/julia_env/ with tracked Project.toml
# and Manifest.toml. This script just activates it and calls Pkg.instantiate()
# so the exact pinned package versions are installed.
#
# Run once, from the repository root:
#   julia setup/setup_julia.jl

using Pkg

const REPO_ENV = normpath(joinpath(@__DIR__, "..", "evaluation", "julia_env"))

println("Instantiating pinned environment at $REPO_ENV")
Pkg.activate(REPO_ENV)
Pkg.instantiate()

println("Precompiling packages...")
Pkg.precompile()

println("\nDone.")
println("Activate in a Julia session with:")
println("    using Pkg; Pkg.activate(\"$REPO_ENV\")")
println("or from a shell with:")
println("    julia --project=$REPO_ENV")
