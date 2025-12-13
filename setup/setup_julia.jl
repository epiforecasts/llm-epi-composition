# Setup Julia environment for experiment execution
# Run with: julia setup/setup_julia.jl

using Pkg

println("Setting up Julia environment...")

# Packages needed for Turing.jl experiments
turing_packages = [
    "Turing",
    "Distributions",
    "StatsPlots",
    "DataFrames",
    "CSV",
    "MCMCChains",
    "Random"
]

# EpiAware and dependencies
epiaware_packages = [
    "EpiAware",
    "Turing",
    "Distributions",
    "DataFrames",
    "CSV",
    "Plots",
    "Dates"
]

# Combine unique packages
all_packages = unique(vcat(turing_packages, epiaware_packages))

println("Installing packages...")
for pkg in all_packages
    try
        @eval using $(Symbol(pkg))
        println("  $pkg: already installed")
    catch
        println("  Installing $pkg...")
        Pkg.add(pkg)
    end
end

# Precompile packages
println("\nPrecompiling packages...")
Pkg.precompile()

println("\nJulia environment setup complete!")
println("Installed packages:")
for pkg in all_packages
    try
        version = Pkg.dependencies()[Base.UUID(Pkg.project().dependencies[pkg])].version
        println("  $pkg: $version")
    catch
        # Package might be a stdlib or have different lookup
        println("  $pkg: installed")
    end
end
