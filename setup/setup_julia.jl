# Setup Julia environment for experiment execution
# Run with: julia setup/setup_julia.jl

using Pkg

println("Setting up Julia environment...")

# Packages from General registry (for Turing.jl experiments)
registry_packages = [
    "Turing",
    "Distributions",
    "StatsPlots",
    "DataFrames",
    "CSV",
    "MCMCChains",
    "Random",
    "Plots",
    "Dates"
]

println("Installing packages from registry...")
for pkg in registry_packages
    try
        @eval using $(Symbol(pkg))
        println("  $pkg: already installed")
    catch
        println("  Installing $pkg...")
        Pkg.add(pkg)
    end
end

# EpiAware from GitHub (not in General registry)
println("\nInstalling EpiAware from GitHub...")
try
    @eval using EpiAware
    println("  EpiAware: already installed")
catch
    println("  Installing EpiAware...")
    Pkg.add(Pkg.PackageSpec(url="https://github.com/CDCgov/Rt-without-renewal", subdir="EpiAware"))
end

# Precompile packages
println("\nPrecompiling packages...")
Pkg.precompile()

println("\nJulia environment setup complete!")
