# Setup Julia environment for experiment execution
# Run with: julia setup/setup_julia.jl

using Pkg

println("Setting up Julia environment...")

# Activate default environment
Pkg.activate()

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
    if Base.find_package(pkg) === nothing
        println("  Installing $pkg...")
        Pkg.add(pkg)
    else
        println("  $pkg: already installed")
    end
end

# EpiAware from GitHub (CDCgov repo)
println("\nInstalling EpiAware from GitHub...")
if Base.find_package("EpiAware") === nothing
    println("  Installing EpiAware...")
    Pkg.add(url="https://github.com/CDCgov/Rt-without-renewal.git", subdir="EpiAware")
else
    println("  EpiAware: already installed")
end

# Precompile packages
println("\nPrecompiling packages...")
Pkg.precompile()

println("\nJulia environment setup complete!")
