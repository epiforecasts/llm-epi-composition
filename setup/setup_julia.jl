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
# Note: EpiAware may have compatibility issues with newer Julia versions
println("\nInstalling EpiAware from GitHub...")
try
    @eval using EpiAware
    println("  EpiAware: already installed")
catch
    println("  Attempting to install EpiAware...")
    try
        Pkg.add(url="https://github.com/CDCgov/Rt-without-renewal", subdir="EpiAware")
    catch e1
        try
            # Try without subdir
            Pkg.add(url="https://github.com/CDCgov/Rt-without-renewal")
        catch e2
            println("  WARNING: Could not install EpiAware. EpiAware experiments will fail.")
            println("  Error: $(e2)")
            println("  Continuing with other packages...")
        end
    end
end

# Precompile packages
println("\nPrecompiling packages...")
Pkg.precompile()

println("\nJulia environment setup complete!")
