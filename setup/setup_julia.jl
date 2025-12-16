# Setup Julia environment for experiment execution
# Run with: julia setup/setup_julia.jl

using Pkg

println("Setting up Julia environment...")

# Activate default environment (create if needed)
default_env = joinpath(homedir(), ".julia", "environments", "v$(VERSION.major).$(VERSION.minor)")
mkpath(default_env)
Pkg.activate(default_env)

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

# Warm up packages to avoid segfaults on first use
println("\nWarming up packages...")
try
    @eval using Turing, Distributions, DataFrames, CSV, MCMCChains, Plots
    println("  Packages loaded successfully")
catch e
    println("  Warning: Could not load all packages: $e")
end

println("\nJulia environment setup complete!")
