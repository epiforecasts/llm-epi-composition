#!/usr/bin/env julia
# Generate development data; evaluation seeds remain reserved for registration.
# Usage: julia --project=simulations simulations/generate.jl --development OUTPUT SEED [BLOCKS]

include("renewal.jl")

function main(args)
    length(args) in (3, 4) && args[1] == "--development" ||
        error("Usage: generate.jl --development OUTPUT SEED [BLOCKS]")
    output = args[2]
    seed = parse(Int, args[3])
    blocks = length(args) == 4 ? parse(Int, args[4]) : 10
    RenewalSimulation.generate_development(output, seed; blocks)
    println("Development data written to $(abspath(output))")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
