#!/usr/bin/env julia
# Development stream-count and prehistory checks; no fitting or evaluation seeds.
# Usage: julia --project=simulations simulations/validate_generator.jl OUTPUT SEED [BLOCKS]

using CSV, DataFrames, Distributions, Random, Statistics, SHA
include("renewal.jl")

function validate_generator(output, master; blocks = 100)
    (master == 20261012 || master in 101:120) &&
        throw(ArgumentError("Use a separate development master"))
    manifest = RenewalSimulation.seed_manifest(master, blocks)
    ispath(output) && throw(ArgumentError("Output already exists; refusing to overwrite"))
    mkpath(output)
    CSV.write(joinpath(output, "seed_manifest.csv"), manifest)
    inputs = RenewalSimulation.simulation_inputs()
    windows = (all = 1:150, interior = 25:125, recent = 144:150, current = 150:150)
    rows = NamedTuple[]
    trajectories = NamedTuple[]
    boundary_matches = true
    for block in 1:blocks, entry in RenewalSimulation.draw_block(manifest, block)
        (; phase, seeds, trajectory) = entry
        dataset = RenewalSimulation.simulate_dataset(trajectory, seeds, inputs)
        extended = RenewalSimulation.simulate_infections(MersenneTwister(seeds[:infections]),
            trajectory.rt, inputs.g; prehistory = 120)
        boundary_matches &= extended.infections == dataset.infection.infections
        for stream in RenewalSimulation.STREAMS
            extended_mean = RenewalSimulation.observation_means(extended.infections,
                extended.history, inputs.delay_pmfs[stream], inputs.ascertainment[stream],
                RenewalSimulation.WEEKLY_FACTORS)
            boundary_matches &= extended_mean == dataset.means[stream]
        end
        push!(trajectories, (; block, phase = string(phase), tail = trajectory.tail,
            extinct_final_day = iszero(last(dataset.infection.infections))))
        for stream in (:simple_cases, RenewalSimulation.STREAMS...)
            expected = stream == :simple_cases ? dataset.simple_means : dataset.means[stream]
            observed = stream == :simple_cases ? dataset.simple_cases : dataset.counts[stream]
            for (window, days) in pairs(windows)
                push!(rows, (; block, phase = string(phase), tail = trajectory.tail,
                    stream = string(stream), window = string(window),
                    expected_total = sum(expected[days]), expected_minimum = minimum(expected[days]),
                    observed_total = sum(observed[days]), zero_days = count(iszero, observed[days])))
            end
        end
    end
    counts = DataFrame(rows)
    CSV.write(joinpath(output, "stream_counts.csv"), counts)
    CSV.write(joinpath(output, "trajectories.csv"), DataFrame(trajectories))
    summary = combine(groupby(counts, [:stream, :window]),
        :expected_total => minimum => :minimum_expected_total,
        :expected_total => median => :median_expected_total,
        :observed_total => median => :median_observed_total,
        :observed_total => (x -> mean(iszero.(x))) => :zero_total_fraction,
        :expected_minimum => (x -> mean(x .< 1)) => :fraction_with_expected_day_below_one)
    CSV.write(joinpath(output, "count_summary.csv"), summary)
    RenewalSimulation.write_json(joinpath(output, "validation.json"), Dict(
        "purpose" => "development", "master_seed" => master, "blocks" => blocks,
        "datasets" => 3blocks, "julia_version" => string(VERSION),
        "rng" => "MersenneTwister", "longer_prehistory_matches" => boundary_matches,
        "final_day_extinction_fraction" => mean(t.extinct_final_day for t in trajectories),
        "renewal_source_sha256" => bytes2hex(sha256(read(joinpath(@__DIR__, "renewal.jl")))),
        "validation_source_sha256" => bytes2hex(sha256(read(@__FILE__))),
        "manifest_sha256" => bytes2hex(sha256(read(joinpath(@__DIR__, "Manifest.toml"))))))
    boundary_matches || error("Longer prehistory changed infections or observation means")
    println(summary)
    return summary
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) in (2, 3) || error("Usage: validate_generator.jl OUTPUT SEED [BLOCKS]")
    blocks = length(ARGS) == 3 ? parse(Int, ARGS[3]) : 100
    validate_generator(ARGS[1], parse(Int, ARGS[2]); blocks)
end
