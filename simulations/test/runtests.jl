using Test, Random, Distributions, CSV, DataFrames, JSON3, Dates, SHA

include("../renewal.jl")
const Sim = RenewalSimulation

@testset "Daily distributions and exact delay translation" begin
    @test Sim.daily_delay_pmf(Uniform(0, 1), 1) ≈ [0.5, 0.5]
    @test Sim.positive_gi_pmf(Uniform(0, 1), 3) ≈ [1.0, 0.0, 0.0]
    inputs = Sim.simulation_inputs()
    @test length(inputs.g) == 30
    @test sum(inputs.g) ≈ 1
    for distribution in values(inputs.delays)
        baseline = Sim.daily_delay_pmf(distribution, 30)
        shifted = Sim.daily_delay_pmf(distribution, 33; location = 3)
        @test shifted ≈ vcat(zeros(3), baseline) atol = 1e-10
        @test sum(shifted) ≈ 1
        baseline_mean = sum((0:30) .* baseline)
        shifted_mean = sum((0:33) .* shifted)
        @test shifted_mean ≈ baseline_mean + 3
        @test sum(((0:33) .- shifted_mean).^2 .* shifted) ≈
            sum(((0:30) .- baseline_mean).^2 .* baseline)
    end
    @test_throws ArgumentError Sim.daily_delay_pmf(Uniform(0, 1), 1; location = 3)
    @test_throws ArgumentError Sim.positive_gi_pmf(Uniform(0, 1), 0)
end

@testset "Positive-lag initialisation and instantaneous process" begin
    @test Sim.euler_lotka_rate(0.8, [1.0]) ≈ log(0.8)
    @test Sim.euler_lotka_rate(1.5, [0.0, 1.0]) ≈ log(1.5) / 2
    @test Sim.euler_lotka_rate(1.0, [0.2, 0.8]) == 0
    g = Sim.simulation_inputs().g
    growth = Sim.euler_lotka_rate(0.8, g)
    @test 0.8sum(g[s] * exp(-growth*s) for s in eachindex(g)) ≈ 1
    @test_throws ArgumentError Sim.euler_lotka_rate(0.8, [0.2, 0.2])

    R, Λ, k = 1.4, 50.0, 1.0
    distribution = Sim.infection_distribution(R, Λ, k)
    @test mean(distribution) ≈ R * Λ
    @test var(distribution) ≈ R * Λ + R^2 * Λ / k
    @test Sim.infection_distribution(R, Λ, Inf) isa Poisson
    @test var(Sim.infection_distribution(R, Λ, 1e8)) ≈ R * Λ rtol = 1e-7
    @test rand(MersenneTwister(501), Sim.infection_distribution(R, 0.0, k)) == 0
    @test_throws ArgumentError Sim.infection_distribution(R, Λ, 0)

    # With a unit-lag GI, each day's multiplier acts on yesterday's count.
    rt = [0.8, 2.0, 0.3]
    result = Sim.simulate_infections(MersenneTwister(502), rt, [1.0])
    rng = MersenneTwister(502)
    expected = Int[]
    previous = 1000
    for R in rt
        previous = rand(rng, NegativeBinomial(previous, 1 / (1 + R)))
        push!(expected, previous)
    end
    @test result.infections == expected
    @test result.infectiousness == vcat(1000, expected[1:end-1])

    rt = Sim.draw_trajectory(MersenneTwister(503)).rt
    normal = Sim.simulate_infections(MersenneTwister(504), rt, g)
    longer = Sim.simulate_infections(MersenneTwister(504), rt, g; prehistory = 120)
    @test normal.history[end] == 1000
    @test normal.infections == longer.infections
    @test normal.history == longer.history[end-59:end]
    empty = Sim.simulate_infections(MersenneTwister(505), rt, g; initial_incidence = 0)
    @test all(iszero, empty.infections)
end

@testset "Report-date ascertainment, prehistory and weekly scale" begin
    # Day one receives the seeded day-zero infections under a one-day delay.
    means = Sim.observation_means([20, 40], [10], [0.0, 1.0], [0.2, 0.8], ones(7))
    @test means ≈ [2.0, 16.0]
    same_day = Sim.observation_means([20, 40], Int[], [1.0], [0.2, 0.8], ones(7))
    @test same_day ≈ [4.0, 32.0]
    @test sum(Sim.WEEKLY_FACTORS) / 7 ≈ 1
    weekday = Sim.observation_means(fill(60, 7), Int[], [1.0], ones(7),
        Sim.WEEKLY_FACTORS; start_date = Date(2023, 1, 2))
    @test weekday ≈ [70, 70, 70, 70, 70, 35, 35]
    @test_throws ArgumentError Sim.observation_means([20], Int[], [0.0, 1.0], [0.2], ones(7))
end

@testset "Trajectory family and separated random streams" begin
    manifest = Sim.seed_manifest(2026101201, 10)
    @test manifest == Sim.seed_manifest(2026101201, 10)
    @test nrow(manifest) == 10 * 3 * 6
    @test length(unique(manifest.seed)) == nrow(manifest)
    @test all(s -> !(s in 101:120) && s != 20261012, manifest.seed)
    for direction in (-1, 0, 1)
        trajectory = Sim.draw_trajectory(MersenneTwister(506); tail = direction)
        @test trajectory.rt[1] == 0.8
        @test trajectory.rt[trajectory.knots] ≈ trajectory.values
        @test 35 <= trajectory.knots[2] <= 55
        @test 85 <= trajectory.knots[3] <= 110
        @test sign(trajectory.rt[end] - trajectory.rt[trajectory.knots[3]]) == direction
    end
end

@testset "Development output, source separation and overwrite protection" begin
    mktempdir() do temporary
        output = joinpath(temporary, "pilot")
        Sim.generate_development(output, 2026101202; blocks = 1)
        manifest = JSON3.read(read(joinpath(output, "development_manifest.json"), String))
        @test manifest.purpose == "development"
        @test manifest.rng == "MersenneTwister"
        for (filename, digest) in pairs(manifest.source_sha256)
            @test bytes2hex(sha256(read(joinpath(output, "provenance", string(filename))))) == digest
        end
        for phase in Sim.PHASES
            root = joinpath(output, "block_01", string(phase))
            richer = read(joinpath(root, "scenario_2", "data", "cases.csv"))
            @test richer == read(joinpath(root, "scenario_3", "data", "cases.csv"))
            for task in ("scenario_1b", "scenario_2", "scenario_3")
                data = joinpath(root, task, "data")
                config = JSON3.read(read(joinpath(data, "config.json"), String))
                @test !haskey(config, :rt_knots)
                @test !haskey(config, :gi_pmf)
                @test !haskey(config, :weekly_factors)
                @test !ispath(joinpath(data, "truth"))
                @test config.delays.cases.location == 0
                @test config.delays.cases.lag_max == 30
                observations = CSV.read(joinpath(data, "cases.csv"), DataFrame)
                @test nrow(observations) == 150
                @test all(>=(0), observations.cases)
            end
        end
        heldout = JSON3.read(read(joinpath(output, "block_01", "heldout", "truth", "params.json"), String))
        diagnostic = JSON3.read(read(joinpath(output, "block_01", "diagnostic", "truth", "params.json"), String))
        @test heldout.tail_direction == 0 ? abs(diagnostic.tail_direction) == 1 :
            diagnostic.tail_direction == -heldout.tail_direction
        @test_throws ArgumentError Sim.generate_development(output, 2026101202; blocks = 1)
        @test_throws ArgumentError Sim.generate_development(joinpath(temporary, "reserved"), 20261012)
    end
end
