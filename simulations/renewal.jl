module RenewalSimulation

using CSV, DataFrames, Dates, Distributions, JSON3, QuadGK, Random, SHA

const PROTOCOL_VERSION = "instantaneous-renewal-development-v1"
const DAYS = 150
const START_DATE = Date(2023, 1, 1)
const GI_SUPPORT = 30
const DELAY_SUPPORT = 30
const PREHISTORY = 60
const WEEKLY_FACTORS = [7 / 6, 7 / 6, 7 / 6, 7 / 6, 7 / 6, 7 / 12, 7 / 12]
const STREAMS = (:cases, :hospitalisations, :deaths)
const PHASES = (:generation, :heldout, :diagnostic)
const SEED_ROLES = (:trajectory, :infections, :simple_cases, STREAMS...)

function gamma_from_mean_sd(m, s)
    isfinite(m) && isfinite(s) && m > 0 && s > 0 ||
        throw(ArgumentError("Mean and SD must be finite and positive"))
    return Gamma((m / s)^2, s^2 / m)
end

function lognormal_from_mean_sd(m, s)
    isfinite(m) && isfinite(s) && m > 0 && s > 0 ||
        throw(ArgumentError("Mean and SD must be finite and positive"))
    σ² = log1p((s / m)^2)
    return LogNormal(log(m) - σ² / 2, sqrt(σ²))
end

"""Report-delay masses integrating over uniform within-day infection time."""
function daily_delay_pmf(distribution, support::Int; location = 0.0)
    support >= 0 || throw(ArgumentError("Delay support must be non-negative"))
    isfinite(location) && location >= 0 ||
        throw(ArgumentError("Delay location must be finite and non-negative"))
    masses = map(0:support) do e
        mass, _ = quadgk(0.0, 1.0; rtol = 1e-10) do p
            cdf(distribution, e + 1 - p - location) -
                cdf(distribution, max(e - p, 0.0) - location)
        end
        mass
    end
    all(isfinite, masses) && all(>=(0), masses) && sum(masses) > 0 ||
        throw(ArgumentError("Delay distribution has no valid mass on support"))
    return masses / sum(masses)
end

"""GI uses daily integration conditioned on positive lags 1:support."""
function positive_gi_pmf(distribution, support::Int)
    support > 0 || throw(ArgumentError("GI support must be positive"))
    masses = daily_delay_pmf(distribution, support)[2:end]
    sum(masses) > 0 || throw(ArgumentError("GI has no positive-lag mass"))
    return masses / sum(masses)
end

function validate_gi(g)
    !isempty(g) && all(isfinite, g) && all(>=(0), g) &&
        isapprox(sum(g), 1; atol = 1e-12, rtol = 1e-12) ||
        throw(ArgumentError("GI must be a normalised non-negative positive-lag PMF"))
end

"""Solve 1 = R × sum(g[s] × exp(-r*s)); g[1] is the lag-one mass."""
function euler_lotka_rate(R, g)
    isfinite(R) && R > 0 || throw(ArgumentError("Initial R must be positive"))
    validate_gi(g)
    R == 1 && return 0.0
    function log_ratio(r)
        terms = [log(g[s]) - r * s for s in eachindex(g)]
        largest = maximum(terms)
        log(R) + largest + log(sum(exp(t - largest) for t in terms))
    end
    lo, hi = -1.0, 1.0
    while log_ratio(lo) < 0
        lo *= 2
    end
    while log_ratio(hi) > 0
        hi *= 2
    end
    for _ in 1:100
        mid = (lo + hi) / 2
        if log_ratio(mid) > 0
            lo = mid
        else
            hi = mid
        end
    end
    return (lo + hi) / 2
end

function infection_distribution(R, Λ, k)
    isfinite(R) && R >= 0 && isfinite(Λ) && Λ >= 0 ||
        throw(ArgumentError("R and infectiousness must be finite and non-negative"))
    k > 0 || throw(ArgumentError("Process dispersion must be positive"))
    (Λ == 0 || R == 0) && return Poisson(0.0)
    isinf(k) && return Poisson(R * Λ)
    return NegativeBinomial(k * Λ, k / (k + R))
end

function simulate_infections(rng, rt, g; k = 1.0, initial_incidence = 1000,
                             prehistory = PREHISTORY)
    validate_gi(g)
    !isempty(rt) && all(isfinite, rt) && all(>=(0), rt) ||
        throw(ArgumentError("R trajectory must be finite and non-negative"))
    initial_incidence >= 0 && prehistory >= length(g) ||
        throw(ArgumentError("Invalid initial incidence or insufficient prehistory"))
    growth = euler_lotka_rate(first(rt), g)
    history = round.(Int, initial_incidence .* exp.(growth .* (1-prehistory:0)))
    incidence = vcat(history, zeros(Int, length(rt)))
    infectiousness = zeros(length(rt))
    for d in eachindex(rt)
        index = prehistory + d
        Λ = sum(g[s] * incidence[index - s] for s in eachindex(g))
        infectiousness[d] = Λ
        incidence[index] = rand(rng, infection_distribution(rt[d], Λ, k))
    end
    return (; history, infections = incidence[prehistory+1:end], infectiousness, growth)
end

function draw_trajectory(rng; tail = nothing)
    t2, t3 = rand(rng, 35:55), rand(rng, 85:110)
    r2, r3 = rand(rng, Uniform(1.2, 1.6)), rand(rng, Uniform(0.75, 1.05))
    direction = if isnothing(tail)
        (0, 1, -1)[rand(rng, Categorical([0.4, 0.3, 0.3]))]
    else
        tail in (-1, 0, 1) || throw(ArgumentError("Tail must be -1, 0 or 1"))
        tail
    end
    amplitude = rand(rng, Uniform(0.1, 0.3))
    knots = [1, t2, t3, DAYS]
    values = [0.8, r2, r3, r3 + direction * amplitude]
    rt = map(1:DAYS) do d
        segment = min(searchsortedlast(knots, d), length(knots) - 1)
        fraction = (d - knots[segment]) / (knots[segment+1] - knots[segment])
        values[segment] + fraction * (values[segment+1] - values[segment])
    end
    return (; knots, values, rt, tail = direction)
end

function observation_means(infections, history, f, ascertainment, weekly;
                           start_date = START_DATE)
    n = length(infections)
    length(ascertainment) == n && length(weekly) == 7 ||
        throw(ArgumentError("Observation inputs have incompatible lengths"))
    !isempty(f) && all(isfinite, f) && all(>=(0), f) &&
        isapprox(sum(f), 1; atol = 1e-12, rtol = 1e-12) ||
        throw(ArgumentError("Delay PMF must be normalised and non-negative"))
    length(history) >= length(f) - 1 ||
        throw(ArgumentError("Insufficient infection history for reporting delays"))
    all(isfinite, ascertainment) && all(>=(0), ascertainment) &&
        all(isfinite, weekly) && all(>=(0), weekly) ||
        throw(ArgumentError("Observation rates must be finite and non-negative"))
    incidence = vcat(history, infections)
    return map(1:n) do d
        delayed = sum(f[e+1] * incidence[length(history) + d - e]
                      for e in 0:length(f)-1)
        ascertainment[d] * weekly[dayofweek(start_date + Day(d - 1))] * delayed
    end
end

function simulation_inputs()
    gi = gamma_from_mean_sd(5.5, 2.0)
    delays = Dict(:cases => lognormal_from_mean_sd(5.0, 2.0),
                  :hospitalisations => lognormal_from_mean_sd(10.0, 3.0),
                  :deaths => lognormal_from_mean_sd(20.0, 5.0))
    ascertainment = Dict(:cases => [0.4 + 0.2sin(2π*d/DAYS) for d in 1:DAYS],
        :hospitalisations => [0.04 + 0.02sin(2π*d/DAYS + π/3) for d in 1:DAYS],
        :deaths => [0.008 + 0.004cos(2π*d/(1.5DAYS)) for d in 1:DAYS])
    return (; gi, g = positive_gi_pmf(gi, GI_SUPPORT), delays, ascertainment,
            delay_pmfs = Dict(s => daily_delay_pmf(delays[s], DELAY_SUPPORT) for s in STREAMS))
end

"""Separate RNG seeds for trajectory, infection process and each observation draw."""
function seed_manifest(master::Integer, blocks::Int)
    master >= 0 && blocks > 0 || throw(ArgumentError("Invalid master seed or blocks"))
    rng = MersenneTwister(master)
    used = Set{Int}([0, master, 20261012, collect(101:120)...])
    rows = NamedTuple[]
    for block in 1:blocks, phase in PHASES, role in SEED_ROLES
        seed = Int(rand(rng, UInt32))
        while seed in used
            seed = Int(rand(rng, UInt32))
        end
        push!(used, seed)
        push!(rows, (; block, phase = string(phase), role = string(role), seed))
    end
    return DataFrame(rows)
end

function distribution_metadata(distribution)
    return Dict("family" => distribution isa Gamma ? "Gamma" : "LogNormal",
        "mean" => mean(distribution), "sd" => std(distribution))
end

function write_json(path, value)
    open(path, "w") do io
        JSON3.pretty(io, value)
        println(io)
    end
end

function simulate_dataset(trajectory, seeds, inputs)
    infection = simulate_infections(MersenneTwister(seeds[:infections]), trajectory.rt, inputs.g)
    dates = START_DATE .+ Day.(0:DAYS-1)
    means = Dict(s => observation_means(infection.infections, infection.history,
        inputs.delay_pmfs[s], inputs.ascertainment[s], WEEKLY_FACTORS) for s in STREAMS)
    counts = Dict(s => rand.(Ref(MersenneTwister(seeds[s])), Poisson.(means[s])) for s in STREAMS)
    simple_means = observation_means(infection.infections, infection.history,
        inputs.delay_pmfs[:cases], fill(0.4, DAYS), ones(7))
    simple_cases = rand.(Ref(MersenneTwister(seeds[:simple_cases])), Poisson.(simple_means))
    return (; infection, dates, means, counts, simple_means, simple_cases)
end

function write_dataset(root, trajectory, seeds, inputs)
    (; infection, dates, means, counts, simple_means, simple_cases) =
        simulate_dataset(trajectory, seeds, inputs)
    truth = joinpath(root, "truth")
    mkpath(truth)
    CSV.write(joinpath(truth, "true_rt.csv"), DataFrame(day = 1:DAYS, date = dates, R_t = trajectory.rt))
    CSV.write(joinpath(truth, "true_infections.csv"),
        DataFrame(day = 1-PREHISTORY:DAYS, I = vcat(infection.history, infection.infections)))
    CSV.write(joinpath(truth, "true_expected.csv"), DataFrame(day = 1:DAYS, date = dates,
        E_simple_cases = simple_means, E_cases = means[:cases],
        E_hospitalisations = means[:hospitalisations], E_deaths = means[:deaths]))
    write_json(joinpath(truth, "params.json"), Dict(
        "protocol_version" => PROTOCOL_VERSION, "rt_target" => "instantaneous",
        "k" => 1.0, "initial_incidence" => 1000, "prehistory_days" => PREHISTORY,
        "rt_knots" => trajectory.knots, "rt_values" => trajectory.values,
        "tail_direction" => trajectory.tail, "growth_rate" => infection.growth,
        "gi_pmf" => inputs.g, "delay_pmfs" => inputs.delay_pmfs,
        "weekly_factors" => WEEKLY_FACTORS, "seeds" => seeds,
        "gi_discretisation" => "uniform within-day integration, conditioned on positive lags 1:30",
        "delay_discretisation" => "uniform within-day integration, normalised on lags 0:30"))
    for task in ("scenario_1b", "scenario_2", "scenario_3")
        streams = task == "scenario_3" ? STREAMS : (:cases,)
        data = joinpath(root, task, "data")
        mkpath(data)
        rates = DataFrame(date = dates)
        for s in streams
            observed = task == "scenario_1b" ? simple_cases : counts[s]
            CSV.write(joinpath(data, "$(s).csv"), DataFrame(:date => dates, s => observed))
            rates[!, s] = task == "scenario_1b" ? fill(0.4, DAYS) : inputs.ascertainment[s]
        end
        CSV.write(joinpath(data, "ascertainment.csv"), rates)
        write_json(joinpath(data, "config.json"), Dict(
            "schema_version" => 1, "task" => task, "start_date" => string(first(dates)),
            "observation_cutoff" => string(last(dates)), "date_bin_width_days" => 1,
            "target" => "instantaneous effective reproduction number, indexed by transmission date",
            "gi" => merge(distribution_metadata(inputs.gi), Dict("lag_min" => 1, "lag_max" => GI_SUPPORT)),
            "delays" => Dict(s => merge(distribution_metadata(inputs.delays[s]),
                Dict("location" => 0.0, "lag_min" => 0, "lag_max" => DELAY_SUPPORT)) for s in streams),
            "ascertainment_file" => "ascertainment.csv",
            "ascertainment_interpretation" => "baseline report-date rates on a mean-one weekly reporting-factor scale"))
    end
    return nothing
end

function draw_block(manifest, block)
    primary_tail = 0
    entries = NamedTuple[]
    for phase in PHASES
        rows = manifest[(manifest.block .== block) .& (manifest.phase .== string(phase)), :]
        seeds = Dict(Symbol(row.role) => row.seed for row in eachrow(rows))
        trajectory_rng = MersenneTwister(seeds[:trajectory])
        tail = phase == :diagnostic ?
            (primary_tail == 0 ? rand(trajectory_rng, (-1, 1)) : -primary_tail) : nothing
        trajectory = draw_trajectory(trajectory_rng; tail)
        phase == :heldout && (primary_tail = trajectory.tail)
        push!(entries, (; phase, seeds, trajectory))
    end
    return entries
end

function generate_development(output, master::Integer; blocks = 10)
    (master == 20261012 || master in 101:120) &&
        throw(ArgumentError("Use a separate development master, not reserved or legacy seeds"))
    manifest = seed_manifest(master, blocks)
    ispath(output) && throw(ArgumentError("Output already exists; refusing to overwrite"))
    mkpath(output)
    CSV.write(joinpath(output, "seed_manifest.csv"), manifest)
    inputs = simulation_inputs()
    for block in 1:blocks, entry in draw_block(manifest, block)
        root = joinpath(output, "block_$(lpad(block, 2, '0'))", string(entry.phase))
        write_dataset(root, entry.trajectory, entry.seeds, inputs)
    end
    source = joinpath(output, "provenance")
    mkpath(source)
    hashes = Dict{String,String}()
    for filename in ("renewal.jl", "generate.jl", "Project.toml", "Manifest.toml")
        path = joinpath(@__DIR__, filename)
        cp(path, joinpath(source, filename))
        hashes[filename] = bytes2hex(sha256(read(path)))
    end
    write_json(joinpath(output, "development_manifest.json"), Dict(
        "purpose" => "development", "protocol_version" => PROTOCOL_VERSION,
        "master_seed" => master, "blocks" => blocks, "rng" => "MersenneTwister",
        "julia_version" => string(VERSION), "source_sha256" => hashes))
    return manifest
end

end
