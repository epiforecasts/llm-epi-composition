#!/usr/bin/env julia
# Ground-truth simulation generator: individual-level Lloyd-Smith Bellman-Harris
# branching process. The renewal equation is the expectation of this process
# (Mishra et al. 2020); we generate the realised process explicitly.
#
# Each infection at continuous time t_i:
#   - draws an individual reproduction number  ν_i ~ Gamma(k, R(t_i)/k)
#   - draws an offspring count                 Z_i ~ Poisson(ν_i)
#   - each offspring j draws a continuous GI   τ_ij ~ g(τ)
#   - the j-th offspring is born at            t_j = t_i + τ_ij
#
# k → ∞ recovers Poisson offspring (no individual heterogeneity).
#
# Daily aggregation only enters at the observation step:
#   E[C_d^stream] = α(d) · w_dow(d) · Σ_e f_e^stream · I_{d-e}
#   C_d^stream    ~ Poisson(E[C_d^stream])
# where I_{d-e} is the realised count in day d-e and f_e is the daily-PMF
# (double interval censoring) of the continuous delay.
#
# The recovery target is R(d) — definition (c) of Funk, Abbott & Bracher 2022.
#
# Usage:
#   julia --project=simulations simulations/generate.jl                  # all variants × all reps
#   julia --project=simulations simulations/generate.jl canonical        # one variant, all reps
#   julia --project=simulations simulations/generate.jl canonical 101    # one variant, one seed

using Distributions
using Random
using CSV
using DataFrames
using JSON3
using Dates
using QuadGK
using DataStructures

# ---- Fixed configuration ----

const TAU_MAX           = 20         # GI truncation (days)
const D_MAX             = 30         # delay truncation (days)
const T_DAYS            = 150
const START_DATE        = Date(2023, 1, 1)
const REPLICATES        = collect(101:120)
const DOW_MULT_CANON    = [1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5]   # Mon..Sun
const I_REF             = 100.0       # seed reference incidence at t = 0

# ---- Distribution parameterisations ----

gamma_from_mean_sd(m, s)     = Gamma(m^2 / s^2, s^2 / m)
function lognormal_from_mean_sd(m, s)
    σ² = log(1 + s^2 / m^2)
    LogNormal(log(m) - σ² / 2, sqrt(σ²))
end

# ---- Daily PMF via double interval censoring (used for delays only) ----

function double_censored_pmf(dist, dmax)
    pmf = Vector{Float64}(undef, dmax + 1)
    for d in 0:dmax
        val, _ = quadgk(p -> cdf(dist, d + 1 - p) - cdf(dist, max(d - p, 0.0)),
                        0.0, 1.0; rtol = 1e-10)
        pmf[d + 1] = val
    end
    pmf ./= sum(pmf)
    return pmf
end

# ---- Euler-Lotka growth rate (for seeding the equilibrium-like profile) ----

function euler_lotka_r(R0, g; tol = 1e-12, maxiter = 200)
    f(r) = R0 * sum(g[s] * exp(-r * (s - 1)) for s in 1:length(g)) - 1
    lo, hi = -5.0, 5.0
    flo, fhi = f(lo), f(hi)
    (flo * fhi < 0) || error("euler_lotka: bracket not found for R0=$R0")
    for _ in 1:maxiter
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if sign(fm) == sign(flo); lo, flo = mid, fm; else; hi, fhi = mid, fm; end
        abs(hi - lo) < tol && break
    end
    return 0.5 * (lo + hi)
end

# ---- R(t) piecewise-linear interpolation ----

struct RtKnots
    days::Vector{Float64}
    values::Vector{Float64}
end

function r_at(k::RtKnots, t)
    t <= k.days[1]   && return k.values[1]
    t >= k.days[end] && return k.values[end]
    i = searchsortedfirst(k.days, t)
    k.days[i] == t && return k.values[i]
    d0, d1 = k.days[i - 1], k.days[i]
    v0, v1 = k.values[i - 1], k.values[i]
    return v0 + (v1 - v0) * (t - d0) / (d1 - d0)
end

# ---- Variant definitions ----

struct Variant
    name::String
    gi::Distribution
    delay_cases::Distribution
    delay_hosp::Distribution
    delay_deaths::Distribution
    alpha_cases::Function
    alpha_hosp::Function
    alpha_deaths::Function
    dow_mult::Vector{Float64}
    k::Float64
    rt_knots::RtKnots
end

const ALPHA_CASES_CANON  = t -> 0.40  + 0.20  * sin(2π * t / T_DAYS)
const ALPHA_HOSP_CANON   = t -> 0.040 + 0.020 * sin(2π * t / T_DAYS + π / 3)
const ALPHA_DEATHS_CANON = t -> 0.008 + 0.004 * cos(2π * t / (1.5 * T_DAYS))
const RT_KNOTS_CANON     = RtKnots([1.0, 50.0, 100.0, 150.0], [0.8, 1.5, 0.8, 0.8])

function all_variants()
    gi_c    = gamma_from_mean_sd(5.5, 2.0)
    del_c   = lognormal_from_mean_sd(5.0, 2.0)
    del_h   = lognormal_from_mean_sd(10.0, 3.0)
    del_dd  = lognormal_from_mean_sd(20.0, 5.0)

    K_CANON = 1.0

    canon = Variant("canonical", gi_c, del_c, del_h, del_dd,
        ALPHA_CASES_CANON, ALPHA_HOSP_CANON, ALPHA_DEATHS_CANON,
        DOW_MULT_CANON, K_CANON, RT_KNOTS_CANON)

    short_gi = Variant("short_gi", gamma_from_mean_sd(2.5, 1.0),
        del_c, del_h, del_dd,
        ALPHA_CASES_CANON, ALPHA_HOSP_CANON, ALPHA_DEATHS_CANON,
        DOW_MULT_CANON, K_CANON, RT_KNOTS_CANON)

    long_delay = Variant("long_delay", gi_c,
        lognormal_from_mean_sd(10.0, 3.0), del_h, del_dd,
        ALPHA_CASES_CANON, ALPHA_HOSP_CANON, ALPHA_DEATHS_CANON,
        DOW_MULT_CANON, K_CANON, RT_KNOTS_CANON)

    strong_dow = Variant("strong_dow", gi_c, del_c, del_h, del_dd,
        ALPHA_CASES_CANON, ALPHA_HOSP_CANON, ALPHA_DEATHS_CANON,
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.25, 0.25],
        K_CANON, RT_KNOTS_CANON)

    high_asc_var = Variant("high_asc_var", gi_c, del_c, del_h, del_dd,
        t -> 0.40 + 0.35 * sin(2π * t / T_DAYS),
        ALPHA_HOSP_CANON, ALPHA_DEATHS_CANON,
        DOW_MULT_CANON, K_CANON, RT_KNOTS_CANON)

    low_disp = Variant("low_dispersion", gi_c, del_c, del_h, del_dd,
        ALPHA_CASES_CANON, ALPHA_HOSP_CANON, ALPHA_DEATHS_CANON,
        DOW_MULT_CANON, 1000.0, RT_KNOTS_CANON)

    high_disp = Variant("extreme_dispersion", gi_c, del_c, del_h, del_dd,
        ALPHA_CASES_CANON, ALPHA_HOSP_CANON, ALPHA_DEATHS_CANON,
        DOW_MULT_CANON, 0.1, RT_KNOTS_CANON)

    abrupt_knots = RtKnots([1.0, 50.0, 74.0, 77.0, 150.0], [0.8, 1.5, 1.5, 0.5, 0.5])
    abrupt = Variant("abrupt_change", gi_c, del_c, del_h, del_dd,
        ALPHA_CASES_CANON, ALPHA_HOSP_CANON, ALPHA_DEATHS_CANON,
        DOW_MULT_CANON, K_CANON, abrupt_knots)

    # Sinusoidal Rt: continuous oscillation, never piecewise linear. Tests
    # whether smoothing priors that prefer piecewise-linear are over-fitted.
    sin_days = collect(1.0:1.0:150.0)
    sin_vals = 1.0 .+ 0.4 .* sin.(2π .* sin_days ./ 60.0)
    sinusoidal_knots = RtKnots(sin_days, sin_vals)
    sinusoidal = Variant("sinusoidal_rt", gi_c, del_c, del_h, del_dd,
        ALPHA_CASES_CANON, ALPHA_HOSP_CANON, ALPHA_DEATHS_CANON,
        DOW_MULT_CANON, K_CANON, sinusoidal_knots)

    return [canon, short_gi, long_delay, strong_dow, high_asc_var, low_disp, high_disp, abrupt, sinusoidal]
end

# ---- Bellman-Harris BP simulation ----

function bp_simulate(v::Variant, rng::AbstractRNG)
    g_dist = v.gi
    k = v.k

    # Place seed individuals at rate seed_scale·I_REF·exp(r₀·t) per day so that,
    # after BP propagation through descendants in the pre-obs window, the
    # realised rate at t ≈ 0 equilibrates to ≈ I_REF. Without rescaling, the
    # geometric chain of offspring inflates the realised rate; the asymptotic
    # 1/(1 − R₀) factor under-shoots given the finite seed window, so seed_scale
    # is calibrated empirically to make BP I[1] match the moment-closed
    # population NegBinL I[1] on canonical (R₀=0.8 → seed_scale=0.3).
    # Seeds themselves do not count as new obs-window infections; their
    # descendants do.
    g_pmf_for_r0 = double_censored_pmf(v.gi, TAU_MAX)
    R0 = r_at(v.rt_knots, 1.0)
    r0 = euler_lotka_r(R0, g_pmf_for_r0)
    seed_scale = 0.3

    seeds = Float64[]
    for d in (-(TAU_MAX - 1)):0
        λ_int = if abs(r0) > 1e-9
            seed_scale * I_REF * (exp(r0 * d) - exp(r0 * (d - 1))) / r0
        else
            seed_scale * I_REF
        end
        n_d = rand(rng, Poisson(max(λ_int, 0.0)))
        for _ in 1:n_d
            push!(seeds, d - 1 + rand(rng))
        end
    end

    queue = BinaryMinHeap{Float64}()
    for t_i in seeds
        R_i = r_at(v.rt_knots, t_i)
        R_i > 0 || continue
        ν_i = !isfinite(k) ? R_i : rand(rng, Gamma(k, R_i / k))
        Z_i = rand(rng, Poisson(max(ν_i, 0.0)))
        for _ in 1:Z_i
            τ = rand(rng, g_dist)
            t_j = t_i + τ
            if t_j > -TAU_MAX && t_j <= T_DAYS
                push!(queue, t_j)
            end
        end
    end

    realised = Float64[]
    while !isempty(queue)
        t_i = pop!(queue)
        t_i > T_DAYS && continue
        push!(realised, t_i)
        R_i = r_at(v.rt_knots, t_i)
        R_i > 0 || continue
        ν_i = !isfinite(k) ? R_i : rand(rng, Gamma(k, R_i / k))
        Z_i = rand(rng, Poisson(max(ν_i, 0.0)))
        for _ in 1:Z_i
            τ = rand(rng, g_dist)
            t_j = t_i + τ
            if t_j > -TAU_MAX && t_j <= T_DAYS
                push!(queue, t_j)
            end
        end
    end

    return realised, r0
end

function aggregate_daily(realised, day_lo::Int, day_hi::Int)
    n = day_hi - day_lo + 1
    out = zeros(Int, n)
    for t in realised
        if t > day_lo - 1 && t <= day_hi
            d = Int(ceil(t)) - day_lo + 1
            out[d] += 1
        end
    end
    return out
end

# ---- Observation process ----

function expected_reports(I_obs::Vector{Int}, I_seed::Vector{Int},
                          delay_dist, alpha_fn, dow_mult)
    f = double_censored_pmf(delay_dist, D_MAX)
    n_f = length(f)
    n_seed = length(I_seed)
    dates = [START_DATE + Day(d - 1) for d in 1:T_DAYS]
    E = zeros(Float64, T_DAYS)
    for d in 1:T_DAYS
        s = 0.0
        @inbounds for e in 1:n_f
            past = d - (e - 1)
            past_val = if past >= 1
                Float64(I_obs[past])
            elseif past >= -(n_seed - 1)
                Float64(I_seed[end + past])
            else
                0.0
            end
            s += f[e] * past_val
        end
        α_d = alpha_fn(Float64(d))
        w_d = dow_mult[dayofweek(dates[d])]
        E[d] = α_d * w_d * s
    end
    return E, dates, f
end

function sample_obs(E, rng)
    out = Vector{Int}(undef, length(E))
    for i in eachindex(E)
        out[i] = rand(rng, Poisson(max(E[i], 1e-12)))
    end
    return out
end

# ---- Output ----

function dist_spec(d::Distribution)
    if d isa Gamma
        shape, scale = Distributions.params(d)
        return Dict("family" => "Gamma", "shape" => shape, "scale" => scale,
                    "mean" => shape * scale, "sd" => sqrt(shape) * scale)
    elseif d isa LogNormal
        μ, σ = Distributions.params(d)
        return Dict("family" => "LogNormal", "mu_log" => μ, "sigma_log" => σ,
                    "mean" => exp(μ + σ^2 / 2),
                    "sd" => sqrt((exp(σ^2) - 1) * exp(2μ + σ^2)))
    else
        return Dict("family" => string(typeof(d)),
                    "params" => collect(Distributions.params(d)))
    end
end

function write_replicate(v::Variant, rep_idx::Int, seed::Int, script_path::String)
    rep_dir   = joinpath("simulations", v.name, "rep_$(lpad(rep_idx, 2, '0'))")
    truth_dir = joinpath(rep_dir, "truth")
    data_dir  = joinpath(rep_dir, "data")
    mkpath(truth_dir); mkpath(data_dir)

    rng = MersenneTwister(seed)
    realised, r0 = bp_simulate(v, rng)

    I_seed_arr = aggregate_daily(realised, -(TAU_MAX - 1), 0)
    I_obs      = aggregate_daily(realised,  1, T_DAYS)

    E_c, dates, f_c = expected_reports(I_obs, I_seed_arr, v.delay_cases,  v.alpha_cases,  v.dow_mult)
    E_h, _,     f_h = expected_reports(I_obs, I_seed_arr, v.delay_hosp,   v.alpha_hosp,   v.dow_mult)
    E_d, _,     f_d = expected_reports(I_obs, I_seed_arr, v.delay_deaths, v.alpha_deaths, v.dow_mult)

    cases  = sample_obs(E_c, rng)
    hosp   = sample_obs(E_h, rng)
    deaths = sample_obs(E_d, rng)

    n_days = length(dates)
    Rt_true = [r_at(v.rt_knots, Float64(d)) for d in 1:n_days]

    CSV.write(joinpath(truth_dir, "true_rt.csv"),
        DataFrame(day = 1:n_days, date = dates, R_t = Rt_true))
    CSV.write(joinpath(truth_dir, "true_infections.csv"),
        DataFrame(day = 1:n_days, I = I_obs))
    CSV.write(joinpath(truth_dir, "true_expected.csv"),
        DataFrame(day = 1:n_days, E_cases = E_c,
                  E_hospitalisations = E_h, E_deaths = E_d))

    params_dict = Dict(
        "variant"                  => v.name,
        "replicate_index"          => rep_idx,
        "seed"                     => seed,
        "T_days"                   => T_DAYS,
        "tau_max"                  => TAU_MAX,
        "D_max"                    => D_MAX,
        "start_date"               => string(START_DATE),
        "rt_knots"                 => Dict("days"   => v.rt_knots.days,
                                           "values" => v.rt_knots.values),
        "rt_target_definition"     => "(c) parameter R(d) of the renewal equation; renewal equation is the expectation of the underlying age-dependent branching process (Mishra et al. 2020); see Funk, Abbott & Bracher 2022 for the multiple-Rt-definitions issue",
        "infection_model"          => "individual-level Lloyd-Smith Bellman-Harris branching process: ν_i ~ Gamma(k, R(t_i)/k), Z_i ~ Poisson(ν_i), τ_ij ~ g(τ); k → ∞ recovers Poisson offspring",
        "observation_model"        => "X_d ~ Poisson(α(d) · w_dow(d) · Σ_e f_e · I_{d-e})  (delay PMF is double-interval-censored)",
        "discretisation_note"      => "GI is continuous (per-individual draws); only the delay PMF for the observation convolution is daily-discretised by double interval censoring",
        "gi"                       => dist_spec(v.gi),
        "delay_cases"              => dist_spec(v.delay_cases),
        "delay_hospitalisations"   => dist_spec(v.delay_hosp),
        "delay_deaths"             => dist_spec(v.delay_deaths),
        "delay_pmf_cases"          => f_c,
        "delay_pmf_hospitalisations" => f_h,
        "delay_pmf_deaths"         => f_d,
        "dow_multiplier_mon_to_sun"=> v.dow_mult,
        "k_offspring_dispersion"   => v.k,
        "i_ref"                    => I_REF,
        "euler_lotka_r0"           => r0,
        "n_total_individuals"      => length(realised),
    )
    open(joinpath(truth_dir, "params.json"), "w") do io
        JSON3.pretty(io, params_dict)
    end

    cp(script_path, joinpath(truth_dir, "sim_script.jl"); force = true)

    CSV.write(joinpath(data_dir, "cases.csv"),
        DataFrame(date = dates, cases = cases))
    CSV.write(joinpath(data_dir, "hospitalisations.csv"),
        DataFrame(date = dates, hospitalisations = hosp))
    CSV.write(joinpath(data_dir, "deaths.csv"),
        DataFrame(date = dates, deaths = deaths))

    return nothing
end

function main(args)
    variant_filter = length(args) >= 1 ? args[1] : nothing
    seed_filter    = length(args) >= 2 ? parse(Int, args[2]) : nothing
    script_path    = @__FILE__

    for v in all_variants()
        variant_filter === nothing || v.name == variant_filter || continue
        for (i, seed) in enumerate(REPLICATES)
            seed_filter === nothing || seed == seed_filter || continue
            println("Generating $(v.name) rep_$(lpad(i, 2, '0')) (seed $seed)")
            write_replicate(v, i, seed, script_path)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
