#!/usr/bin/env julia
# Ground-truth simulation generator for the LLM composition study.
#
# Daily-resolution stochastic Poisson renewal:
#   I_d ~ Poisson(R(d) * Σ_s g_s · I_{d-s})
#   E[C_d^stream] = α_stream(d) · w_dow(d) · Σ_e f_e^stream · I_{d-e}
#   C_d^stream ~ NegBin(E[C_d^stream], φ_stream)
#
# GI and per-stream delay daily PMFs computed by double interval censoring
# (CDF-difference of [d-1+p, d+p] integrated over p ~ U(0,1) via quadrature).
#
# The recovery target is R(d) — the parameter (definition (c) in Funk, Abbott
# & Bracher 2022, J R Stat Soc A).
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

# ---- Fixed configuration ----

const TAU_MAX           = 20         # GI truncation (days)
const D_MAX             = 30         # delay truncation (days; longer for deaths)
const T_DAYS            = 150
const START_DATE        = Date(2023, 1, 1)
const REPLICATES        = collect(101:120)
const DOW_MULT_CANON    = [1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5]   # Mon..Sun
const I_REF             = 100.0       # seed reference incidence at d = 0

# ---- Distribution parameterisations ----

gamma_from_mean_sd(m, s)     = Gamma(m^2 / s^2, s^2 / m)
function lognormal_from_mean_sd(m, s)
    σ² = log(1 + s^2 / m^2)
    LogNormal(log(m) - σ² / 2, sqrt(σ²))
end

# ---- Daily PMF via double interval censoring ----
# P(D = d) = ∫_0^1 [F(d + 1 - p) - F(d - p)] dp
# For d = 0 the lower CDF clips at 0; non-negative continuous distributions
# already vanish for negative arguments.

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

# ---- Euler–Lotka growth rate (matches discrete-day Poisson renewal mean) ----

function euler_lotka_r(R0, g; tol = 1e-12, maxiter = 200)
    f(r) = R0 * sum(g[s] * exp(-r * (s - 1)) for s in 1:length(g)) - 1
    lo, hi = -5.0, 5.0
    flo, fhi = f(lo), f(hi)
    (flo * fhi < 0) || error("euler_lotka: bracket not found for R0=$R0")
    for _ in 1:maxiter
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if sign(fm) == sign(flo)
            lo, flo = mid, fm
        else
            hi, fhi = mid, fm
        end
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
    k::Float64                        # offspring dispersion (Lloyd-Smith); k → ∞ is Poisson
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

    K_CANON = 1.0     # moderate offspring dispersion (within Lloyd-Smith range)

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

    high_disp = Variant("high_dispersion", gi_c, del_c, del_h, del_dd,
        ALPHA_CASES_CANON, ALPHA_HOSP_CANON, ALPHA_DEATHS_CANON,
        DOW_MULT_CANON, 0.1, RT_KNOTS_CANON)

    abrupt_knots = RtKnots([1.0, 50.0, 74.0, 77.0, 150.0], [0.8, 1.5, 1.5, 0.5, 0.5])
    abrupt = Variant("abrupt_change", gi_c, del_c, del_h, del_dd,
        ALPHA_CASES_CANON, ALPHA_HOSP_CANON, ALPHA_DEATHS_CANON,
        DOW_MULT_CANON, K_CANON, abrupt_knots)

    return [canon, short_gi, long_delay, strong_dow, high_asc_var, low_disp, high_disp, abrupt]
end

# ---- Forward simulation (daily) ----

function simulate_infections(v::Variant, rng::AbstractRNG)
    g = double_censored_pmf(v.gi, TAU_MAX)
    n_g = length(g)

    r0 = euler_lotka_r(r_at(v.rt_knots, 1.0), g)

    # Pre-obs seed: deterministic float exponential profile, indexed -19..0
    n_seed = TAU_MAX
    I_seed = [I_REF * exp(r0 * d) for d in (-(n_seed - 1)):0]   # length 20

    I = zeros(Float64, T_DAYS)
    for d in 1:T_DAYS
        Rd = r_at(v.rt_knots, Float64(d))
        Λ = 0.0
        @inbounds for s in 1:n_g
            past = d - s
            past_val = past >= 1 ? I[past] : I_seed[end + past]
            Λ += g[s] * past_val
        end
        μ = max(Rd * Λ, 0.0)
        if !isfinite(v.k) || μ == 0
            I[d] = Float64(rand(rng, Poisson(μ)))
        else
            # NegBinL: I_d ~ NegBin(μ_d = R Λ, k Λ), with Var = μ(1 + R/k)
            # Distributions.jl NegativeBinomial(r, p): mean = r(1-p)/p
            r_param = v.k * Λ
            p_param = v.k / (Rd + v.k)
            I[d] = Float64(rand(rng, NegativeBinomial(r_param, p_param)))
        end
    end
    return I, I_seed, g, r0
end

function expected_reports(I::Vector{Float64}, I_seed::Vector{Float64},
                          delay_dist, alpha_fn, dow_mult)
    f = double_censored_pmf(delay_dist, D_MAX)
    n_f = length(f)
    dates = [START_DATE + Day(d - 1) for d in 1:T_DAYS]
    E = zeros(Float64, T_DAYS)
    for d in 1:T_DAYS
        s = 0.0
        @inbounds for e in 1:n_f
            past = d - (e - 1)        # delay e-1 means same-day for e=1
            past_val = past >= 1 ? I[past] : (past >= -(length(I_seed) - 1) ? I_seed[end + past] : 0.0)
            s += f[e] * past_val
        end
        α_d = alpha_fn(Float64(d))
        w_d = dow_mult[dayofweek(dates[d])]
        E[d] = α_d * w_d * s
    end
    return E, dates, f
end

function sample_obs(E, rng)
    # Observation noise is Poisson: cases inherit overdispersion from the
    # NegBinL-distributed infections; no free observation-level dispersion knob.
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
    I, I_seed, g, r0 = simulate_infections(v, rng)
    E_c, dates, f_c = expected_reports(I, I_seed, v.delay_cases,  v.alpha_cases,  v.dow_mult)
    E_h, _,     f_h = expected_reports(I, I_seed, v.delay_hosp,   v.alpha_hosp,   v.dow_mult)
    E_d, _,     f_d = expected_reports(I, I_seed, v.delay_deaths, v.alpha_deaths, v.dow_mult)

    cases  = sample_obs(E_c, rng)
    hosp   = sample_obs(E_h, rng)
    deaths = sample_obs(E_d, rng)

    n_days = length(dates)
    Rt_true = [r_at(v.rt_knots, Float64(d)) for d in 1:n_days]

    CSV.write(joinpath(truth_dir, "true_rt.csv"),
        DataFrame(day = 1:n_days, date = dates, R_t = Rt_true))
    CSV.write(joinpath(truth_dir, "true_infections.csv"),
        DataFrame(day = 1:n_days, I = Int.(I)))
    CSV.write(joinpath(truth_dir, "true_expected.csv"),
        DataFrame(day = 1:n_days,
                  E_cases = E_c, E_hospitalisations = E_h, E_deaths = E_d))

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
        "rt_target_definition"     => "(c) parameter R(d) of the daily renewal model (Funk, Abbott & Bracher 2022); the renewal equation is the expectation of an age-dependent branching process (Mishra et al. 2020), regardless of offspring distribution",
        "infection_model"          => "I_d ~ NegBin(μ = R(d) Σ_s g_s I_{d-s}, k Σ_s g_s I_{d-s}); k → ∞ recovers Poisson; phenomenological NegBinL marginal motivated by — but not strictly derived from — individual offspring heterogeneity (Lloyd-Smith et al. 2005)",
        "observation_model"        => "X_d ~ Poisson(α(d) · w_dow(d) · Σ_e f_e · I_{d-e})",
        "discretisation"           => "double interval censoring (P(D=d) = ∫_0^1 [F(d+1-p) - F(d-p)] dp)",
        "gi"                       => dist_spec(v.gi),
        "delay_cases"              => dist_spec(v.delay_cases),
        "delay_hospitalisations"   => dist_spec(v.delay_hosp),
        "delay_deaths"             => dist_spec(v.delay_deaths),
        "gi_pmf"                   => g,
        "delay_pmf_cases"          => f_c,
        "delay_pmf_hospitalisations" => f_h,
        "delay_pmf_deaths"         => f_d,
        "dow_multiplier_mon_to_sun"=> v.dow_mult,
        "k_offspring_dispersion"   => v.k,
        "i_ref"                    => I_REF,
        "euler_lotka_r0"           => r0,
        "seed_pre_obs"             => I_seed,
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
