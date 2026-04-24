#!/usr/bin/env julia
# Ground-truth simulation generator for the LLM composition study.
#
# Solves the continuous-time renewal integral equation on a fine sub-day grid,
# aggregates to daily incidence and per-stream expected reports, and samples
# NegBin-distributed observations. No GI or delay PMF is exposed; estimator-side
# discretisation is tested against the continuous-process truth.
#
# Usage:
#   julia --project=simulations simulations/generate.jl                  # all variants × reps
#   julia --project=simulations simulations/generate.jl canonical        # one variant, all reps
#   julia --project=simulations simulations/generate.jl canonical 101    # one variant, one seed
#   DT_OVERRIDE=0.025 julia ... simulations/generate.jl                  # convergence check

using Distributions
using Random
using CSV
using DataFrames
using JSON3
using Dates

# ---- Fixed configuration ----

const DT_DEFAULT        = 0.05
const TAU_MAX           = 20.0
const D_MAX             = 20.0
const T_DAYS            = 150
const START_DATE        = Date(2023, 1, 1)
const REPLICATES        = [101, 102, 103]
const DOW_MULT_CANON    = [1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5]  # Mon..Sun
const I_REF             = 100.0   # reference infection rate at t=0 (end of seed, start of obs)

get_dt() = parse(Float64, get(ENV, "DT_OVERRIDE", string(DT_DEFAULT)))

# ---- Distribution parameterisations ----

gamma_from_mean_sd(m, s)     = Gamma(m^2 / s^2, s^2 / m)
function lognormal_from_mean_sd(m, s)
    σ² = log(1 + s^2 / m^2)
    LogNormal(log(m) - σ² / 2, sqrt(σ²))
end

# ---- Sub-day convolution weights (CDF-exact) ----

function sub_day_weights(dist, tmax, dt)
    n = Int(round(tmax / dt))
    w = Vector{Float64}(undef, n)
    F0 = 0.0
    for s in 0:(n - 1)
        F1 = cdf(dist, (s + 1) * dt)
        w[s + 1] = F1 - F0
        F0 = F1
    end
    w ./= sum(w)
    return w
end

# ---- Euler–Lotka exponential growth rate for seed profile ----
# Solves 1 = R0 · Σ_s w_s · exp(-r · s · dt) for r, matching the left-shifted
# discretised convolution used in the forward step.

function euler_lotka_r(R0, w, dt; tol = 1e-12, maxiter = 200)
    f(r) = R0 * sum(w[s] * exp(-r * s * dt) for s in 1:length(w)) - 1
    lo, hi = -5.0, 5.0
    flo, fhi = f(lo), f(hi)
    (flo * fhi < 0) || error("euler_lotka: bracket not found for R0=$R0 (flo=$flo, fhi=$fhi)")
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
    phi_cases::Float64
    phi_hosp::Float64
    phi_deaths::Float64
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

    canon = Variant("canonical", gi_c, del_c, del_h, del_dd,
        ALPHA_CASES_CANON, ALPHA_HOSP_CANON, ALPHA_DEATHS_CANON,
        DOW_MULT_CANON, 10.0, 10.0, 20.0, RT_KNOTS_CANON)

    short_gi = Variant("short_gi", gamma_from_mean_sd(2.5, 1.0),
        del_c, del_h, del_dd,
        ALPHA_CASES_CANON, ALPHA_HOSP_CANON, ALPHA_DEATHS_CANON,
        DOW_MULT_CANON, 10.0, 10.0, 20.0, RT_KNOTS_CANON)

    long_delay = Variant("long_delay", gi_c,
        lognormal_from_mean_sd(10.0, 3.0), del_h, del_dd,
        ALPHA_CASES_CANON, ALPHA_HOSP_CANON, ALPHA_DEATHS_CANON,
        DOW_MULT_CANON, 10.0, 10.0, 20.0, RT_KNOTS_CANON)

    strong_dow = Variant("strong_dow", gi_c, del_c, del_h, del_dd,
        ALPHA_CASES_CANON, ALPHA_HOSP_CANON, ALPHA_DEATHS_CANON,
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.25, 0.25],
        10.0, 10.0, 20.0, RT_KNOTS_CANON)

    high_asc_var = Variant("high_asc_var", gi_c, del_c, del_h, del_dd,
        t -> 0.40 + 0.35 * sin(2π * t / T_DAYS),
        ALPHA_HOSP_CANON, ALPHA_DEATHS_CANON,
        DOW_MULT_CANON, 10.0, 10.0, 20.0, RT_KNOTS_CANON)

    low_disp = Variant("low_dispersion", gi_c, del_c, del_h, del_dd,
        ALPHA_CASES_CANON, ALPHA_HOSP_CANON, ALPHA_DEATHS_CANON,
        DOW_MULT_CANON, 1000.0, 1000.0, 1000.0, RT_KNOTS_CANON)

    high_disp = Variant("high_dispersion", gi_c, del_c, del_h, del_dd,
        ALPHA_CASES_CANON, ALPHA_HOSP_CANON, ALPHA_DEATHS_CANON,
        DOW_MULT_CANON, 2.0, 2.0, 2.0, RT_KNOTS_CANON)

    abrupt_knots = RtKnots([1.0, 50.0, 74.0, 77.0, 150.0], [0.8, 1.5, 1.5, 0.5, 0.5])
    abrupt = Variant("abrupt_change", gi_c, del_c, del_h, del_dd,
        ALPHA_CASES_CANON, ALPHA_HOSP_CANON, ALPHA_DEATHS_CANON,
        DOW_MULT_CANON, 10.0, 10.0, 20.0, abrupt_knots)

    return [canon, short_gi, long_delay, strong_dow, high_asc_var, low_disp, high_disp, abrupt]
end

# ---- Solver ----

function solve_lambda(v::Variant, dt)
    gi_w = sub_day_weights(v.gi, TAU_MAX, dt)
    spd  = Int(round(1 / dt))
    n_burn = Int(round(TAU_MAX / dt))
    n_obs  = Int(round(T_DAYS / dt))
    N = n_burn + n_obs

    t_center(k) = (k - 0.5) * dt - TAU_MAX

    R_init = r_at(v.rt_knots, 1.0)
    r0 = euler_lotka_r(R_init, gi_w, dt)

    λ0 = I_REF
    λ = Vector{Float64}(undef, N)
    for k in 1:n_burn
        λ[k] = λ0 * exp(r0 * t_center(k))
    end

    n_gi = length(gi_w)
    for k in (n_burn + 1):N
        Rk = r_at(v.rt_knots, t_center(k))
        s = 0.0
        @inbounds for i in 1:n_gi
            kprev = k - i
            kprev >= 1 || break
            s += λ[kprev] * gi_w[i]
        end
        λ[k] = Rk * s
    end

    return (λ = λ, gi_w = gi_w, r0 = r0, λ0 = λ0,
            n_burn = n_burn, n_obs = n_obs, N = N, spd = spd, dt = dt)
end

function daily_infections(sol)
    n_days = Int(sol.n_obs / sol.spd)
    I = zeros(n_days)
    for k_obs in 1:sol.n_obs
        d = Int(ceil(k_obs / sol.spd))
        I[d] += sol.λ[sol.n_burn + k_obs] * sol.dt
    end
    return I
end

function stream_expected(sol, delay_dist, alpha_fn, dow_mult)
    del_w = sub_day_weights(delay_dist, D_MAX, sol.dt)
    n_del = length(del_w)
    r_rate = zeros(sol.n_obs)
    for k_obs in 1:sol.n_obs
        k = sol.n_burn + k_obs
        s = 0.0
        @inbounds for i in 1:n_del
            kprev = k - i
            kprev >= 1 || break
            s += sol.λ[kprev] * del_w[i]
        end
        r_rate[k_obs] = s
    end

    n_days = Int(sol.n_obs / sol.spd)
    E = zeros(n_days)
    for k_obs in 1:sol.n_obs
        d = Int(ceil(k_obs / sol.spd))
        E[d] += r_rate[k_obs] * sol.dt
    end
    dates = [START_DATE + Day(d - 1) for d in 1:n_days]
    for d in 1:n_days
        α_d = alpha_fn(Float64(d))
        w_d = dow_mult[dayofweek(dates[d])]
        E[d] *= α_d * w_d
    end
    return E, dates, del_w
end

function sample_negbin(E, ϕ, rng)
    out = Vector{Int}(undef, length(E))
    for i in eachindex(E)
        μ = max(E[i], 1e-12)
        p = ϕ / (μ + ϕ)
        out[i] = rand(rng, NegativeBinomial(ϕ, p))
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

function write_replicate(v::Variant, rep_idx::Int, seed::Int, script_path::String, dt)
    rep_dir   = joinpath("simulations", v.name, "rep_$(lpad(rep_idx, 2, '0'))")
    truth_dir = joinpath(rep_dir, "truth")
    data_dir  = joinpath(rep_dir, "data")
    mkpath(truth_dir); mkpath(data_dir)

    sol = solve_lambda(v, dt)
    I_daily = daily_infections(sol)
    E_c, dates, _ = stream_expected(sol, v.delay_cases,  v.alpha_cases,  v.dow_mult)
    E_h, _,     _ = stream_expected(sol, v.delay_hosp,   v.alpha_hosp,   v.dow_mult)
    E_d, _,     _ = stream_expected(sol, v.delay_deaths, v.alpha_deaths, v.dow_mult)

    rng = MersenneTwister(seed)
    cases  = sample_negbin(E_c, v.phi_cases, rng)
    hosp   = sample_negbin(E_h, v.phi_hosp,  rng)
    deaths = sample_negbin(E_d, v.phi_deaths, rng)

    n_days = length(dates)
    Rt_true = [r_at(v.rt_knots, Float64(d)) for d in 1:n_days]

    CSV.write(joinpath(truth_dir, "true_rt.csv"),
        DataFrame(day = 1:n_days, date = dates, R_t = Rt_true))
    CSV.write(joinpath(truth_dir, "true_infections.csv"),
        DataFrame(day = 1:n_days, I = I_daily))
    CSV.write(joinpath(truth_dir, "true_expected.csv"),
        DataFrame(day = 1:n_days,
                  E_cases = E_c, E_hospitalisations = E_h, E_deaths = E_d))

    params_dict = Dict(
        "variant"                  => v.name,
        "replicate_index"          => rep_idx,
        "seed"                     => seed,
        "T_days"                   => T_DAYS,
        "dt"                       => dt,
        "tau_max"                  => TAU_MAX,
        "D_max"                    => D_MAX,
        "start_date"               => string(START_DATE),
        "rt_knots"                 => Dict("days"   => v.rt_knots.days,
                                           "values" => v.rt_knots.values),
        "gi"                       => dist_spec(v.gi),
        "delay_cases"              => dist_spec(v.delay_cases),
        "delay_hospitalisations"   => dist_spec(v.delay_hosp),
        "delay_deaths"             => dist_spec(v.delay_deaths),
        "dow_multiplier_mon_to_sun"=> v.dow_mult,
        "phi_cases"                => v.phi_cases,
        "phi_hospitalisations"     => v.phi_hosp,
        "phi_deaths"               => v.phi_deaths,
        "i_ref"                    => I_REF,
        "euler_lotka_r0"           => sol.r0,
        "lambda_0"                 => sol.λ0,
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

# ---- Main ----

function main(args)
    variant_filter = length(args) >= 1 ? args[1] : nothing
    seed_filter    = length(args) >= 2 ? parse(Int, args[2]) : nothing
    dt             = get_dt()
    script_path    = @__FILE__

    for v in all_variants()
        variant_filter === nothing || v.name == variant_filter || continue
        for (i, seed) in enumerate(REPLICATES)
            seed_filter === nothing || seed == seed_filter || continue
            println("Generating $(v.name) rep_$(lpad(i, 2, '0')) (seed $seed, dt $dt)")
            write_replicate(v, i, seed, script_path, dt)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
