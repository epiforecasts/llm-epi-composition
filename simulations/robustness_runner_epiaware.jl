#!/usr/bin/env julia
# Reference EpiAware robustness check: BP vs popn canonical.
# Bypasses EpiMethod / apply_method (which inadvertently selects AutoReverseDiff
# in this env) by calling Turing.sample directly with AutoForwardDiff.
#
# Usage:
#   julia --project=evaluation/julia_env simulations/robustness_runner_epiaware.jl GT_ROOT REP_INDEX OUT_PATH

using EpiAware
using Distributions
using Turing
using ADTypes
using ReverseDiff
using LogDensityProblemsAD
using CSV
using DataFrames
using Random
using Statistics
using Logging

Logging.disable_logging(Logging.Warn)

function run_one(gt_root::String, rep_idx::Int)
    rep_dir = joinpath(gt_root, "canonical", "rep_$(lpad(rep_idx, 2, '0'))")
    cases   = CSV.read(joinpath(rep_dir, "data", "cases.csv"), DataFrame).cases
    truth   = CSV.read(joinpath(rep_dir, "truth", "true_rt.csv"), DataFrame)

    gi_dist    = Gamma(5.5^2 / 2.0^2, 2.0^2 / 5.5)
    σ²_d       = log(1 + 2.0^2 / 5.0^2)
    delay_dist = LogNormal(log(5.0) - σ²_d / 2, sqrt(σ²_d))

    gi_pmf    = censored_pmf(gi_dist;    Δd = 1.0, D = 21.0)
    delay_pmf = censored_pmf(delay_dist; Δd = 1.0, D = 31.0)
    gi_pmf    = gi_pmf[2:end]
    gi_pmf  ./= sum(gi_pmf)

    Random.seed!(42)
    epi_data     = EpiData(gi_pmf, exp)
    epi_model    = Renewal(epi_data; initialisation_prior = Normal(log(100.0), 2.0))
    latent_model = AR(
        damp_priors = [truncated(Normal(0.8, 0.1), 0, 1)],
        init_priors = [Normal(0.0, 0.5)],
        ϵ_t         = HierarchicalNormal(std_prior = HalfNormal(0.1)),
    )
    obs_model = LatentDelay(
        NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1)),
        delay_pmf,
    )
    epi_prob = EpiProblem(
        epi_model         = epi_model,
        latent_model      = latent_model,
        observation_model = obs_model,
        tspan             = (1, length(cases)),
    )

    mdl = generate_epiaware(epi_prob, (y_t = cases,))
    nuts = Turing.NUTS(0.8; adtype = AutoReverseDiff(compile = false))
    chain = sample(mdl, nuts, MCMCThreads(), 500, 2; progress = false)

    gen_q = generated_quantities(
        generate_epiaware(epi_prob, (y_t = fill(missing, length(cases)),)),
        chain,
    )
    Z_samples  = mapreduce(hcat, gen_q) do g; g.Z_t; end
    Rt_samples = exp.(Z_samples)

    Rt_lo  = vec(mapslices(x -> quantile(x, 0.05), Rt_samples; dims = 2))
    Rt_med = vec(mapslices(x -> quantile(x, 0.50), Rt_samples; dims = 2))
    Rt_hi  = vec(mapslices(x -> quantile(x, 0.95), Rt_samples; dims = 2))

    n   = length(cases)
    win = 8:(n - 7)
    rt_true = truth.R_t
    rmse     = sqrt(mean((Rt_med[win] .- rt_true[win]) .^ 2))
    coverage = mean((Rt_lo[win] .<= rt_true[win]) .& (Rt_hi[win] .>= rt_true[win]))
    width    = mean(Rt_hi[win] .- Rt_lo[win])

    (; rmse, coverage, width, Rt_lo, Rt_med, Rt_hi, rt_true)
end

function main()
    gt_root  = ARGS[1]
    rep_idx  = parse(Int, ARGS[2])
    out_path = ARGS[3]

    println("Running EpiAware on $gt_root rep_$rep_idx ...")
    r = run_one(gt_root, rep_idx)

    println("RESULT  rmse=$(round(r.rmse, digits=4))  cov=$(round(r.coverage, digits=3))  width=$(round(r.width, digits=3))")

    n = length(r.Rt_med)
    df = DataFrame(
        gt = fill(basename(gt_root), n),
        rep = fill(rep_idx, n),
        day = 1:n,
        Rt_lo = r.Rt_lo,
        Rt_med = r.Rt_med,
        Rt_hi = r.Rt_hi,
        rt_true = r.rt_true,
    )
    if isfile(out_path)
        existing = CSV.read(out_path, DataFrame)
        df = vcat(existing, df)
    end
    CSV.write(out_path, df)
    println("Saved to $out_path")
end

main()
