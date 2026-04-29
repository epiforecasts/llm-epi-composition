#!/usr/bin/env julia
# Calibration of the reference EpiAware solution on canonical replicates.
#
# Reports per-replicate Rt RMSE, 90%-credible-interval coverage, and width
# on the evaluation window (days 25..125). Median targets per the plan's
# §"Sanity check before running any LLM condition":
#   - median RMSE < 0.10
#   - median coverage in [0.80, 0.95]
#   - calibration error |median(coverage) - 0.90| < 0.10
#
# The AR(1) std prior is the calibration knob. Default 0.05; pass via
# AR_STD_PRIOR env var to retune.
#
# Usage:
#   julia --project=evaluation/julia_env evaluation/calibrate_reference.jl
#   AR_STD_PRIOR=0.025 N_REPS=20 OUT_FILE=simulations/calib_0p025.csv \
#     julia --project=evaluation/julia_env evaluation/calibrate_reference.jl

using EpiAware, Distributions, Turing, ReverseDiff, LogDensityProblemsAD
using CSV, DataFrames, Random, Statistics, Logging
Logging.disable_logging(Logging.Warn)

const AR_STD_PRIOR = parse(Float64, get(ENV, "AR_STD_PRIOR", "0.05"))
const N_REPS       = parse(Int,     get(ENV, "N_REPS",       "20"))
const OUT_FILE     = get(ENV, "OUT_FILE", "simulations/calib_results.csv")
const EVAL_WIN_LO  = parse(Int, get(ENV, "EVAL_WIN_LO", "25"))
const EVAL_WIN_HI  = parse(Int, get(ENV, "EVAL_WIN_HI", "125"))

println("AR std prior = HalfNormal($AR_STD_PRIOR)   N reps = $N_REPS")
println("Eval window: days $EVAL_WIN_LO..$EVAL_WIN_HI")
println("Output: $OUT_FILE")
println()

function run_rep(rep_idx, ar_std)
    rep_dir = "simulations/canonical/rep_$(lpad(rep_idx, 2, '0'))"
    cases = CSV.read(joinpath(rep_dir, "data", "cases.csv"), DataFrame).cases
    truth = CSV.read(joinpath(rep_dir, "truth", "true_rt.csv"), DataFrame)

    gi_dist    = Gamma(5.5^2 / 2.0^2, 2.0^2 / 5.5)
    σ²_d       = log(1 + 2.0^2 / 5.0^2)
    delay_dist = LogNormal(log(5.0) - σ²_d / 2, sqrt(σ²_d))
    gi_pmf    = censored_pmf(gi_dist;    Δd = 1.0, D = 21.0)
    delay_pmf = censored_pmf(delay_dist; Δd = 1.0, D = 31.0)
    gi_pmf    = gi_pmf[2:end]; gi_pmf ./= sum(gi_pmf)

    Random.seed!(42)
    epi_data     = EpiData(gi_pmf, exp)
    epi_model    = Renewal(epi_data; initialisation_prior = Normal(log(100.0), 2.0))
    latent_model = AR(
        damp_priors = [truncated(Normal(0.8, 0.1), 0, 1)],
        init_priors = [Normal(0.0, 0.5)],
        ϵ_t         = HierarchicalNormal(std_prior = HalfNormal(ar_std)),
    )
    obs_model = LatentDelay(NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1)), delay_pmf)
    epi_prob = EpiProblem(epi_model = epi_model, latent_model = latent_model,
                          observation_model = obs_model, tspan = (1, length(cases)))

    mdl = generate_epiaware(epi_prob, (y_t = cases,))
    nuts = Turing.NUTS(0.8; adtype = AutoReverseDiff(compile = false))
    chain = sample(mdl, nuts, MCMCThreads(), 500, 2; progress = false)

    gen_q = generated_quantities(
        generate_epiaware(epi_prob, (y_t = fill(missing, length(cases)),)), chain)
    Z_samples  = mapreduce(hcat, gen_q) do g; g.Z_t; end
    Rt_samples = exp.(Z_samples)
    Rt_lo  = vec(mapslices(x -> quantile(x, 0.05), Rt_samples; dims = 2))
    Rt_med = vec(mapslices(x -> quantile(x, 0.50), Rt_samples; dims = 2))
    Rt_hi  = vec(mapslices(x -> quantile(x, 0.95), Rt_samples; dims = 2))

    win = EVAL_WIN_LO:EVAL_WIN_HI
    rt_true = truth.R_t
    rmse     = sqrt(mean((Rt_med[win] .- rt_true[win]) .^ 2))
    coverage = mean((Rt_lo[win] .<= rt_true[win]) .& (Rt_hi[win] .>= rt_true[win]))
    width    = mean(Rt_hi[win] .- Rt_lo[win])
    return rmse, coverage, width
end

results = DataFrame(rep = Int[], rmse = Float64[], coverage = Float64[], width = Float64[])
for rep in 1:N_REPS
    rmse, cov, w = run_rep(rep, AR_STD_PRIOR)
    push!(results, (rep, rmse, cov, w))
    println("rep $rep: rmse=$(round(rmse, digits=4))  cov=$(round(cov, digits=3))  width=$(round(w, digits=3))")
    CSV.write(OUT_FILE, results)
end

println()
println("=== Summary across $N_REPS reps (AR std HalfNormal($AR_STD_PRIOR)) ===")
println("median RMSE:     ", round(median(results.rmse), digits=4),  "  (target < 0.10)")
println("median coverage: ", round(median(results.coverage), digits=3), "  (target [0.80, 0.95])")
println("median width:    ", round(median(results.width), digits=3))
println("calibration err: ", round(abs(median(results.coverage) - 0.9), digits=3),
        "  (target < 0.10)")
