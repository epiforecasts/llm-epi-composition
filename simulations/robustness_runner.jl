#!/usr/bin/env julia
# Robustness check: deterministic EpiEstim-style Rt estimator on canonical from
# both GTs (individual-level BP vs moment-closed population NegBinL). Compares
# point-estimate RMSE and 90%-credible-interval coverage of the true R(d).
#
# Estimator: For each day d, conjugate Gamma(α₀, β₀) prior on R, Poisson
# likelihood. Posterior R_d | C_window ~ Gamma(α₀ + ΣC_w, β₀ + ΣΛ_w) where
# Λ_d = Σ_τ g_τ · C_{d-τ} and the sliding window covers days d-w+1..d.
# This deconvolves the delay implicitly via the same window over reported
# cases (a deliberate simplification for the comparison).
#
# Usage: julia --project=simulations simulations/robustness_runner.jl

using Distributions
using QuadGK
using CSV
using DataFrames
using Statistics
using Printf

const WINDOW   = 7
const PRIOR_α  = 1.0
const PRIOR_β  = 5.0
const N_REP    = 3

function censored_pmf_local(dist, dmax)
    pmf = Float64[]
    F0 = 0.0
    for d in 0:dmax
        F1, _ = quadgk(p -> cdf(dist, d + 1 - p) - cdf(dist, max(d - p, 0.0)), 0.0, 1.0; rtol = 1e-8)
        push!(pmf, F1)
        F0 = F1
    end
    pmf ./ sum(pmf)
end

function lambda_t(C, g)
    n = length(C)
    n_g = length(g)
    Λ = zeros(n)
    for d in 1:n
        s = 0.0
        for τ in 1:min(d - 1, n_g)
            s += g[τ] * C[d - τ]
        end
        Λ[d] = s
    end
    Λ
end

function epiestim_rt(C, g; window = WINDOW, α₀ = PRIOR_α, β₀ = PRIOR_β)
    n = length(C)
    Λ = lambda_t(C, g)
    R_lo = fill(NaN, n); R_med = fill(NaN, n); R_hi = fill(NaN, n)
    for d in window:n
        win = (d - window + 1):d
        sum_C = sum(C[win])
        sum_Λ = sum(Λ[win])
        sum_Λ > 0 || continue
        post = Gamma(α₀ + sum_C, 1.0 / (β₀ + sum_Λ))
        R_lo[d]  = quantile(post, 0.05)
        R_med[d] = quantile(post, 0.50)
        R_hi[d]  = quantile(post, 0.95)
    end
    R_lo, R_med, R_hi
end

function score(rep_dir, g)
    cases = CSV.read(joinpath(rep_dir, "data", "cases.csv"), DataFrame).cases
    truth = CSV.read(joinpath(rep_dir, "truth", "true_rt.csv"), DataFrame)
    R_lo, R_med, R_hi = epiestim_rt(cases, g)
    n   = length(cases)
    win = max(WINDOW + 7, 8):(n - 7)
    rt_true  = truth.R_t[win]
    rmse     = sqrt(mean((R_med[win] .- rt_true) .^ 2))
    coverage = mean((R_lo[win] .<= rt_true) .& (R_hi[win] .>= rt_true))
    width    = mean(R_hi[win] .- R_lo[win])
    rmse, coverage, width, R_lo, R_med, R_hi, win
end

function main()
    gi_dist = Gamma(5.5^2 / 2.0^2, 2.0^2 / 5.5)
    g_full  = censored_pmf_local(gi_dist, 20)
    g       = g_full[2:end] ./ sum(g_full[2:end])

    rows = DataFrame(gt = String[], rep = Int[], rmse = Float64[], coverage = Float64[], width = Float64[])
    @printf "\n%-6s %4s %8s %10s %8s\n" "GT" "rep" "RMSE" "coverage" "width"
    @printf "%s\n" repeat("-", 42)
    for (gt_label, gt_root) in (("BP", "simulations"), ("popn", "simulations/_popn_check"))
        for rep in 1:N_REP
            rep_dir = joinpath(gt_root, "canonical", "rep_$(lpad(rep, 2, '0'))")
            isdir(rep_dir) || (println("missing: $rep_dir"); continue)
            rmse, cov, w, _, _, _, _ = score(rep_dir, g)
            push!(rows, (gt_label, rep, rmse, cov, w))
            @printf "%-6s %4d %8.4f %10.3f %8.3f\n" gt_label rep rmse cov w
        end
    end
    @printf "\n%-6s %8s %10s %8s\n" "GT" "med RMSE" "med cov" "med width"
    @printf "%s\n" repeat("-", 38)
    for gt_label in ("BP", "popn")
        sub = rows[rows.gt .== gt_label, :]
        @printf "%-6s %8.4f %10.3f %8.3f\n" gt_label median(sub.rmse) median(sub.coverage) median(sub.width)
    end
    println()
    CSV.write("simulations/robustness_results.csv", rows)
    println("Saved to simulations/robustness_results.csv")
end

main()
