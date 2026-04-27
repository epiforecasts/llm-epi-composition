# Diagnose whether EpiAware's apply_method genuinely fails in this env on a
# standard reference-style call, or whether it was incidental to my setup.

using EpiAware
using Distributions
using Turing
using ADTypes
using ReverseDiff
using LogDensityProblemsAD
using Random
using Logging

Logging.disable_logging(Logging.Warn)
Random.seed!(42)

println("== Versions ==")
using Pkg
for p in ["EpiAware", "Turing", "ReverseDiff", "LogDensityProblemsAD", "ADTypes", "Pathfinder"]
    info = Pkg.project().dependencies
    if haskey(info, p)
        println("$p: ", info[p])
    end
end

# Build a minimal EpiAware model
gi_pmf    = [0.0, 0.05, 0.15, 0.25, 0.25, 0.15, 0.10, 0.05]
delay_pmf = [0.0, 0.05, 0.15, 0.25, 0.25, 0.15, 0.10, 0.05]
md  = EpiData(gi_pmf, exp)
epm = Renewal(md; initialisation_prior = Normal(log(100.0), 2.0))
lat = AR(damp_priors = [truncated(Normal(0.8, 0.1), 0, 1)],
         init_priors = [Normal(0.0, 0.5)],
         ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.1)))
obs = LatentDelay(NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1)), delay_pmf)
ep  = EpiProblem(epi_model = epm, latent_model = lat, observation_model = obs, tspan = (1, 30))
fake_cases = rand(50:200, 30)
mdl = generate_epiaware(ep, (y_t = fake_cases,))

println("\n== Test A: apply_method with AutoReverseDiff(compile=true), MCMCSerial ==")
try
    method_A = EpiMethod(
        pre_sampler_steps = AbstractEpiOptMethod[],
        sampler           = NUTSampler(
            adtype         = AutoReverseDiff(compile = true),
            ndraws         = 100,
            nchains        = 1,
            mcmc_parallel  = MCMCSerial(),
        ),
    )
    r = apply_method(mdl, method_A, (y_t = fake_cases,))
    println("OK — apply_method works with ReverseDiff(compile=true), MCMCSerial")
catch e
    println("FAILED: ", typeof(e), " — ", first(sprint(showerror, e), 200))
end

println("\n== Test B: apply_method with AutoReverseDiff(compile=false), MCMCSerial ==")
try
    method_B = EpiMethod(
        pre_sampler_steps = AbstractEpiOptMethod[],
        sampler           = NUTSampler(
            adtype         = AutoReverseDiff(compile = false),
            ndraws         = 100,
            nchains        = 1,
            mcmc_parallel  = MCMCSerial(),
        ),
    )
    r = apply_method(mdl, method_B, (y_t = fake_cases,))
    println("OK — apply_method works with ReverseDiff(compile=false), MCMCSerial")
catch e
    println("FAILED: ", typeof(e), " — ", first(sprint(showerror, e), 200))
end

println("\n== Test C: apply_method with AutoForwardDiff, MCMCSerial ==")
try
    using ForwardDiff
    method_C = EpiMethod(
        pre_sampler_steps = AbstractEpiOptMethod[],
        sampler           = NUTSampler(
            adtype         = AutoForwardDiff(),
            ndraws         = 100,
            nchains        = 1,
            mcmc_parallel  = MCMCSerial(),
        ),
    )
    r = apply_method(mdl, method_C, (y_t = fake_cases,))
    println("OK — apply_method works with ForwardDiff, MCMCSerial")
catch e
    println("FAILED: ", typeof(e), " — ", first(sprint(showerror, e), 200))
end

println("\n== Test D: with ManyPathfinder pre-step, AutoReverseDiff(compile=true), MCMCThreads ==")
try
    method_D = EpiMethod(
        pre_sampler_steps = [ManyPathfinder(nruns = 2, maxiters = 50)],
        sampler           = NUTSampler(
            adtype         = AutoReverseDiff(compile = true),
            ndraws         = 100,
            nchains        = 2,
            mcmc_parallel  = MCMCThreads(),
        ),
    )
    r = apply_method(mdl, method_D, (y_t = fake_cases,))
    println("OK — full reference-style apply_method works")
catch e
    println("FAILED: ", typeof(e), " — ", first(sprint(showerror, e), 250))
end
