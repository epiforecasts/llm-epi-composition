# Test whether `using EpiAware` alone is enough to activate the
# LogDensityProblemsADReverseDiffExt extension, i.e. whether LLM-generated
# code that only does `using EpiAware` will work with AutoReverseDiff.

using EpiAware
using Distributions
using Turing
using Random

Random.seed!(42)
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

# Minimal LLM-style code: no `using ReverseDiff` import, defaults
method = EpiMethod(
    pre_sampler_steps = [ManyPathfinder(nruns = 2, maxiters = 50)],
    sampler           = NUTSampler(
        adtype         = AutoReverseDiff(compile = true),
        ndraws         = 100,
        nchains        = 2,
        mcmc_parallel  = MCMCThreads(),
    ),
)
println("Testing apply_method with only `using EpiAware` + Turing + Distributions ...")
try
    r = apply_method(mdl, method, (y_t = fake_cases,))
    println("OK — works without explicit `using ReverseDiff`")
catch e
    println("FAILED: ", typeof(e))
    println(first(sprint(showerror, e), 400))
end
