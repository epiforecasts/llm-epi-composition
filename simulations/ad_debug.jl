# Minimal debug: does NUTS+ForwardDiff work at all in this env?
# Then: does EpiAware NUTSampler preserve our adtype?

using Turing
using Distributions
using ADTypes
using ForwardDiff

println("== Test 1: Minimal Turing NUTS + ForwardDiff ==")
@model function toy()
    x ~ Normal(0, 1)
end
chain = sample(toy(), NUTS(0.65; adtype = AutoForwardDiff()), 100; progress = false)
println("OK Turing NUTS+ForwardDiff: ", typeof(chain).name.name)

println("\n== Test 2: Inspect EpiAware NUTSampler adtype field ==")
using EpiAware
sampler = NUTSampler(adtype = AutoForwardDiff(), ndraws = 100, nchains = 1)
println("sampler type: ", typeof(sampler))
println("sampler.adtype = ", sampler.adtype)

println("\n== Test 3: Build EpiMethod with empty pre_sampler_steps ==")
em = EpiMethod(pre_sampler_steps = AbstractEpiOptMethod[], sampler = sampler)
println("EpiMethod type: ", typeof(em))
println("em.sampler.adtype = ", em.sampler.adtype)

println("\n== Test 4: Build a simple EpiAware model and inspect what NUTS gets ==")
gi_pmf    = [0.0, 0.05, 0.15, 0.25, 0.25, 0.15, 0.10, 0.05]
delay_pmf = [0.0, 0.05, 0.15, 0.25, 0.25, 0.15, 0.10, 0.05]
md = EpiData(gi_pmf, exp)
epm = Renewal(md; initialisation_prior = Normal(log(100.0), 2.0))
lat = AR(damp_priors = [truncated(Normal(0.8, 0.1), 0, 1)],
         init_priors = [Normal(0.0, 0.5)],
         ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.1)))
obs = LatentDelay(NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1)), delay_pmf)
ep  = EpiProblem(epi_model = epm, latent_model = lat, observation_model = obs, tspan = (1, 30))
fake_cases = rand(50:200, 30)
mdl = generate_epiaware(ep, (y_t = fake_cases,))
println("model OK")

println("\n== Test 5: Direct Turing.sample with our NUTS ==")
nuts = Turing.NUTS(0.65; adtype = AutoForwardDiff())
println("NUTS type: ", typeof(nuts))
chain2 = sample(mdl, nuts, 50; progress = false)
println("OK direct Turing.sample: ", typeof(chain2).name.name)
