
cd("/home/eidesfun/code/llm-epi-composition/evaluation/results/scenario_1a/epiaware/llama3.1:8b/run_02")

# Load required packages
using EpiAware
using Distributions
using Turing

# Load data
cases = readdlm("cases.csv", ',', String, '\n')
date = [parse(Date, d[1:10]) for d in cases[:, 1]]
cases = map(x -> parse(Int64, x), cases[:, 2])

# Define EpiData
model_data = EpiData(
    gen_distribution = Gamma(6.5, 0.62),
    initial_cases = Normal(log(100.0), 1.0)
)

# Create infection model (Renewal) and latent model for log(Rt)
epi = Renewal(model_data; initialisation_prior = Normal(log(100.0), 1.0))
latent = AR(
    damp_priors = [truncated(Normal(0.8, 0.05), 0, 1)],
    init_priors = [Normal(0.0, 0.5)],
    ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.1))
)

# Create observation model with delay
obs = LatentDelay(
    NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1)),
    Gamma(5.0, 1.0)
)

# Compose into EpiProblem
epi_prob = EpiProblem(
    epi_model = epi,
    latent_model = latent,
    observation_model = obs,
    tspan = (date[1], date[end])
)

# Run inference
mdl = generate_epiaware(epi_prob, (y_t = cases,))
inference_method = EpiMethod(
    pre_sampler_steps = [ManyPathfinder(nruns = 4, maxiters = 100)],
    sampler = NUTSampler(ndraws = 1000, nchains = 4)
)

# Run the inference
results = apply_method(mdl, inference_method, (y_t = cases,))

# Extract Rt estimates
Rt = exp.(results[latent_model].Z_t)

# Plot Rt over time
using Plots
plot(date, Rt, xlabel="Date", ylabel="Rt")

