# Quick GT comparison: BP vs popn-NegBinL on the canonical DGP.
# Prints per-day mean and SD of infections and cases, plus peak distributions.

using CSV, DataFrames, Statistics

function load_reps(root, n_rep)
    rows = DataFrame(rep=Int[], day=Int[], I=Int[], cases=Int[])
    for r in 1:n_rep
        path = joinpath(root, "canonical", "rep_$(lpad(r, 2, '0'))")
        isdir(path) || continue
        infs = CSV.read(joinpath(path, "truth", "true_infections.csv"), DataFrame)
        ca   = CSV.read(joinpath(path, "data",  "cases.csv"), DataFrame)
        for d in 1:nrow(infs)
            push!(rows, (r, infs.day[d], infs.I[d], ca.cases[d]))
        end
    end
    return rows
end

bp   = load_reps("simulations", 3)
popn = load_reps("simulations/_popn_check", 3)

n_bp_reps   = length(unique(bp.rep))
n_popn_reps = length(unique(popn.rep))
println("Loaded $n_bp_reps BP reps and $n_popn_reps popn reps")
println()

# Daily summary table
println("day |  BP I (mean ± sd)  |  popn I (mean ± sd)  |  BP cases  |  popn cases")
println("----+--------------------+----------------------+------------+-------------")
for d in [1, 15, 30, 50, 70, 90, 110, 130, 150]
    bp_d   = bp[bp.day .== d, :]
    popn_d = popn[popn.day .== d, :]
    @printf "%3d |  %6.1f ± %5.1f    |  %6.1f ± %5.1f      |  %6.1f    |  %6.1f\n" d mean(bp_d.I) std(bp_d.I) mean(popn_d.I) std(popn_d.I) mean(bp_d.cases) mean(popn_d.cases)
end
println()

# Peak distribution
function peak_stats(df, label)
    peaks = Float64[]
    pdays = Int[]
    for r in unique(df.rep)
        d = df[df.rep .== r, :]
        pk, ipk = findmax(d.I)
        push!(peaks, pk); push!(pdays, ipk)
    end
    println("$label peak I:  median=$(round(median(peaks), digits=0))  range=$(round(minimum(peaks), digits=0))..$(round(maximum(peaks), digits=0))  CV=$(round(std(peaks)/mean(peaks), digits=3))")
    println("$label peak day: median=$(median(pdays))  range=$(minimum(pdays))..$(maximum(pdays))")
end
peak_stats(bp, "BP  ")
peak_stats(popn, "popn")
