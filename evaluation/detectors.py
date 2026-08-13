#!/usr/bin/env python3
"""Automated correctness detectors for LLM-generated Rt estimators.

Implements the static/output detectors listed in analysis_plan.md
§"Evaluation → Diagnostic: Automated correctness detectors".

Usage:
    python detectors.py <run_dir>                # one run, JSON to stdout
    python detectors.py --all <runs_root>        # every run, CSV to stdout

A run_dir is the leaf directory created by evaluation/run_agentic.sh:
    runs/{scenario}/{condition}/{variant}/rep_{rr}/{model}/run_{NN}/

containing the agent's source files, an outputs/ subdir with
rt_estimates.csv (if produced), conversation.jsonl, metadata.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

SOURCE_EXTENSIONS = (".jl", ".py", ".R", ".r", ".stan")


# ---------------------------------------------------------------------------
# Source collection
# ---------------------------------------------------------------------------

def collect_sources(run_dir: Path) -> dict[str, str]:
    """Collect candidate source files from the run directory.

    Walk the working directory but skip the data/ subdirectory and any
    bundled docs files (turing_api_docs.md, epiaware_docs.md).
    """
    skip_files = {"turing_api_docs.md", "epiaware_docs.md"}
    sources: dict[str, str] = {}
    for ext in SOURCE_EXTENSIONS:
        for path in run_dir.glob(f"*{ext}"):
            if path.name in skip_files or path.parent.name == "data":
                continue
            try:
                sources[path.name] = path.read_text(errors="replace")
            except OSError:
                continue
    return sources


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    name: str
    flagged: bool | None
    feasibility: str = "clean"
    evidence: dict[str, Any] = field(default_factory=dict)


def detect_poisson_only(sources: dict[str, str]) -> Detection:
    """Flag: Poisson observation likelihood with no NegBin alternative."""
    poisson_re = re.compile(r"\bPoisson\b")
    negbin_re = re.compile(
        r"\b(NegativeBinomial|NegBin|NegBinomial|negative_binomial|negbinomial|"
        r"NegativeBinomialError|negbin)\b",
        re.IGNORECASE,
    )
    poisson_files = [n for n, s in sources.items() if poisson_re.search(s)]
    negbin_files = [n for n, s in sources.items() if negbin_re.search(s)]
    poisson_only = bool(poisson_files) and not negbin_files
    return Detection(
        name="flag_poisson_only",
        flagged=poisson_only,
        feasibility="clean",
        evidence={
            "poisson_files": poisson_files,
            "negbin_files": negbin_files,
        },
    )


def detect_no_smoothing(sources: dict[str, str]) -> Detection:
    """Detect absence of smoothing on Rt: AR/RW/GP/spline."""
    patterns = [
        r"\bAR\s*\(",
        r"\bAR\d?\b",
        r"\bRandomWalk\b",
        r"\bRW\s*\(",
        r"\brandom[ _]walk\b",
        r"\bGP\s*\(",
        r"\bGaussianProcess\b",
        r"\bgp_\w+",
        r"\bspline\b",
        r"\bsmooth\w*",
        r"\bautoregress\w*",
        r"\bAutoRegress\w*",
    ]
    matched_files: list[str] = []
    for name, src in sources.items():
        if any(re.search(p, src, re.IGNORECASE) for p in patterns):
            matched_files.append(name)
    return Detection(
        name="flag_no_smoothing_term",
        flagged=not matched_files,
        feasibility="clean",
        evidence={"matched_files": matched_files},
    )


def detect_no_delay(sources: dict[str, str]) -> Detection:
    """Detect absence of any delay handling in the model."""
    patterns = [
        r"\bdelay\b",
        r"\bLatentDelay\b",
        r"\breporting[_ ]delay\b",
        r"\bf_e\b",
        r"\bf_d\b",
        r"\bdelay_pmf\b",
        r"\bconvolv\w*",
        r"\bconvol_\w*",
    ]
    matched_files: list[str] = []
    for name, src in sources.items():
        if any(re.search(p, src, re.IGNORECASE) for p in patterns):
            matched_files.append(name)
    return Detection(
        name="flag_no_delay_handling",
        flagged=not matched_files,
        feasibility="mostly_clean",
        evidence={"matched_files": matched_files},
    )


def detect_no_uncertainty(run_dir: Path) -> Detection:
    """Output check: are credible/confidence intervals present in the output?"""
    out = run_dir / "outputs" / "rt_estimates.csv"
    if not out.exists():
        return Detection(
            name="flag_no_uncertainty",
            flagged=None,
            feasibility="clean",
            evidence={"reason": "no outputs/rt_estimates.csv"},
        )
    try:
        header = out.read_text(errors="replace").splitlines()[0].lower()
    except OSError:
        return Detection(name="no_uncertainty", flagged=None, evidence={"reason": "unreadable"})
    has_lower = "rt_lower" in header or "lower" in header
    has_upper = "rt_upper" in header or "upper" in header
    has_intervals = has_lower and has_upper
    return Detection(
        name="flag_no_uncertainty",
        flagged=not has_intervals,
        feasibility="clean",
        evidence={"header": header, "has_lower": has_lower, "has_upper": has_upper},
    )


def detect_no_discretisation(sources: dict[str, str]) -> Detection:
    """Heuristic: continuous density evaluated at integer points without integration.

    Plan flags this as `partial`. We look for two patterns:
      naive   = `pdf(dist, 1:N)` / `pdf.(dist, 1:N)` / similar with integer args
      proper  = `cdf(...) - cdf(...)`, `censored_pmf`, `quadgk`, `integrate`
    A run is flagged when there is naive density use but no proper integration.
    """
    naive_re = re.compile(
        r"\bpdf\s*\(\s*\w+(?:\(.*?\))?\s*,\s*[0-9:.\sA-Za-z]+\)|"
        r"\bpdf\.\s*\(\s*\w+\s*,\s*[0-9:]",
    )
    proper_re = re.compile(
        r"\bcensored_pmf\b|"
        r"\bdouble_censored_pmf\b|"
        r"\bcdf\s*\(.+?\)\s*-\s*cdf\s*\(|"
        r"\bquadgk\b|"
        r"\bintegrate\b|"
        r"\bnumerical_integ\w*",
        re.IGNORECASE,
    )
    naive_files = [n for n, s in sources.items() if naive_re.search(s)]
    proper_files = [n for n, s in sources.items() if proper_re.search(s)]
    flagged = bool(naive_files) and not proper_files
    return Detection(
        name="flag_naive_density_at_integers",
        flagged=flagged,
        feasibility="partial",
        evidence={"naive_files": naive_files, "proper_files": proper_files},
    )


def detect_negative_rt(run_dir: Path) -> Detection:
    """Output check: any negative Rt in the posterior summary."""
    out = run_dir / "outputs" / "rt_estimates.csv"
    if not out.exists():
        return Detection(
            name="flag_negative_rt",
            flagged=None,
            feasibility="clean",
            evidence={"reason": "no outputs/rt_estimates.csv"},
        )
    cols_to_check = ("Rt_median", "Rt_lower", "rt_median", "rt_lower")
    bad: list[tuple[str, str]] = []
    try:
        with open(out, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for col in cols_to_check:
                    if col in row and row[col] not in ("", None):
                        try:
                            v = float(row[col])
                        except ValueError:
                            continue
                        if v < 0:
                            bad.append((col, row[col]))
    except OSError:
        return Detection(name="flag_negative_rt", flagged=None, evidence={"reason": "unreadable"})
    return Detection(
        name="flag_negative_rt",
        flagged=bool(bad),
        feasibility="clean",
        evidence={"negative_entries": bad[:10]},  # cap to avoid bloating output
    )


def detect_no_censoring(sources: dict[str, str]) -> Detection:
    """Flag when reporting-delay handling omits interval censoring.

    Aligned with the DGP's double interval censoring. Distinguishes proper
    censoring (censored_pmf, CDF-difference discretisation, double interval
    censoring) from delays handled without censoring (raw pdf lookup at integers,
    exponential-decay convolution kernels, or no delay at all).
    """
    proper_patterns = [
        r"\bcensored_pmf\b",
        r"\bdouble[_ ]censored[_ ]pmf\b",
        r"\bdouble[_ ]interval[_ ]censoring\b",
        r"\binterval[_ ]censor\w*",
        r"\bcdf\s*\(.+?\)\s*-\s*cdf\s*\(",
    ]
    proper_files = [
        n for n, s in sources.items()
        if any(re.search(p, s, re.IGNORECASE) for p in proper_patterns)
    ]
    # Only meaningful if the submission actually has delay handling at all.
    # If no delay is present at all, the censoring detector abstains (None).
    delay_present = detect_no_delay(sources).flagged is False
    return Detection(
        name="flag_no_censoring",
        flagged=(delay_present and not proper_files) if delay_present else None,
        feasibility="partial",
        evidence={
            "proper_files": proper_files,
            "delay_present": delay_present,
        },
    )


def detect_no_truncation(sources: dict[str, str]) -> Detection:
    """Flag when the delay or generation-interval distribution is not truncated.

    Estimator-side approaches to truncation include: renormalising a
    discretised PMF over a finite window, using `Truncated(dist, lo, hi)`,
    a `truncation` kwarg, or an explicit maximum-lag constant (D_max, S_max,
    tau_max, gen_max, delay_max, S, D). Distributions used as continuous
    convolution kernels without any of these are flagged.
    """
    patterns = [
        r"\bTruncated\s*\(",
        r"\btruncated\s*\(",
        r"\btruncation\b",
        r"\btruncate\w*",
        r"\bD_?max\b",
        r"\bS_?max\b",
        r"\btau_?max\b",
        r"\bgen_?max\b",
        r"\bdelay_?max\b",
        r"\bmax_?lag\b",
        r"\brenormali[sz]\w*",
        r"\bsum\s*\(pmf\)",
        r"\bpmf\s*/=?\s*sum\s*\(pmf\)",
    ]
    matched = [
        n for n, s in sources.items()
        if any(re.search(p, s, re.IGNORECASE) for p in patterns)
    ]
    return Detection(
        name="flag_no_truncation",
        flagged=not matched,
        feasibility="partial",
        evidence={"matched_files": matched},
    )


def detect_no_dow(sources: dict[str, str]) -> Detection:
    """Flag absence of any day-of-week or weekend effect in the model.

    Relevant for scenarios 2 and 3, where the DGP includes a Mon–Sun
    multiplier. The detector is condition- and scenario-agnostic; downstream
    analysis conditions on the scenario.
    """
    patterns = [
        r"\bdow\b",
        r"\bday[_ ]of[_ ]week\b",
        r"\bdayofweek\b",
        r"\bweekend\b",
        r"\bweekly[_ ](cycle|effect|multiplier)\b",
        r"\bascertainment_dayofweek\b",
        r"\bbroadcast_weekly\b",
        r"\bdow_\w+",
        r"\bw_dow\b",
        r"\bday_effect\b",
        r"Dates?\.dayofweek",
    ]
    matched = [
        n for n, s in sources.items()
        if any(re.search(p, s, re.IGNORECASE) for p in patterns)
    ]
    return Detection(
        name="flag_no_dow",
        flagged=not matched,
        feasibility="partial",
        evidence={"matched_files": matched},
    )


def detect_no_ascertainment(sources: dict[str, str]) -> Detection:
    """Flag absence of a time-varying ascertainment structure.

    Relevant for scenarios 2 and 3, where the DGP has a time-varying
    reporting fraction. Constant ascertainment counts as absent for the
    purposes of this flag: any scalar `alpha` or `p_report` without a time
    index is not enough. We require some indication of temporal variation
    (a vector, a random walk / AR on the ascertainment, or an EpiAware
    `Ascertainment` block).
    """
    patterns = [
        r"\bAscertainment\b",
        r"\bascertainment\w*\[",           # indexed ascertainment (time-varying)
        r"\balpha_t\b",
        r"\balpha\[",
        r"\breporting[_ ]fraction\b.*(RandomWalk|RW|AR|GP|spline|walk)",
        r"\bp_report\b.*(RandomWalk|RW|AR|GP|spline|walk)",
        r"\btime[_ ]varying[_ ]ascertain\w*",
        r"\btime[_ ]varying[_ ]report\w*",
    ]
    matched = [
        n for n, s in sources.items()
        if any(re.search(p, s, re.IGNORECASE) for p in patterns)
    ]
    return Detection(
        name="flag_no_ascertainment",
        flagged=not matched,
        feasibility="partial",
        evidence={"matched_files": matched},
    )


def detect_no_multistream_latent(sources: dict[str, str], run_dir: Path) -> Detection:
    """Scenario-3-only: flag when the model does not share latent Rt across streams.

    A scenario-3 submission is expected to model a single latent Rt (or a
    single latent infection process) that generates cases, hospitalisations,
    and deaths through separate observation models. Submissions that fit
    three independent single-stream models, or that only use one of the
    three data files, are flagged.

    Only meaningful for scenario_3. Returns `None` (abstain) for other
    scenarios.
    """
    scenario = _infer_scenario(run_dir)
    if scenario != "scenario_3":
        return Detection(
            name="flag_no_multistream_latent",
            flagged=None,
            feasibility="partial",
            evidence={"reason": f"scenario is {scenario}, not scenario_3"},
        )
    all_three_present = all(
        any(kw in s for s in sources.values())
        for kw in ("cases.csv", "hospitalisations.csv", "deaths.csv")
    )
    sharing_patterns = [
        r"\bStackObservationModels\b",
        r"\bstack_observation_models\b",
        r"\bshared[_ ]latent\b",
        r"\bshared[_ ]Rt\b",
        r"\bshared[_ ]infection\w*",
        r"\bshared[_ ]I_t\b",
        r"\bjoint[_ ]latent\b",
        r"\bjoint[_ ]infection\w*",
    ]
    sharing_files = [
        n for n, s in sources.items()
        if any(re.search(p, s, re.IGNORECASE) for p in sharing_patterns)
    ]
    flagged = not (all_three_present and sharing_files)
    return Detection(
        name="flag_no_multistream_latent",
        flagged=flagged,
        feasibility="partial",
        evidence={
            "all_three_streams_referenced": all_three_present,
            "sharing_files": sharing_files,
        },
    )


def detect_chosen_package(sources: dict[str, str]) -> Detection:
    """Identify the primary language + package the agent used.

    Non-boolean detector: `flagged` is the package label (e.g. "R_EpiNow2",
    "Python_numpyro", "Julia_EpiAware", "Julia_Turing"), or None if
    indeterminate. Used by analyse.R to test the "no-spec defaults to
    packages" prediction and to populate Table 6.
    """
    label: str | None = None
    # Prefer the language of the largest source file that isn't a helper.
    largest = None
    largest_len = -1
    for name, src in sources.items():
        n = len(src)
        if n > largest_len:
            largest_len = n
            largest = (name, src)
    if largest is None:
        return Detection(name="chosen_package", flagged=None, feasibility="partial",
                          evidence={"reason": "no source files"})
    name, src = largest
    ext = Path(name).suffix.lower()
    # Detect language.
    if ext == ".jl":
        lang = "Julia"
    elif ext == ".r":
        lang = "R"
    elif ext == ".py":
        lang = "Python"
    elif ext == ".stan":
        lang = "Stan"
    else:
        lang = "unknown"
    # Detect package by looking across all sources (imports may be elsewhere).
    joined = "\n".join(sources.values())
    pkg = None
    # Julia
    if re.search(r"\busing\s+EpiAware\b", joined) or re.search(r"\bEpiAware\.\w+", joined):
        pkg = "EpiAware"; lang = "Julia"
    elif re.search(r"\busing\s+Turing\b", joined) and lang == "Julia":
        pkg = "Turing"
    # R
    elif re.search(r"library\s*\(\s*EpiNow2\s*\)", joined) or re.search(r"\bEpiNow2::", joined):
        pkg = "EpiNow2"; lang = "R"
    elif re.search(r"library\s*\(\s*EpiEstim\s*\)", joined) or re.search(r"\bEpiEstim::", joined):
        pkg = "EpiEstim"; lang = "R"
    elif re.search(r"library\s*\(\s*epinowcast\s*\)", joined) or re.search(r"\bepinowcast::", joined):
        pkg = "epinowcast"; lang = "R"
    elif re.search(r"library\s*\(\s*cmdstanr\s*\)", joined) or re.search(r"\bcmdstanr::", joined):
        pkg = "cmdstanr"; lang = "R"
    elif re.search(r"library\s*\(\s*rstan\s*\)", joined) or re.search(r"\brstan::", joined):
        pkg = "rstan"; lang = "R"
    # Python
    elif re.search(r"\bimport\s+numpyro\b", joined) or re.search(r"\bfrom\s+numpyro\b", joined):
        pkg = "numpyro"; lang = "Python"
    elif re.search(r"\bimport\s+pymc\b", joined) or re.search(r"\bfrom\s+pymc\b", joined) or re.search(r"\bimport\s+pymc\s+as\s+pm\b", joined):
        pkg = "PyMC"; lang = "Python"
    elif re.search(r"\bimport\s+pyro\b", joined) or re.search(r"\bfrom\s+pyro\b", joined):
        pkg = "pyro"; lang = "Python"
    elif re.search(r"\bimport\s+cmdstanpy\b", joined) or re.search(r"\bfrom\s+cmdstanpy\b", joined):
        pkg = "cmdstanpy"; lang = "Python"
    elif re.search(r"\bimport\s+stan\b", joined) or re.search(r"\bfrom\s+stan\b", joined):
        pkg = "pystan"; lang = "Python"
    label = f"{lang}_{pkg}" if pkg else lang
    return Detection(
        name="chosen_package",
        flagged=label,
        feasibility="partial",
        evidence={"primary_file": name, "primary_ext": ext},
    )


def _infer_scenario(run_dir: Path) -> str | None:
    """Extract scenario name from run_dir path.

    Expected layout: runs/{scenario}/{condition}/par_{p}/{variant}/rep_{r}/{model}/run_{n}
    """
    parts = run_dir.parts
    for part in parts:
        if part.startswith("scenario_"):
            return part
    return None


def detect_wrong_likelihood(sources: dict[str, str]) -> Detection:
    """Heuristic: observation modelled with Normal/Gaussian rather than count distribution."""
    patterns = [
        r"y_t\s*~\s*Normal",
        r"\bcases?\s*~\s*Normal",
        r"\bC_t\s*~\s*Normal",
        r"\bNormal\([^)]*observ",
        r"\bGaussian\b.*y_t",
        r"\bnormal\(.*?\)\s*~\s*y",
    ]
    matched_files: list[str] = []
    for name, src in sources.items():
        if any(re.search(p, src) for p in patterns):
            matched_files.append(name)
    return Detection(
        name="flag_normal_observation",
        flagged=bool(matched_files),
        feasibility="partial",
        evidence={"matched_files": matched_files},
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

DETECTOR_FUNCS_SOURCE = (
    detect_poisson_only,
    detect_no_smoothing,
    detect_no_delay,
    detect_no_censoring,
    detect_no_truncation,
    detect_no_dow,
    detect_no_ascertainment,
    detect_no_discretisation,
    detect_wrong_likelihood,
    detect_chosen_package,
)
DETECTOR_FUNCS_OUTPUT = (
    detect_no_uncertainty,
    detect_negative_rt,
)
DETECTOR_FUNCS_SOURCE_AND_DIR = (
    detect_no_multistream_latent,
)


def run_detectors(run_dir: Path) -> dict[str, Any]:
    sources = collect_sources(run_dir)
    detections: list[Detection] = []
    for fn in DETECTOR_FUNCS_SOURCE:
        detections.append(fn(sources))
    for fn in DETECTOR_FUNCS_OUTPUT:
        detections.append(fn(run_dir))
    for fn in DETECTOR_FUNCS_SOURCE_AND_DIR:
        detections.append(fn(sources, run_dir))
    return {
        "run_dir": str(run_dir),
        "n_source_files": len(sources),
        "source_files": sorted(sources.keys()),
        "detections": [
            {
                "name": d.name,
                "flagged": d.flagged,
                "feasibility": d.feasibility,
                "evidence": d.evidence,
            }
            for d in detections
        ],
    }


def collect_run_dirs(root: Path) -> Iterable[Path]:
    """Yield every leaf run directory under root."""
    for path in root.rglob("conversation.jsonl"):
        yield path.parent


def emit_csv_summary(results: list[dict[str, Any]]) -> str:
    detector_names = []
    if results:
        detector_names = [d["name"] for d in results[0]["detections"]]
    fieldnames = ["run_dir", "n_source_files"] + detector_names
    out_lines = [",".join(fieldnames)]
    for r in results:
        flags = {d["name"]: d["flagged"] for d in r["detections"]}
        row = [r["run_dir"], str(r["n_source_files"])]
        for name in detector_names:
            v = flags.get(name)
            if v is None:
                row.append("")
            elif isinstance(v, bool):
                row.append("1" if v else "0")
            else:
                # String label (e.g. chosen_package). Quote if it contains a
                # comma; otherwise emit as-is.
                s = str(v)
                if "," in s or '"' in s:
                    s = '"' + s.replace('"', '""') + '"'
                row.append(s)
        out_lines.append(",".join(row))
    return "\n".join(out_lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", type=Path, help="run directory or runs root")
    p.add_argument("--all", action="store_true", help="walk runs root, emit CSV summary")
    args = p.parse_args()

    if args.all:
        results = [run_detectors(d) for d in collect_run_dirs(args.path)]
        print(emit_csv_summary(results))
    else:
        result = run_detectors(args.path)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
