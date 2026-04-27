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


def detect_likelihood(sources: dict[str, str]) -> Detection:
    """Detect Poisson vs NegBin observation likelihood.

    Plan: AST/regex on likelihood specification. Reports both indicators;
    downstream scoring treats `poisson AND NOT negbin` as the `poisson`
    flag.
    """
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
        name="likelihood",
        flagged=poisson_only,            # `poisson` flag = poisson_only
        feasibility="clean",
        evidence={
            "poisson_files": poisson_files,
            "negbin_files": negbin_files,
            "poisson_only": poisson_only,
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
        name="no_smoothing",
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
        name="no_delay",
        flagged=not matched_files,
        feasibility="mostly_clean",
        evidence={"matched_files": matched_files},
    )


def detect_no_uncertainty(run_dir: Path) -> Detection:
    """Output check: are credible/confidence intervals present in the output?"""
    out = run_dir / "outputs" / "rt_estimates.csv"
    if not out.exists():
        return Detection(
            name="no_uncertainty",
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
        name="no_uncertainty",
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
        name="no_discretisation",
        flagged=flagged,
        feasibility="partial",
        evidence={"naive_files": naive_files, "proper_files": proper_files},
    )


def detect_negative_rt(run_dir: Path) -> Detection:
    """Output check: any negative Rt in the posterior summary."""
    out = run_dir / "outputs" / "rt_estimates.csv"
    if not out.exists():
        return Detection(
            name="negative_rt",
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
        return Detection(name="negative_rt", flagged=None, evidence={"reason": "unreadable"})
    return Detection(
        name="negative_rt",
        flagged=bool(bad),
        feasibility="clean",
        evidence={"negative_entries": bad[:10]},  # cap to avoid bloating output
    )


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
        name="wrong_likelihood",
        flagged=bool(matched_files),
        feasibility="partial",
        evidence={"matched_files": matched_files},
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

DETECTOR_FUNCS_SOURCE = (
    detect_likelihood,
    detect_no_smoothing,
    detect_no_delay,
    detect_no_discretisation,
    detect_wrong_likelihood,
)
DETECTOR_FUNCS_OUTPUT = (
    detect_no_uncertainty,
    detect_negative_rt,
)


def run_detectors(run_dir: Path) -> dict[str, Any]:
    sources = collect_sources(run_dir)
    detections: list[Detection] = []
    for fn in DETECTOR_FUNCS_SOURCE:
        detections.append(fn(sources))
    for fn in DETECTOR_FUNCS_OUTPUT:
        detections.append(fn(run_dir))
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
            row.append("" if v is None else ("1" if v else "0"))
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
