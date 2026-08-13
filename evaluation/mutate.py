#!/usr/bin/env python3
"""Inject defects from evaluation/mutation_catalogue.md into a reference solution.

Usage:
    python evaluation/mutate.py --list
    python evaluation/mutate.py --mutation M_dow_remove_obvious \
        --input reference_solutions/scenario_2_epiaware.jl \
        --output review_pool/sample_042/code/mutated.jl

The tool applies exactly one mutation to the input file and writes the mutated
version to the output. A JSON record of what was applied is written alongside
(same basename, `.mutation.json` extension) so the review coordinator can seal
the mapping.

Not every mutation in the catalogue is automatable. Subtle numerical / structural
mutations (M_conv_offbyone, M_prior_wrong, M_stream_independent) are marked
`manual = true` in the registry and must be applied by hand; running --mutation
on one of them writes an instruction stub instead of a modified file.

Automated mutations are pattern-based and can over-apply on complex references.
For example, M_smooth_remove replaces every RandomWalk/AR call, so if the
reference uses RandomWalk for both the Rt latent AND the ascertainment latent,
both are replaced. The coordinator should diff the mutated file against the
original after each automated application and trim any collateral changes by
hand.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class Mutation:
    id: str
    description: str
    component: str
    difficulty: str  # obvious / moderate / subtle
    applies_to: list[str]  # scenario tokens that this mutation makes sense for
    manual: bool  # True if the mutation is too complex to automate cleanly
    apply: Callable[[str], str] | None  # code -> code, or None if manual


# ---------------------------------------------------------------------------
# Concrete mutations
# ---------------------------------------------------------------------------


def _remove_matching_block(code: str, pattern: re.Pattern[str]) -> str:
    """Remove lines whose stripped text matches the pattern.

    Also drops immediately-following continuation lines (indented deeper than
    the matched line) up to the next line at equal-or-shallower indentation.
    """
    lines = code.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if pattern.search(stripped):
            indent = len(line) - len(line.lstrip())
            i += 1
            while i < len(lines):
                nxt = lines[i]
                nxt_indent = len(nxt) - len(nxt.lstrip())
                if nxt.strip() == "" or nxt_indent > indent:
                    i += 1
                else:
                    break
        else:
            out.append(line)
            i += 1
    return "".join(out)


def _replace_first(code: str, pattern: re.Pattern[str], replacement: str) -> str:
    return pattern.sub(replacement, code, count=1)


def _replace_call(code: str, name: str, replacement: str) -> str:
    """Replace every `name(...)` call in `code` with `replacement`, handling nested parens.

    Matches a bareword identifier followed by `(` and walks the paren balance
    to find the matching `)`. Safer than a simple regex for calls whose
    arguments contain further calls.
    """
    out: list[str] = []
    i = 0
    ident_re = re.compile(rf"\b{re.escape(name)}\s*\(")
    while i < len(code):
        m = ident_re.search(code, i)
        if not m:
            out.append(code[i:])
            break
        out.append(code[i:m.start()])
        depth = 1
        j = m.end()
        while j < len(code) and depth > 0:
            c = code[j]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            j += 1
        if depth != 0:
            # Unbalanced; skip this occurrence and continue after the identifier.
            out.append(code[m.start():j])
        else:
            out.append(replacement)
        i = j
    return "".join(out)


def _unwrap_call(code: str, name: str, keep_arg: int = 0) -> str:
    """Replace every `name(a, b, ...)` with the `keep_arg`-th argument (default: first).

    Handles nested parens inside arguments. Splits the top-level argument list
    on commas at depth 0.
    """
    out: list[str] = []
    i = 0
    ident_re = re.compile(rf"\b{re.escape(name)}\s*\(")
    while i < len(code):
        m = ident_re.search(code, i)
        if not m:
            out.append(code[i:])
            break
        out.append(code[i:m.start()])
        depth = 1
        j = m.end()
        args: list[str] = []
        cur = []
        while j < len(code) and depth > 0:
            c = code[j]
            if c == "(":
                depth += 1
                cur.append(c)
            elif c == ")":
                depth -= 1
                if depth == 0:
                    args.append("".join(cur))
                    cur = []
                else:
                    cur.append(c)
            elif c == "," and depth == 1:
                args.append("".join(cur))
                cur = []
            else:
                cur.append(c)
            j += 1
        if depth != 0 or keep_arg >= len(args):
            out.append(code[m.start():j])
        else:
            out.append(args[keep_arg].strip())
        i = j
    return "".join(out)


# --- DoW -----------------------------------------------------------------

def mutate_dow_remove_obvious(code: str) -> str:
    # EpiAware idiom: ascertainment_dayofweek(base_obs) — remove the wrapper.
    code = re.sub(
        r"ascertainment_dayofweek\s*\(\s*([^)]+?)\s*\)",
        r"\1",
        code,
    )
    # Raw-code idiom: `dow_effect[dow_idx[t]]` factor in an expected-count
    # expression. Drop the factor if we can find it.
    code = re.sub(
        r"\s*\*\s*dow_effect\s*\[[^\]]+\]",
        "",
        code,
    )
    code = re.sub(
        r"dow_effect\s*\[[^\]]+\]\s*\*\s*",
        "",
        code,
    )
    return code


# --- Ascertainment -------------------------------------------------------

def mutate_asc_remove_obvious(code: str) -> str:
    # EpiAware: unwrap Ascertainment(inner, latent_process) → inner
    code = re.sub(
        r"Ascertainment\s*\(\s*([^,]+?)\s*,\s*[^)]+?\s*\)",
        r"\1",
        code,
        flags=re.DOTALL,
    )
    # Raw-code: alpha[t] → 0.5 (a scalar constant)
    code = re.sub(
        r"alpha\s*\[\s*[a-zA-Z0-9_]+\s*\]",
        "0.5",
        code,
    )
    return code


# --- Delay ---------------------------------------------------------------

def mutate_delay_remove(code: str) -> str:
    # EpiAware: unwrap LatentDelay(inner, delay_pmf) → inner
    code = re.sub(
        r"LatentDelay\s*\(\s*([^,]+?)\s*,\s*[^)]+?\s*\)",
        r"\1",
        code,
        flags=re.DOTALL,
    )
    # Raw-code: sum(f[e] * I[t-e] for e in 0:D) → I[t]
    code = re.sub(
        r"sum\s*\(\s*f\s*\[\s*e\s*\]\s*\*\s*I\s*\[\s*t\s*-\s*e\s*\]\s+for\s+e\s+in\s+0:\s*D\s*\)",
        "I[t]",
        code,
    )
    return code


# --- Censoring -----------------------------------------------------------

def mutate_censor_replace_with_pdf(code: str) -> str:
    # EpiAware `censored_pmf(dist; Δd=1.0)` idiom → naive `pdf.(dist, 1:D)`
    code = re.sub(
        r"censored_pmf\s*\(\s*([^;)]+?)\s*(?:;[^)]*)?\)",
        r"pdf.(\1, 1:D)",
        code,
    )
    # cdf-difference discretisation: `cdf(d, x+0.5) - cdf(d, x-0.5)` →
    # `pdf(d, x)`
    code = re.sub(
        r"cdf\s*\(\s*([^,]+?)\s*,\s*([^)]+?)\s*\+\s*0\.5\s*\)\s*-\s*cdf\s*\(\s*\1\s*,\s*\2\s*-\s*0\.5\s*\)",
        r"pdf(\1, \2)",
        code,
    )
    return code


# --- Truncation ----------------------------------------------------------

def mutate_trunc_remove(code: str) -> str:
    # Unwrap Truncated(dist, lo, hi) → dist
    code = re.sub(
        r"[Tt]runcated\s*\(\s*([^,]+?)\s*,\s*[^,]+?\s*,\s*[^)]+?\s*\)",
        r"\1",
        code,
        flags=re.DOTALL,
    )
    # Remove PMF renormalisation
    code = re.sub(r"pmf\s*/?=\s*sum\s*\(\s*pmf\s*\)\s*\n?", "\n", code)
    code = re.sub(r"\bpmf\s*=\s*pmf\s*/\s*sum\s*\(\s*pmf\s*\)\s*\n?", "\n", code)
    return code


# --- Likelihood ----------------------------------------------------------

def mutate_likelihood_poisson(code: str) -> str:
    # EpiAware: NegativeBinomialError(cluster_factor_prior=X) → PoissonError()
    code = _replace_call(code, "NegativeBinomialError", "PoissonError()")
    # Raw-code NegativeBinomial(mean, ...) → Poisson(mean). Uses _unwrap_call
    # to keep the first argument (typically the mean); wraps in Poisson(...).
    def _to_poisson(src: str) -> str:
        return _replace_by_first_arg(src, "NegativeBinomial", "Poisson")
    code = _to_poisson(code)
    return code


def _replace_by_first_arg(code: str, name: str, new_name: str) -> str:
    """Replace `name(first_arg, ...)` with `new_name(first_arg)`. Nested-paren-safe."""
    out: list[str] = []
    i = 0
    ident_re = re.compile(rf"\b{re.escape(name)}\s*\(")
    while i < len(code):
        m = ident_re.search(code, i)
        if not m:
            out.append(code[i:])
            break
        out.append(code[i:m.start()])
        depth = 1
        j = m.end()
        args: list[str] = []
        cur = []
        while j < len(code) and depth > 0:
            c = code[j]
            if c == "(":
                depth += 1
                cur.append(c)
            elif c == ")":
                depth -= 1
                if depth == 0:
                    args.append("".join(cur))
                    cur = []
                else:
                    cur.append(c)
            elif c == "," and depth == 1:
                args.append("".join(cur))
                cur = []
            else:
                cur.append(c)
            j += 1
        if depth == 0 and args:
            out.append(f"{new_name}({args[0].strip()})")
        else:
            out.append(code[m.start():j])
        i = j
    return "".join(out)


# --- Multi-stream --------------------------------------------------------

def mutate_stream_cases_only(code: str) -> str:
    # Drop hospitalisations and deaths from the stack, keep cases only.
    # EpiAware: StackObservationModels((cases=..., hospitalisations=..., deaths=...))
    # → StackObservationModels((cases=...,))
    def strip_extra(match: re.Match[str]) -> str:
        inner = match.group(1)
        # Keep only the cases entry.
        cases_re = re.search(r"cases\s*=\s*[^,]+(?:\([^)]*\))?[^,]*", inner, re.DOTALL)
        if cases_re:
            return f"StackObservationModels(({cases_re.group(0)},))"
        return match.group(0)
    code = re.sub(
        r"StackObservationModels\s*\(\s*\(\s*(.+?)\s*\)\s*\)",
        strip_extra,
        code,
        flags=re.DOTALL,
    )
    # Raw-code: comment-out reads of hospitalisations.csv and deaths.csv
    code = re.sub(r"^([^\n]*hospitalisations\.csv[^\n]*)$", r"# \1", code, flags=re.MULTILINE)
    code = re.sub(r"^([^\n]*deaths\.csv[^\n]*)$", r"# \1", code, flags=re.MULTILINE)
    return code


# --- Smoothing -----------------------------------------------------------

def mutate_smooth_remove(code: str) -> str:
    # EpiAware: RandomWalk(...) → IID(Normal(0, 1))
    code = _replace_call(code, "RandomWalk", "IID(Normal(0, 1))")
    # AR(k, ...) → IID(Normal(0, 1))
    code = _replace_call(code, "AR", "IID(Normal(0, 1))")
    return code


# ---------------------------------------------------------------------------
# Mutation registry
# ---------------------------------------------------------------------------

MUTATIONS: dict[str, Mutation] = {
    "M_dow_remove_obvious": Mutation(
        id="M_dow_remove_obvious",
        description="Remove day-of-week effect entirely.",
        component="flag_no_dow",
        difficulty="obvious",
        applies_to=["scenario_2", "scenario_3"],
        manual=False,
        apply=mutate_dow_remove_obvious,
    ),
    "M_dow_broadcast_wrong": Mutation(
        id="M_dow_broadcast_wrong",
        description="Broadcast DoW parameter against the wrong axis.",
        component="flag_no_dow",
        difficulty="moderate",
        applies_to=["scenario_2", "scenario_3"],
        manual=True,
        apply=None,
    ),
    "M_asc_remove_obvious": Mutation(
        id="M_asc_remove_obvious",
        description="Replace time-varying ascertainment with a scalar constant.",
        component="flag_no_ascertainment",
        difficulty="obvious",
        applies_to=["scenario_2", "scenario_3"],
        manual=False,
        apply=mutate_asc_remove_obvious,
    ),
    "M_asc_fixed_wrong": Mutation(
        id="M_asc_fixed_wrong",
        description="Ascertainment vector hard-coded to constant across time.",
        component="flag_no_ascertainment",
        difficulty="moderate",
        applies_to=["scenario_2", "scenario_3"],
        manual=True,
        apply=None,
    ),
    "M_delay_remove": Mutation(
        id="M_delay_remove",
        description="Remove reporting-delay convolution.",
        component="flag_no_delay_handling",
        difficulty="obvious",
        applies_to=["scenario_1a", "scenario_1b", "scenario_2", "scenario_3"],
        manual=False,
        apply=mutate_delay_remove,
    ),
    "M_censor_replace_with_pdf": Mutation(
        id="M_censor_replace_with_pdf",
        description="Replace censored_pmf / cdf-diff with naive pdf(dist, 1:D).",
        component="flag_no_censoring",
        difficulty="moderate",
        applies_to=["scenario_1a", "scenario_1b", "scenario_2", "scenario_3"],
        manual=False,
        apply=mutate_censor_replace_with_pdf,
    ),
    "M_trunc_remove": Mutation(
        id="M_trunc_remove",
        description="Remove PMF renormalisation and Truncated wrapper.",
        component="flag_no_truncation",
        difficulty="moderate",
        applies_to=["scenario_1a", "scenario_1b", "scenario_2", "scenario_3"],
        manual=False,
        apply=mutate_trunc_remove,
    ),
    "M_trunc_extend_wrong": Mutation(
        id="M_trunc_extend_wrong",
        description="Extend the truncation window well beyond the delay support.",
        component="flag_no_truncation",
        difficulty="subtle",
        applies_to=["scenario_1a", "scenario_1b", "scenario_2", "scenario_3"],
        manual=True,
        apply=None,
    ),
    "M_likelihood_poisson": Mutation(
        id="M_likelihood_poisson",
        description="Replace NegBin observation likelihood with Poisson.",
        component="flag_poisson_only",
        difficulty="obvious",
        applies_to=["scenario_2", "scenario_3"],
        manual=False,
        apply=mutate_likelihood_poisson,
    ),
    "M_stream_independent": Mutation(
        id="M_stream_independent",
        description="Break shared latent: fit three independent Rt latents.",
        component="flag_no_multistream_latent",
        difficulty="obvious",
        applies_to=["scenario_3"],
        manual=True,
        apply=None,
    ),
    "M_stream_cases_only": Mutation(
        id="M_stream_cases_only",
        description="Ignore hospitalisations and deaths; use cases only.",
        component="flag_no_multistream_latent",
        difficulty="obvious",
        applies_to=["scenario_3"],
        manual=False,
        apply=mutate_stream_cases_only,
    ),
    "M_smooth_remove": Mutation(
        id="M_smooth_remove",
        description="Replace AR/RW smoothing with independent-day prior on Rt.",
        component="flag_no_smoothing_term",
        difficulty="obvious",
        applies_to=["scenario_1a", "scenario_1b", "scenario_2", "scenario_3"],
        manual=False,
        apply=mutate_smooth_remove,
    ),
    "M_smooth_scale_wrong": Mutation(
        id="M_smooth_scale_wrong",
        description="Innovation prior scale wrong by 10×.",
        component="flag_no_smoothing_term",
        difficulty="subtle",
        applies_to=["scenario_1a", "scenario_1b", "scenario_2", "scenario_3"],
        manual=True,
        apply=None,
    ),
    "M_gi_reversed": Mutation(
        id="M_gi_reversed",
        description="Reverse the generation-interval vector in the convolution.",
        component="expert_only",
        difficulty="moderate",
        applies_to=["scenario_1a", "scenario_1b", "scenario_2", "scenario_3"],
        manual=True,
        apply=None,
    ),
    "M_conv_offbyone": Mutation(
        id="M_conv_offbyone",
        description="Off-by-one in the delay convolution index.",
        component="expert_only",
        difficulty="subtle",
        applies_to=["scenario_1a", "scenario_1b", "scenario_2", "scenario_3"],
        manual=True,
        apply=None,
    ),
    "M_prior_wrong": Mutation(
        id="M_prior_wrong",
        description="Wildly wrong initialisation prior on log-Rt AR(1).",
        component="expert_only",
        difficulty="subtle",
        applies_to=["scenario_1a", "scenario_1b", "scenario_2", "scenario_3"],
        manual=True,
        apply=None,
    ),
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_list() -> None:
    print(f"{'ID':<30} {'component':<28} {'diff':<10} {'auto?':<5}")
    print("-" * 78)
    for m in MUTATIONS.values():
        auto = "no" if m.manual else "yes"
        print(f"{m.id:<30} {m.component:<28} {m.difficulty:<10} {auto:<5}")
    print()
    print("Applies-to per mutation is available via --describe <ID>.")


def cmd_describe(mid: str) -> None:
    m = MUTATIONS.get(mid)
    if m is None:
        sys.exit(f"Unknown mutation ID: {mid}")
    print(f"ID:          {m.id}")
    print(f"Description: {m.description}")
    print(f"Component:   {m.component}")
    print(f"Difficulty:  {m.difficulty}")
    print(f"Applies to:  {', '.join(m.applies_to)}")
    print(f"Automated:   {'no (manual edit required)' if m.manual else 'yes'}")


def cmd_apply(mid: str, in_path: Path, out_path: Path) -> None:
    m = MUTATIONS.get(mid)
    if m is None:
        sys.exit(f"Unknown mutation ID: {mid}")
    src = in_path.read_text()
    if m.manual:
        note = (
            f"# MANUAL MUTATION REQUIRED: {m.id}\n"
            f"# Description: {m.description}\n"
            f"# Component: {m.component}\n"
            f"# Original file: {in_path}\n"
            f"# Apply the mutation by hand, then remove this header.\n\n"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(note + src)
        record = {"mutation": m.id, "manual": True, "input": str(in_path), "output": str(out_path)}
    else:
        assert m.apply is not None
        mutated = m.apply(src)
        if mutated == src:
            sys.exit(
                f"Mutation {m.id} produced no change on {in_path}. The mutation may not apply "
                f"to this file (patterns not found)."
            )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(mutated)
        record = {"mutation": m.id, "manual": False, "input": str(in_path), "output": str(out_path)}
    (out_path.with_suffix(out_path.suffix + ".mutation.json")).write_text(
        json.dumps(record, indent=2)
    )
    print(f"Applied {m.id} to {in_path} → {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true", help="list all mutation IDs")
    p.add_argument("--describe", metavar="ID", help="describe one mutation")
    p.add_argument("--mutation", metavar="ID", help="mutation to apply")
    p.add_argument("--input", type=Path, help="reference solution to mutate")
    p.add_argument("--output", type=Path, help="path to write the mutated file")
    args = p.parse_args()

    if args.list:
        cmd_list()
        return
    if args.describe:
        cmd_describe(args.describe)
        return
    if args.mutation:
        if not args.input or not args.output:
            sys.exit("--mutation requires --input and --output")
        cmd_apply(args.mutation, args.input, args.output)
        return
    p.print_help()


if __name__ == "__main__":
    main()
