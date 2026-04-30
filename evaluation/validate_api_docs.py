#!/usr/bin/env python3
"""LLM cross-check of the bundled API docs against the MWK operational rule.

Implements the LLM half of plan §"API docs MWK validation". Two LLM
instances from different families read each docs file and flag any section
that constitutes a worked Rt example. The internal-author validation is a
separate, parallel task (the human reviewer reads the same files and the
checklist in `prompts/paraphrase_brief.md`).

Usage:
    python evaluation/validate_api_docs.py
    python evaluation/validate_api_docs.py --providers openai google
    python evaluation/validate_api_docs.py --docs prompts/turing_api_docs.md

Output: a single markdown report at prompts/mwk_validation_report.md with
one section per (document × LLM family). Re-running overwrites the report.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Reuse provider adapters from the paraphrase generator.
sys.path.insert(0, str(Path(__file__).parent))
from generate_paraphrases import call_anthropic, call_openai, call_google  # noqa: E402

PROVIDERS = {
    "anthropic": ("anthropic", "claude-sonnet-4-5", call_anthropic),
    "openai":    ("openai",    "gpt-5",             call_openai),
    "google":    ("google",    "gemini-2.5-flash",  call_google),
}

VALIDATION_PROMPT = """You are validating an API reference document against the "minimum working knowledge" (MWK) rule from a study evaluating code-writing AI assistants.

The MWK rule says the API reference must be API-level only.

ALLOWED:
- function and type signatures, arguments, return types
- brief primitive-level usage snippets (single-function examples)
- abstract type hierarchies
- conceptual descriptions of what each function or type does

NOT ALLOWED:
- end-to-end examples that estimate the time-varying reproduction number $R_t$ from observed infection-related counts
- tutorials walking through full $R_t$-estimation model construction (load case data → compose a renewal-equation model → fit → extract $R_t$)
- worked examples with the structure of a complete solution to: "estimate $R_t$ from a daily case time series using this library"
- code blocks that, taken together, would let an LLM pattern-match to a complete answer for the task above

You are looking for any content that violates the NOT ALLOWED list. Brief snippets that demonstrate a single primitive (e.g. how to construct a single distribution, how to call a single function) are fine and should NOT be flagged.

Read the document below carefully. For each section you think violates the rule, output:

- The section heading or anchor (e.g. `## Generate observations`)
- The approximate line numbers
- A brief explanation (1–2 sentences) of why it violates
- A suggested edit (e.g. "remove the example block", "replace lines X–Y with the function signature only")

If no sections violate, output exactly: `No violations found.`

Be precise. Do not flag content that is allowed. Do not soften by suggesting "consider rewording"; either it violates or it does not. Output a markdown list, no preamble.

Document follows after the dashes.

---
{doc_content}
---

Validation report:"""


def validate_one(provider_key: str, doc_path: Path) -> str:
    if provider_key not in PROVIDERS:
        sys.exit(f"Unknown provider: {provider_key}")
    _, model, fn = PROVIDERS[provider_key]
    user_prompt = VALIDATION_PROMPT.format(doc_content=doc_path.read_text())
    return fn(model, user_prompt).strip()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--docs", nargs="+", type=Path,
                   default=[Path("prompts/turing_api_docs.md"), Path("prompts/epiaware_api_docs.md")])
    p.add_argument("--providers", nargs="+", choices=sorted(PROVIDERS.keys()),
                   default=["openai", "google"],
                   help="LLM families to cross-check with (default: openai google)")
    p.add_argument("--out", type=Path, default=Path("prompts/mwk_validation_report.md"))
    args = p.parse_args()

    sections: list[str] = []
    sections.append(f"# MWK API-docs validation report")
    sections.append(f"")
    sections.append(f"Generated: {datetime.utcnow().isoformat(timespec='seconds')}Z")
    sections.append(f"")
    sections.append(f"This is the LLM half of plan §\"API docs MWK validation\". An internal "
                    f"reviewer must independently read the same docs against the checklist in "
                    f"`prompts/paraphrase_brief.md`. Disagreements are surfaced here for "
                    f"resolution before pre-registration.")
    sections.append(f"")

    for doc_path in args.docs:
        if not doc_path.exists():
            sections.append(f"## {doc_path}\n\n(missing)\n")
            continue
        sections.append(f"## {doc_path}")
        sections.append(f"")
        sections.append(f"({len(doc_path.read_text().splitlines())} lines)")
        sections.append(f"")
        for provider_key in args.providers:
            _, model, _ = PROVIDERS[provider_key]
            print(f"validating {doc_path} via {provider_key}/{model} ...", flush=True)
            try:
                report = validate_one(provider_key, doc_path)
            except Exception as e:
                report = f"_(error: {type(e).__name__}: {e})_"
            sections.append(f"### {provider_key} / {model}")
            sections.append(f"")
            sections.append(report)
            sections.append(f"")
        sections.append(f"")

    args.out.write_text("\n".join(sections))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
