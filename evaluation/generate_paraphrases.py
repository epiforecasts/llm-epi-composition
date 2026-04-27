#!/usr/bin/env python3
"""Generate paraphrases of the base prompts via a separate Claude instance.

Implements the LLM wave of plan §"Randomisation → Prompt paraphrases".
The manual wave (one author blinded to hypothesis direction) is performed
separately by a human and saved to the same directory layout.

Usage:
    python generate_paraphrases.py
    python generate_paraphrases.py --slots 02 03 04 05
    python generate_paraphrases.py --scenarios scenario_1a --conditions epiaware

Reads `prompts/{scenario}/paraphrases/{condition}/01.md` (the base) and
writes paraphrased versions to slots 02..05 in the same directory.

The paraphraser instance receives only the base prompt and the
paraphrasing instruction; it has no knowledge of the study design,
the hypotheses, or the existence of other conditions. Outputs are
deterministically named so re-running with --skip-existing avoids
overwriting hand-edited paraphrases.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable

PARAPHRASE_INSTRUCTION = """You are paraphrasing a task description for a code-writing assistant.

Your job:
- Preserve every factual statement: data file paths and column names; exact numerical parameters; exact distributional assumptions; the framework or language constraint if any; the output file path and its required columns; the structural features mentioned about the data; the expected error-handling behaviour.
- Vary the wording, ordering, headings, sentence structure, and tone.
- Do not add, remove, or reinterpret any factual content.
- Do not add examples, hints, advice, methodological suggestions, or strategies for solving the task.
- Output only the paraphrased task description in markdown. No preamble. No commentary. No explanation of what you changed.

Original task description follows after the dashes.

---
{base_prompt}
---

Paraphrased task description:"""


def paraphrase_one(base_text: str, model: str) -> str:
    prompt = PARAPHRASE_INSTRUCTION.format(base_prompt=base_text)
    proc = subprocess.run(
        ["claude", "--print", "--model", model, prompt],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        sys.stderr.write(f"claude exited {proc.returncode}\nstderr: {proc.stderr}\n")
        sys.exit(proc.returncode)
    return proc.stdout.strip()


def iter_targets(
    prompts_dir: Path,
    scenarios: Iterable[str],
    conditions: Iterable[str],
    slots: Iterable[int],
):
    for sc in scenarios:
        for c in conditions:
            base = prompts_dir / sc / "paraphrases" / c / "01.md"
            if not base.exists():
                yield (sc, c, None, None, f"missing base: {base}")
                continue
            for slot in slots:
                out_path = prompts_dir / sc / "paraphrases" / c / f"{slot:02d}.md"
                yield (sc, c, slot, base, out_path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompts-dir", type=Path, default=Path("prompts"))
    p.add_argument("--scenarios", nargs="+", default=["scenario_1a", "scenario_1b", "scenario_2", "scenario_3"])
    p.add_argument("--conditions", nargs="+", default=["no-spec", "julia", "epiaware"])
    p.add_argument("--slots", nargs="+", type=int, default=[2, 3, 4, 5],
                   help="paraphrase slots to fill (slot 1 is the base, never overwritten)")
    p.add_argument("--model", default="claude-sonnet-4-5")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip slots whose file already exists (default: overwrite)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if 1 in args.slots:
        sys.exit("Refusing to overwrite slot 01 (the base prompt). Remove 1 from --slots.")

    n_done = n_skipped = n_planned = 0
    for sc, c, slot, base, out_path in iter_targets(args.prompts_dir, args.scenarios, args.conditions, args.slots):
        if slot is None:
            print(out_path)  # error message threaded through
            continue
        n_planned += 1
        if args.skip_existing and out_path.exists():
            print(f"SKIP existing: {out_path}")
            n_skipped += 1
            continue
        if args.dry_run:
            print(f"would write: {out_path}")
            continue
        base_text = base.read_text()
        print(f"generating {sc}/{c}/{slot:02d}.md ...", flush=True)
        para = paraphrase_one(base_text, args.model)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(para)
        print(f"  wrote {len(para)} chars to {out_path}")
        n_done += 1

    print(f"\nplanned={n_planned}  written={n_done}  skipped_existing={n_skipped}")


if __name__ == "__main__":
    main()
