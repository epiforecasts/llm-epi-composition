#!/bin/bash
# Driver for batched evaluation runs across scenarios × conditions × paraphrases
# × variants × replicates × runs. Calls evaluation/run_agentic.sh per tuple.
#
# Configuration is via environment variables (override on the command line),
# all space-separated lists:
#
#   SCENARIOS    default: scenario_1a scenario_1b scenario_2 scenario_3
#   CONDITIONS   default: no-spec julia epiaware
#   PARAPHRASES  default: 1 2 3 4 5
#   VARIANTS     default: canonical
#   REPLICATES   default: 1 2 3 4 5 6 7 8 9 10
#   MODELS       default: claude-sonnet-4-6
#   RUNS_PER_CELL default: 1     (number of repeated runs per tuple)
#
# Optional: set PRINT_ONLY=1 to print the planned tuples without launching.
# Each call to run_agentic.sh skips runs that already have conversation.jsonl.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCENARIOS=${SCENARIOS:-"scenario_1a scenario_1b scenario_2 scenario_3"}
CONDITIONS=${CONDITIONS:-"no-spec julia epiaware"}
PARAPHRASES=${PARAPHRASES:-"1 3 4 5"}    # slot 02 intentionally absent
VARIANTS=${VARIANTS:-"canonical"}
REPLICATES=${REPLICATES:-"1 2 3 4 5 6 7 8 9 10"}
MODELS=${MODELS:-"claude-sonnet-4-6"}
RUNS_PER_CELL=${RUNS_PER_CELL:-1}

count_planned=0
for sc in $SCENARIOS; do for c in $CONDITIONS; do for par in $PARAPHRASES; do
    for var in $VARIANTS; do for rep in $REPLICATES; do
        for model in $MODELS; do for run in $(seq 1 "$RUNS_PER_CELL"); do
            count_planned=$((count_planned + 1))
        done; done
    done; done
done; done; done

echo "Planned tuples: $count_planned"
echo "  scenarios:   $SCENARIOS"
echo "  conditions:  $CONDITIONS"
echo "  paraphrases: $PARAPHRASES"
echo "  variants:    $VARIANTS"
echo "  replicates:  $REPLICATES"
echo "  models:      $MODELS"
echo "  runs/cell:   $RUNS_PER_CELL"
echo ""

if [ -n "${PRINT_ONLY:-}" ]; then
    for sc in $SCENARIOS; do for c in $CONDITIONS; do for par in $PARAPHRASES; do
        for var in $VARIANTS; do for rep in $REPLICATES; do
            for model in $MODELS; do for run in $(seq 1 "$RUNS_PER_CELL"); do
                echo "$sc $c $par $var $rep $model $run"
            done; done
        done; done
    done; done; done
    exit 0
fi

i=0
for sc in $SCENARIOS; do for c in $CONDITIONS; do for par in $PARAPHRASES; do
    for var in $VARIANTS; do for rep in $REPLICATES; do
        for model in $MODELS; do for run in $(seq 1 "$RUNS_PER_CELL"); do
            i=$((i + 1))
            echo "[$i/$count_planned] $sc / $c / par_$par / $var / rep_$rep / $model / run_$run"
            "$SCRIPT_DIR/run_agentic.sh" "$sc" "$c" "$par" "$var" "$rep" "$model" "$run" \
                > /dev/null 2>&1 || echo "  FAILED — check $sc/$c/par_$par/$var/rep_$rep/$model/run_$run"
        done; done
    done; done
done; done; done

echo "Study complete."
