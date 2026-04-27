#!/bin/bash
# Run agentic evaluation using Claude Code on a simulated dataset.
#
# Usage:
#   ./run_agentic.sh <scenario> <condition> <paraphrase> <variant> <replicate> <model> <run_number>
# Example:
#   ./run_agentic.sh scenario_1a epiaware 1 canonical 1 claude-sonnet-4-6 1
#
# Runs Claude in an isolated mktemp working directory containing only:
#   - the paraphrase of the prompt at prompts/<scenario>/paraphrases/<condition>/<NN>.md
#   - the relevant data files from simulations/<variant>/rep_<rr>/data/
#       (scenarios 1a/1b/2 → cases.csv only;  scenario 3 → all three streams)
#   - the docs bundle for the condition
#       (julia → turing_api_docs.md;  epiaware → epiaware_docs.md;  no-spec → none)
#   - a pre-resolved Julia env (Project.toml + Manifest.toml) for julia/epiaware
# The truth/ subdirectory is never copied; the agent never sees the true Rt.

set -e

SCENARIO=$1
CONDITION=$2
PARAPHRASE=$3
VARIANT=$4
REPLICATE=$5
MODEL=$6
RUN_NUM=$7

if [ -z "$SCENARIO" ] || [ -z "$CONDITION" ] || [ -z "$PARAPHRASE" ] || \
   [ -z "$VARIANT" ] || [ -z "$REPLICATE" ] || [ -z "$MODEL" ] || [ -z "$RUN_NUM" ]; then
    echo "Usage: $0 <scenario> <condition> <paraphrase> <variant> <replicate> <model> <run_number>"
    echo "Example: $0 scenario_1a epiaware 1 canonical 1 claude-sonnet-4-6 1"
    exit 1
fi

PAR_PADDED=$(printf '%02d' "$PARAPHRASE")
REP_PADDED=$(printf '%02d' "$REPLICATE")
RUN_PADDED=$(printf '%02d' "$RUN_NUM")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SIM_DATA_DIR="$PROJECT_DIR/simulations/$VARIANT/rep_$REP_PADDED/data"
PROMPT_FILE="$PROJECT_DIR/prompts/$SCENARIO/paraphrases/$CONDITION/$PAR_PADDED.md"
RUN_DIR="$PROJECT_DIR/runs/$SCENARIO/$CONDITION/par_$PAR_PADDED/$VARIANT/rep_$REP_PADDED/$MODEL/run_$RUN_PADDED"

echo "=========================================="
echo "Agentic Evaluation Run"
echo "=========================================="
echo "Scenario:    $SCENARIO"
echo "Condition:   $CONDITION"
echo "Paraphrase:  $PARAPHRASE"
echo "Variant:     $VARIANT"
echo "Replicate:   $REPLICATE"
echo "Model:       $MODEL"
echo "Run:         $RUN_NUM"
echo "Run dir:     $RUN_DIR"
echo "Prompt:      $PROMPT_FILE"
echo "Sim data:    $SIM_DATA_DIR"
echo ""

# Validate inputs
if [ ! -f "$PROMPT_FILE" ]; then
    echo "ERROR: Prompt file not found: $PROMPT_FILE"
    exit 1
fi
if [ ! -d "$SIM_DATA_DIR" ]; then
    echo "ERROR: Simulation data directory not found: $SIM_DATA_DIR"
    exit 1
fi

# Skip if run already complete
if [ -f "$RUN_DIR/conversation.jsonl" ]; then
    echo "SKIP: Run already complete at $RUN_DIR"
    exit 0
fi

mkdir -p "$RUN_DIR"

# Isolated working directory — Claude never sees the broader repo
WORK_DIR=$(mktemp -d)
trap "rm -rf $WORK_DIR" EXIT
echo "Isolated working directory: $WORK_DIR"

# Copy the relevant streams only.
# Scenarios 1a/1b/2 are single-stream (cases); scenario 3 is multi-stream.
mkdir -p "$WORK_DIR/data"
case "$SCENARIO" in
    scenario_1a|scenario_1b|scenario_2)
        cp "$SIM_DATA_DIR/cases.csv" "$WORK_DIR/data/"
        ;;
    scenario_3)
        cp "$SIM_DATA_DIR/cases.csv"            "$WORK_DIR/data/"
        cp "$SIM_DATA_DIR/hospitalisations.csv" "$WORK_DIR/data/"
        cp "$SIM_DATA_DIR/deaths.csv"           "$WORK_DIR/data/"
        ;;
    *)
        echo "ERROR: Unknown scenario '$SCENARIO'"
        exit 1
        ;;
esac

cp "$PROMPT_FILE" "$WORK_DIR/prompt.md"
mkdir -p "$WORK_DIR/outputs"

# Provide a pre-resolved Julia env for julia / epiaware so package install does
# not stall the run.
JULIA_ENV="$PROJECT_DIR/evaluation/julia_env"
if [ -f "$JULIA_ENV/Project.toml" ] && [[ "$CONDITION" == "julia" || "$CONDITION" == "epiaware" ]]; then
    cp "$JULIA_ENV/Project.toml"  "$WORK_DIR/"
    cp "$JULIA_ENV/Manifest.toml" "$WORK_DIR/"
    echo "Julia environment provided in working directory"
fi

# Provide condition-specific API documentation.
case "$CONDITION" in
    julia)
        TURING_DOCS="$PROJECT_DIR/prompts/turing_api_docs.md"
        if [ -f "$TURING_DOCS" ]; then
            cp "$TURING_DOCS" "$WORK_DIR/turing_api_docs.md"
            echo "Turing.jl API documentation provided in working directory"
        else
            echo "WARNING: $TURING_DOCS not found — julia condition will be missing its docs bundle"
        fi
        ;;
    epiaware)
        EPIAWARE_DOCS="$PROJECT_DIR/prompts/epiaware_api_docs.md"
        if [ -f "$EPIAWARE_DOCS" ]; then
            cp "$EPIAWARE_DOCS" "$WORK_DIR/epiaware_docs.md"
            echo "EpiAware API documentation provided in working directory"
        else
            echo "WARNING: $EPIAWARE_DOCS not found — epiaware condition will be missing its docs bundle"
        fi
        ;;
    no-spec)
        # No docs bundle by design.
        ;;
    *)
        echo "ERROR: Unknown condition '$CONDITION'"
        exit 1
        ;;
esac

# Metadata
cat > "$WORK_DIR/metadata.json" << EOF
{
    "scenario":     "$SCENARIO",
    "condition":    "$CONDITION",
    "paraphrase":   $PARAPHRASE,
    "variant":      "$VARIANT",
    "replicate":    $REPLICATE,
    "model":        "$MODEL",
    "run_number":   $RUN_NUM,
    "start_time":   "$(date -Iseconds)",
    "prompt_file":  "$PROMPT_FILE",
    "sim_data_dir": "$SIM_DATA_DIR"
}
EOF

echo "Starting Claude Code..."
echo ""

cd "$WORK_DIR"

claude --print \
    --dangerously-skip-permissions \
    --model "$MODEL" \
    --max-turns 200 \
    --verbose \
    --output-format stream-json \
    "$(cat prompt.md)" \
    > conversation.jsonl 2> error.log || true

END_TIME=$(date -Iseconds)
echo "End time: $END_TIME"

python3 -c "
import json
with open('metadata.json', 'r') as f:
    meta = json.load(f)
meta['end_time'] = '$END_TIME'
with open('metadata.json', 'w') as f:
    json.dump(meta, f, indent=2)
"

# Copy all results back to the persistent run directory.
cp -r "$WORK_DIR"/* "$RUN_DIR/"

echo ""
echo "Run complete. Output saved to: $RUN_DIR"
echo "Files created:"
ls -la "$RUN_DIR"
