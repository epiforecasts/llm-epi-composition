#!/bin/bash
# Run agentic evaluation using Claude Code
# Usage: ./run_agentic.sh <scenario> <condition> <model> <run_number>
# Example: ./run_agentic.sh scenario_1a r claude-sonnet-4-20250514 1
#
# Runs Claude in an isolated temp directory containing only the prompt and
# data files, preventing access to reference solutions, other runs, or the
# study design.

set -e

SCENARIO=$1
CONDITION=$2
MODEL=$3
RUN_NUM=$4

if [ -z "$SCENARIO" ] || [ -z "$CONDITION" ] || [ -z "$MODEL" ] || [ -z "$RUN_NUM" ]; then
    echo "Usage: $0 <scenario> <condition> <model> <run_number>"
    echo "Example: $0 scenario_1a r claude-sonnet-4-20250514 1"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RUN_DIR="$PROJECT_DIR/runs/$SCENARIO/$CONDITION/$MODEL/run_$(printf '%02d' $RUN_NUM)"
PROMPT_FILE="$PROJECT_DIR/prompts/$SCENARIO/$CONDITION.md"

echo "=========================================="
echo "Agentic Evaluation Run"
echo "=========================================="
echo "Scenario: $SCENARIO"
echo "Condition: $CONDITION"
echo "Model: $MODEL"
echo "Run: $RUN_NUM"
echo "Run directory: $RUN_DIR"
echo "Prompt: $PROMPT_FILE"
echo ""

# Check prompt exists
if [ ! -f "$PROMPT_FILE" ]; then
    echo "ERROR: Prompt file not found: $PROMPT_FILE"
    exit 1
fi

# Create final output directory
mkdir -p "$RUN_DIR"

# Create isolated temp directory for the run
# This prevents Claude from exploring the repo, reference solutions, or other runs
WORK_DIR=$(mktemp -d)
trap "rm -rf $WORK_DIR" EXIT

echo "Isolated working directory: $WORK_DIR"

# Copy only the data and prompt into the isolated directory
mkdir -p "$WORK_DIR/data"
cp "$PROJECT_DIR/data/cases.csv" "$WORK_DIR/data/"
cp "$PROJECT_DIR/data/cases_dow.csv" "$WORK_DIR/data/"
cp "$PROJECT_DIR/data/observations.csv" "$WORK_DIR/data/"
cp "$PROMPT_FILE" "$WORK_DIR/prompt.md"

# For Julia/EpiAware conditions: provide a pre-configured Julia environment
# so package installation doesn't break or stall the run
JULIA_ENV="$PROJECT_DIR/evaluation/julia_env"
if [ -f "$JULIA_ENV/Project.toml" ] && [[ "$CONDITION" == "julia" || "$CONDITION" == "epiaware" ]]; then
    cp "$JULIA_ENV/Project.toml" "$WORK_DIR/"
    cp "$JULIA_ENV/Manifest.toml" "$WORK_DIR/"
    echo "Julia environment provided in working directory"
fi

# Record metadata in both locations
cat > "$WORK_DIR/metadata.json" << EOF
{
    "scenario": "$SCENARIO",
    "condition": "$CONDITION",
    "model": "$MODEL",
    "run_number": $RUN_NUM,
    "start_time": "$(date -Iseconds)",
    "prompt_file": "$PROMPT_FILE"
}
EOF

echo "Starting Claude Code..."
echo ""

cd "$WORK_DIR"

# Run Claude Code in the isolated directory
# --print: non-interactive mode
# --dangerously-skip-permissions: allow code execution
# --model: specify the model
# --verbose: required for stream-json output
claude --print \
    --dangerously-skip-permissions \
    --model "$MODEL" \
    --verbose \
    --output-format stream-json \
    "$(cat prompt.md)" \
    > conversation.jsonl 2> error.log

# Record end time
END_TIME=$(date -Iseconds)
echo "End time: $END_TIME"

# Update metadata with end time
python3 -c "
import json
with open('metadata.json', 'r') as f:
    meta = json.load(f)
meta['end_time'] = '$END_TIME'
with open('metadata.json', 'w') as f:
    json.dump(meta, f, indent=2)
"

# Copy all results back to the run directory
cp -r "$WORK_DIR"/* "$RUN_DIR/"

echo ""
echo "Run complete. Output saved to: $RUN_DIR"
echo "Files created:"
ls -la "$RUN_DIR"
