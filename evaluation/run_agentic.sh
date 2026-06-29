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
# List PIDs whose cwd is WORK_DIR. Catches descendants that didn't include
# the path in their argv (e.g. `julia --project=. script.jl` invoked from
# inside the work dir).
pids_in_workdir() {
    for d in /proc/[0-9]*/cwd; do
        if [ "$(readlink "$d" 2>/dev/null)" = "$WORK_DIR" ]; then
            basename "$(dirname "$d")"
        fi
    done
}
cleanup() {
    local pids
    pids=$(pids_in_workdir)
    if [ -n "$pids" ]; then
        kill -9 $pids 2>/dev/null || true
    fi
    # Also catch anything that referenced WORK_DIR by name in argv (covers
    # the case where the cwd has already been swapped out).
    pkill -9 -f "$WORK_DIR" 2>/dev/null || true
    rm -rf "$WORK_DIR"
}
trap cleanup EXIT
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

# Optional warm Julia daemon. Set USE_DAEMON=1 to pre-launch a DaemonMode server
# in WORK_DIR with the env's heavy packages already loaded, plus a jrun wrapper
# the agent can use instead of bare `julia` to avoid cold-start recompilation.
DAEMON_PORT=""
if [ "${USE_DAEMON:-0}" = "1" ] && [[ "$CONDITION" == "julia" || "$CONDITION" == "epiaware" ]]; then
    DAEMON_PORT=$((30000 + RANDOM % 30000))
    PRELOAD_PKGS="using EpiAware, Turing, Distributions, DataFrames, CSV, ReverseDiff, LogDensityProblemsAD, MCMCChains, Pathfinder"
    nohup julia --startup-file=no --project="$WORK_DIR" -e \
        "$PRELOAD_PKGS; using DaemonMode; serve($DAEMON_PORT)" \
        > "$WORK_DIR/daemon.log" 2>&1 &
    DAEMON_PID=$!
    echo "Julia daemon launched (PID $DAEMON_PID, port $DAEMON_PORT); waiting for ready..."
    for i in $(seq 1 90); do
        if ss -tln 2>/dev/null | grep -q ":$DAEMON_PORT "; then
            echo "  daemon ready after ${i}s"
            break
        fi
        sleep 2
    done
    if ! ss -tln 2>/dev/null | grep -q ":$DAEMON_PORT "; then
        echo "WARNING: daemon did not bind within 180s; continuing without it"
        DAEMON_PORT=""
    fi

    if [ -n "$DAEMON_PORT" ]; then
        cat > "$WORK_DIR/jrun" <<EOF
#!/bin/bash
# Run a Julia script via the warm DaemonMode daemon listening on $DAEMON_PORT.
# Uses the local project so DaemonMode (the client side) is on the load path.
exec julia --startup-file=no --project="$WORK_DIR" -e 'using DaemonMode; runargs($DAEMON_PORT)' "\$@"
EOF
        chmod +x "$WORK_DIR/jrun"
        cat >> "$WORK_DIR/prompt.md" <<'EOF'

---

## Warm Julia daemon available

A Julia daemon is running in this working directory with EpiAware, Turing, Distributions, DataFrames, CSV, ReverseDiff, LogDensityProblemsAD, MCMCChains, and Pathfinder already loaded. To execute a Julia script, run `./jrun script.jl` instead of `julia script.jl` — it forwards to the warm daemon and starts in milliseconds rather than paying the 30–60-second compile cost on every invocation. Use `./jrun` for every Julia run, including quick syntax/probe scripts; only fall back to bare `julia` if `./jrun` fails.
EOF
    fi
fi

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
    "use_daemon":   ${USE_DAEMON:-0},
    "daemon_port":  ${DAEMON_PORT:-null},
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

# Retry loop: if the agent ended its response before writing outputs/rt_estimates.csv,
# resume the same session and prompt it to complete the run. Caps at MAX_RETRIES.
# Before triggering a retry, give any backgrounded inference still running in
# WORK_DIR up to POST_AGENT_WAIT_MIN minutes to complete on its own — the agent
# may have launched a slow sampler and exited before it finished. This avoids
# killing inference that would otherwise succeed.
MAX_RETRIES=${MAX_RETRIES:-5}
POST_AGENT_WAIT_MIN=${POST_AGENT_WAIT_MIN:-60}
retry_count=0
post_agent_waits=0
CONTINUATION_PROMPT="The file outputs/rt_estimates.csv does not exist or is empty. The run is not complete. Continue the analysis: if an inference script is still running, wait for it to finish (set a long Bash timeout, e.g. 600000ms, and keep polling if the call has been moved to the background); if it failed or was never launched to completion, fix and re-run it. Only end your response once outputs/rt_estimates.csv exists on disk and contains the required columns."

while [ ! -s "outputs/rt_estimates.csv" ] && [ "$retry_count" -lt "$MAX_RETRIES" ]; do
    # If the agent has backgrounded inference and exited, give that process
    # up to POST_AGENT_WAIT_MIN minutes to finish before consuming a retry.
    if [ -n "$(pids_in_workdir)" ]; then
        echo "Output missing but processes active in WORK_DIR; waiting up to ${POST_AGENT_WAIT_MIN}m for inference to finish"
        post_agent_waits=$((post_agent_waits + 1))
        for _ in $(seq 1 $((POST_AGENT_WAIT_MIN * 2))); do
            sleep 30
            [ -s "outputs/rt_estimates.csv" ] && break
            [ -z "$(pids_in_workdir)" ] && break
        done
        [ -s "outputs/rt_estimates.csv" ] && break
    fi
    retry_count=$((retry_count + 1))
    SESSION_ID=$(python3 -c "
import json, sys
for fname in ['conversation.jsonl'] + [f'conversation_retry_{i}.jsonl' for i in range(1, $retry_count)]:
    try:
        with open(fname) as f:
            for line in f:
                m = json.loads(line)
                if 'session_id' in m and m['session_id']:
                    print(m['session_id'])
                    sys.exit(0)
    except FileNotFoundError:
        pass
" 2>/dev/null)
    if [ -z "$SESSION_ID" ]; then
        echo "Retry $retry_count: could not extract session_id; aborting retries"
        break
    fi
    echo "Retry $retry_count: outputs/rt_estimates.csv missing; resuming session $SESSION_ID"
    claude --print \
        --dangerously-skip-permissions \
        --resume "$SESSION_ID" \
        --model "$MODEL" \
        --max-turns 100 \
        --verbose \
        --output-format stream-json \
        "$CONTINUATION_PROMPT" \
        > "conversation_retry_${retry_count}.jsonl" 2>> error.log || true
done

END_TIME=$(date -Iseconds)
echo "End time: $END_TIME"
if [ -s "outputs/rt_estimates.csv" ]; then
    echo "Output file present (retries used: $retry_count)"
else
    echo "WARNING: outputs/rt_estimates.csv still missing or empty after $retry_count retries"
fi

python3 -c "
import json
with open('metadata.json', 'r') as f:
    meta = json.load(f)
meta['end_time'] = '$END_TIME'
meta['retry_count'] = $retry_count
meta['post_agent_waits'] = $post_agent_waits
meta['output_present'] = $([ -s outputs/rt_estimates.csv ] && echo True || echo False)
with open('metadata.json', 'w') as f:
    json.dump(meta, f, indent=2)
"

# Copy all results back to the persistent run directory.
cp -r "$WORK_DIR"/* "$RUN_DIR/"

echo ""
echo "Run complete. Output saved to: $RUN_DIR"
echo "Files created:"
ls -la "$RUN_DIR"
