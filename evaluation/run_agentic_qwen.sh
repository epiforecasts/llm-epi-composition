#!/bin/bash
# Run agentic evaluation using Qwen3-Coder-30B on the LSHTM HPC.
#
# Mirrors evaluation/run_agentic.sh, but ships the isolated working directory
# to the cluster and submits a SLURM job that runs the Qwen Code CLI against
# a vLLM server. Retry logic and output collection follow the Claude version.
#
# Prerequisites (per ~/code/dotfiles/lshtm-local-llm-stack.md):
#   - Passwordless SSH to hpclogin (ssh alias).
#   - Shared LLM stack installed at /home/shared/llm/ on the cluster.
#   - Reference sbatch script at /home/shared/llm/bin/_agent_job.sbatch that
#     starts vLLM, waits for readiness, invokes `qwen --yolo -m qwen3-coder`,
#     and preserves the run's working directory.
#
# Usage:
#   ./run_agentic_qwen.sh <scenario> <condition> <paraphrase> <variant> <replicate> <run_number>
# Example:
#   ./run_agentic_qwen.sh scenario_1a epiaware 1 canonical 1 1
#
# Note: MODEL is not a parameter here — Qwen3-Coder-30B is the only supported
# model on this harness. The run_agentic.sh MODEL parameter is elided.

set -u

SCENARIO=$1
CONDITION=$2
PARAPHRASE=$3
VARIANT=$4
REPLICATE=$5
RUN_NUM=$6

if [ -z "${SCENARIO:-}" ] || [ -z "${CONDITION:-}" ] || [ -z "${PARAPHRASE:-}" ] || \
   [ -z "${VARIANT:-}" ]  || [ -z "${REPLICATE:-}" ]  || [ -z "${RUN_NUM:-}" ]; then
    echo "Usage: $0 <scenario> <condition> <paraphrase> <variant> <replicate> <run_number>"
    exit 1
fi

MODEL=qwen3-coder-30b
MODEL_TAG=qwen3-coder-30b   # subdir name in runs/

PAR_PADDED=$(printf '%02d' "$PARAPHRASE")
REP_PADDED=$(printf '%02d' "$REPLICATE")
RUN_PADDED=$(printf '%02d' "$RUN_NUM")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SIM_DATA_DIR="$PROJECT_DIR/simulations/$VARIANT/rep_$REP_PADDED/data"
PROMPT_FILE="$PROJECT_DIR/prompts/$SCENARIO/paraphrases/$CONDITION/$PAR_PADDED.md"
RUN_DIR="$PROJECT_DIR/runs/$SCENARIO/$CONDITION/par_$PAR_PADDED/$VARIANT/rep_$REP_PADDED/$MODEL_TAG/run_$RUN_PADDED"

# Cluster settings — override via env if the layout changes.
: "${QWEN_HOST:=hpclogin}"
: "${QWEN_REMOTE_ROOT:=/home/shared/llm/runs/llm-epi-composition}"
: "${QWEN_SBATCH:=/home/shared/llm/bin/_agent_job.sbatch}"

echo "=========================================="
echo "Agentic Evaluation Run (Qwen / LSHTM)"
echo "=========================================="
echo "Scenario:    $SCENARIO"
echo "Condition:   $CONDITION"
echo "Paraphrase:  $PARAPHRASE"
echo "Variant:     $VARIANT"
echo "Replicate:   $REPLICATE"
echo "Model:       $MODEL_TAG"
echo "Run:         $RUN_NUM"
echo "Run dir:     $RUN_DIR"
echo "Prompt:      $PROMPT_FILE"
echo "Sim data:    $SIM_DATA_DIR"
echo "Cluster:     $QWEN_HOST:$QWEN_REMOTE_ROOT"
echo ""

if [ ! -f "$PROMPT_FILE" ]; then
    echo "ERROR: prompt file not found: $PROMPT_FILE"
    exit 1
fi
if [ ! -d "$SIM_DATA_DIR" ]; then
    echo "ERROR: simulation data directory not found: $SIM_DATA_DIR"
    exit 1
fi

if [ -f "$RUN_DIR/conversation.jsonl" ]; then
    echo "SKIP: run already complete at $RUN_DIR"
    exit 0
fi

mkdir -p "$RUN_DIR"

# Build the local staging directory (identical to the Claude harness's WORK_DIR
# layout so the two produce comparable per-run trees).
STAGE=$(mktemp -d)
cleanup_stage() {
    /bin/rm -rf "$STAGE"
}
trap cleanup_stage EXIT

mkdir -p "$STAGE/data" "$STAGE/outputs"
case "$SCENARIO" in
    scenario_1a|scenario_1b|scenario_2)
        cp "$SIM_DATA_DIR/cases.csv" "$STAGE/data/"
        ;;
    scenario_3)
        cp "$SIM_DATA_DIR/cases.csv"            "$STAGE/data/"
        cp "$SIM_DATA_DIR/hospitalisations.csv" "$STAGE/data/"
        cp "$SIM_DATA_DIR/deaths.csv"           "$STAGE/data/"
        ;;
    *)
        echo "ERROR: unknown scenario '$SCENARIO'"
        exit 1
        ;;
esac

cp "$PROMPT_FILE" "$STAGE/prompt.md"

JULIA_ENV="$PROJECT_DIR/evaluation/julia_env"
if [ -f "$JULIA_ENV/Project.toml" ] && [[ "$CONDITION" == "julia" || "$CONDITION" == "epiaware" ]]; then
    cp "$JULIA_ENV/Project.toml"  "$STAGE/"
    cp "$JULIA_ENV/Manifest.toml" "$STAGE/"
fi

case "$CONDITION" in
    julia)    cp "$PROJECT_DIR/prompts/turing_api_docs.md"   "$STAGE/turing_api_docs.md"   ;;
    epiaware) cp "$PROJECT_DIR/prompts/epiaware_api_docs.md" "$STAGE/epiaware_docs.md"     ;;
    no-spec)  : ;;
    *)  echo "ERROR: unknown condition '$CONDITION'"; exit 1 ;;
esac

cat > "$STAGE/metadata.json" << EOF
{
    "scenario":     "$SCENARIO",
    "condition":    "$CONDITION",
    "paraphrase":   $PARAPHRASE,
    "variant":      "$VARIANT",
    "replicate":    $REPLICATE,
    "model":        "$MODEL_TAG",
    "run_number":   $RUN_NUM,
    "start_time":   "$(date -Iseconds)",
    "prompt_file":  "$PROMPT_FILE",
    "sim_data_dir": "$SIM_DATA_DIR",
    "cluster":      "$QWEN_HOST",
    "harness":      "run_agentic_qwen.sh"
}
EOF

# Unique remote directory for this run.
REMOTE_TAG="${SCENARIO}_${CONDITION}_par${PAR_PADDED}_${VARIANT}_rep${REP_PADDED}_run${RUN_PADDED}_$$"
REMOTE_DIR="$QWEN_REMOTE_ROOT/$REMOTE_TAG"

echo "Staging → $QWEN_HOST:$REMOTE_DIR"
ssh "$QWEN_HOST" "mkdir -p $REMOTE_DIR" || { echo "ERROR: could not create remote dir"; exit 1; }
rsync -az --delete "$STAGE/" "$QWEN_HOST:$REMOTE_DIR/" || { echo "ERROR: rsync up failed"; exit 1; }

# Submit the SLURM job. _agent_job.sbatch is expected to:
#   * accept the working directory as its first argument (or via $WORK_DIR
#     inherited from --export=ALL);
#   * start vLLM serving qwen3-coder on the same compute node;
#   * invoke `qwen --yolo -m qwen3-coder --openai-base-url http://127.0.0.1:8000/v1 ...`
#     with the contents of prompt.md as the initial instruction;
#   * preserve conversation logs and any produced outputs/rt_estimates.csv
#     in the working directory.
echo "Submitting SLURM job …"
JOBID=$(ssh "$QWEN_HOST" \
    "sbatch --parsable --export=ALL,WORK_DIR=$REMOTE_DIR $QWEN_SBATCH $REMOTE_DIR" ) || {
    echo "ERROR: sbatch submission failed"
    exit 1
}
echo "Submitted job $JOBID; polling status every 60s"

# Poll until the job leaves the queue.
while true; do
    STATUS=$(ssh "$QWEN_HOST" "squeue -h -j $JOBID -o '%T' 2>/dev/null" || true)
    if [ -z "$STATUS" ]; then
        break
    fi
    echo "  [$JOBID] $STATUS"
    sleep 60
done

# Fetch the exit state for logging.
EXIT_STATE=$(ssh "$QWEN_HOST" "sacct -j $JOBID -o State -n -P | head -1" || echo "UNKNOWN")
echo "Job $JOBID finished with state: $EXIT_STATE"

# Rsync everything back and record the retry count based on presence of the
# output CSV. Retries in the Qwen harness are managed inside the sbatch job
# (the LSHTM stack's _agent_job.sbatch has its own retry loop analogous to
# run_agentic.sh's --resume loop). If the output file is still missing we
# record output_present=false and let the study driver mark this cell as
# failed; a re-run under a new $RUN_NUM is the equivalent of retrying.
echo "Fetching results …"
rsync -az "$QWEN_HOST:$REMOTE_DIR/" "$STAGE/" || echo "WARNING: rsync back partial"

END_TIME=$(date -Iseconds)
if [ -s "$STAGE/outputs/rt_estimates.csv" ]; then
    OUTPUT_PRESENT=true
    echo "Output file present."
else
    OUTPUT_PRESENT=false
    echo "WARNING: outputs/rt_estimates.csv missing or empty."
fi

python3 - "$STAGE/metadata.json" "$END_TIME" "$OUTPUT_PRESENT" "$JOBID" "$EXIT_STATE" <<'PY'
import json, sys
meta_path, end_time, output_present, jobid, slurm_state = sys.argv[1:]
m = json.load(open(meta_path))
m['end_time'] = end_time
m['output_present'] = (output_present == "true")
m['slurm_job_id'] = jobid
m['slurm_exit_state'] = slurm_state
json.dump(m, open(meta_path, "w"), indent=2)
PY

cp -r "$STAGE"/* "$RUN_DIR/"

# Best-effort remote cleanup so the shared LLM area doesn't fill up.
ssh "$QWEN_HOST" "rm -rf $REMOTE_DIR" || true

echo ""
echo "Run complete. Output saved to: $RUN_DIR"
ls -la "$RUN_DIR"
