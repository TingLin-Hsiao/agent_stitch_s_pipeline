#!/bin/bash
#SBATCH --job-name=stitch_pipeline
#SBATCH --output=stitch_pipeline_%j.log
#SBATCH --error=stitch_pipeline_%j.err
#SBATCH --account=mst115022
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=1500G
#SBATCH --time=4:00:00

set -euo pipefail

cd "/work/u1007247/agent_stitch_s_pipeline"
export PYTHONPATH="/work/u1007247/agent_stitch_s_pipeline:${PYTHONPATH:-}"

MODE="${MODE:-${1:-generate}}"
case "$MODE" in
    generate|eval|all) ;;
    *)
        echo "Invalid MODE=$MODE. Use MODE=generate, MODE=eval, or MODE=all."
        exit 1
        ;;
esac

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
TOTAL_GPUS=$((SLURM_JOB_NUM_NODES * GPUS_PER_NODE))
VLLM_CONCURRENCY="${VLLM_CONCURRENCY:-$((TOTAL_GPUS / TENSOR_PARALLEL_SIZE))}"
VLLM_BATCH_SIZE="${VLLM_BATCH_SIZE:-}"
USER_VLLM_BATCH_SIZE="$VLLM_BATCH_SIZE"
USER_MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-}"
MAX_CONCURRENT_BATCHES="${MAX_CONCURRENT_BATCHES:-4}"
PREVIEW_ROWS="${PREVIEW_ROWS:-100}"
RAW_INPUT_BLOCKS="${RAW_INPUT_BLOCKS:-$((VLLM_CONCURRENCY * 16))}"
LLM_INPUT_BLOCKS="${LLM_INPUT_BLOCKS:-$((VLLM_CONCURRENCY * 8))}"
GPU_PREFLIGHT="${GPU_PREFLIGHT:-1}"
MODEL_ID="${MODEL_ID:-google/gemma-4-26B-A4B-it}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"

DATA="${DATA:-voidful/agent-sft}"
OUT="${OUT:-./out_full}"
ONLY_IDS="${ONLY_IDS:-}"
GENERATE_BACKEND="${GENERATE_BACKEND:-local}"
EVAL_INPUT="${EVAL_INPUT:-}"
EVAL_OUT="${EVAL_OUT:-./out_eval_scores_gemma}"
EVAL_BACKEND="${EVAL_BACKEND:-local}"
GENERATE_MODEL="${GENERATE_MODEL:-${GENERATE_API_MODEL:-${API_MODEL:-$MODEL_ID}}}"
EVAL_MODEL="${EVAL_MODEL:-${EVAL_API_MODEL:-${API_MODEL:-$MODEL_ID}}}"
API_KEY="${API_KEY:-${OPENAI_API_KEY:-${GEMINI_API_KEY:-}}}"
API_BASE_URL="${API_BASE_URL:-${OPENAI_BASE_URL:-}}"
API_CONCURRENCY="${API_CONCURRENCY:-4}"
API_TIMEOUT="${API_TIMEOUT:-180}"
API_RETRIES="${API_RETRIES:-3}"
GENERATE_MAX_TOKENS="${GENERATE_MAX_TOKENS:-${GENERATE_API_MAX_TOKENS:-4096}}"
MAX_ROWS="${MAX_ROWS:-}"

case "$GENERATE_BACKEND" in
    local|api) ;;
    *)
        echo "Invalid GENERATE_BACKEND=$GENERATE_BACKEND. Use GENERATE_BACKEND=local or GENERATE_BACKEND=api."
        exit 1
        ;;
esac

case "$EVAL_BACKEND" in
    local|api) ;;
    *)
        echo "Invalid EVAL_BACKEND=$EVAL_BACKEND. Use EVAL_BACKEND=local or EVAL_BACKEND=api."
        exit 1
        ;;
esac

NEED_RAY=1
if [[ "$MODE" == "generate" && "$GENERATE_BACKEND" == "api" ]]; then
    NEED_RAY=0
elif [[ "$MODE" == "eval" && "$EVAL_BACKEND" == "api" ]]; then
    NEED_RAY=0
elif [[ "$MODE" == "all" && "$GENERATE_BACKEND" == "api" && "$EVAL_BACKEND" == "api" ]]; then
    NEED_RAY=0
fi

JOB_TMP_ROOT="${JOB_TMP_ROOT:-/work/u1007247/tmp/stitch_pipeline/${SLURM_JOB_ID:-manual}}"
TMPDIR="${TMPDIR:-$JOB_TMP_ROOT/tmp}"
if [[ "$TMPDIR" == "/tmp" ]]; then
    TMPDIR="$JOB_TMP_ROOT/tmp"
fi
RAY_TMPDIR="${RAY_TMPDIR:-$JOB_TMP_ROOT/ray}"
mkdir -p "$TMPDIR" "$RAY_TMPDIR"

if (( VLLM_CONCURRENCY < 1 )); then
    echo "Invalid VLLM_CONCURRENCY=$VLLM_CONCURRENCY. Check TOTAL_GPUS=$TOTAL_GPUS and TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE."
    exit 1
fi

export TENSOR_PARALLEL_SIZE
export VLLM_CONCURRENCY
export MAX_CONCURRENT_BATCHES
export PREVIEW_ROWS
export RAW_INPUT_BLOCKS
export LLM_INPUT_BLOCKS
export GPU_PREFLIGHT
export MODEL_ID
export MAX_MODEL_LEN
export TMPDIR
export RAY_TMPDIR
export API_KEY

cleanup() {
    if (( NEED_RAY == 0 )); then
        return
    fi
    echo "Shutting down Ray cluster..."
    if [[ -n "${nodes:-}" ]]; then
        for node in "${nodes_array[@]}"; do
            srun --overlap --nodes=1 --ntasks=1 -w "$node" ray stop --force || true
        done
    else
        ray stop --force || true
    fi
}

trap cleanup EXIT

echo "=================================================="
echo "Pipeline job started at: $(date)"
echo "MODE: $MODE"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes allocated: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: $GPUS_PER_NODE"
echo "Total GPUs: $TOTAL_GPUS"
echo "MODEL_ID: $MODEL_ID"
echo "TENSOR_PARALLEL_SIZE: $TENSOR_PARALLEL_SIZE"
echo "VLLM_CONCURRENCY: $VLLM_CONCURRENCY"
echo "MAX_CONCURRENT_BATCHES: $MAX_CONCURRENT_BATCHES"
echo "PREVIEW_ROWS: $PREVIEW_ROWS"
echo "RAW_INPUT_BLOCKS: $RAW_INPUT_BLOCKS"
echo "LLM_INPUT_BLOCKS: $LLM_INPUT_BLOCKS"
echo "GPU_PREFLIGHT: $GPU_PREFLIGHT"
echo "DATA: $DATA"
echo "OUT: $OUT"
echo "ONLY_IDS: ${ONLY_IDS:-<unset>}"
echo "GENERATE_BACKEND: $GENERATE_BACKEND"
echo "EVAL_INPUT: ${EVAL_INPUT:-<auto>}"
echo "EVAL_OUT: $EVAL_OUT"
echo "EVAL_BACKEND: $EVAL_BACKEND"
echo "GENERATE_MODEL: $GENERATE_MODEL"
echo "EVAL_MODEL: $EVAL_MODEL"
echo "API_BASE_URL: ${API_BASE_URL:-<auto>}"
echo "API_CONCURRENCY: $API_CONCURRENCY"
echo "MAX_ROWS: ${MAX_ROWS:-<unset>}"
echo "TMPDIR: $TMPDIR"
echo "RAY_TMPDIR: $RAY_TMPDIR"
echo "=================================================="

CONDA_ROOT="${CONDA_ROOT:-/work/envstack/apps/miniconda3/26.1.1}"
CUDA_ROOT="${CUDA_ROOT:-/work/envstack/apps/cuda/12.6}"

if ml load miniconda3 2>/dev/null; then
    echo "Loaded miniconda3 module."
elif [[ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
    echo "miniconda3 module unavailable; using $CONDA_ROOT."
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
else
    echo "Unable to initialize conda: miniconda3 module is unavailable and $CONDA_ROOT/etc/profile.d/conda.sh does not exist."
    exit 1
fi

export PATH="$CONDA_ROOT/bin:/home/u1007247/.local/bin:$PATH"

if ml load cuda/12.6 2>/dev/null; then
    echo "Loaded cuda/12.6 module."
elif [[ -d "$CUDA_ROOT" ]]; then
    echo "cuda/12.6 module unavailable; using $CUDA_ROOT."
    export CUDA_HOME="$CUDA_ROOT"
    export PATH="$CUDA_ROOT/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_ROOT/lib64:${LD_LIBRARY_PATH:-}"
else
    echo "Unable to initialize CUDA: cuda/12.6 module is unavailable and $CUDA_ROOT does not exist."
    exit 1
fi

if [[ -d "$HOME/.conda/envs/tp1/bin" ]]; then
    conda activate tp1
else
    echo "Conda env tp1 has no bin directory; using envstack Python at $CONDA_ROOT/bin."
fi

if (( NEED_RAY == 1 )); then
    : "${HF_TOKEN:?HF_TOKEN is not set. Submit with: sbatch --export=ALL,HF_TOKEN run_pipeline.sh}"
    echo "HF_TOKEN is set."
fi

if (( NEED_RAY == 1 )); then
    nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
    nodes_array=($nodes)
    head_node=${nodes_array[0]}
    head_node_ip=$(srun --overlap --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')

    port=6379
    ip_head=$head_node_ip:$port
    export ip_head
    export RAY_ADDRESS="$ip_head"
    echo "Ray head node: $head_node"
    echo "Ray head address: $ip_head"

    echo "Starting Ray head node on $head_node"
    srun --overlap --nodes=1 --ntasks=1 -w "$head_node" ray stop --force || true
    srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
        ray start --head \
        --node-ip-address="$head_node_ip" \
        --port="$port" \
        --num-gpus="$GPUS_PER_NODE" \
        --num-cpus="$SLURM_CPUS_PER_TASK" \
        --temp-dir="$RAY_TMPDIR" \
        --block &
    head_srun_pid=$!

    echo "Waiting for Ray head GCS..."
    for attempt in $(seq 1 60); do
        if ray status --address="$ip_head" >/dev/null 2>&1; then
            echo "Ray head is reachable."
            break
        fi
        if (( attempt == 60 )); then
            echo "Ray head did not become reachable at $ip_head"
            exit 1
        fi
        sleep 5
    done

    if (( SLURM_JOB_NUM_NODES > 1 )); then
        for worker_node in "${nodes_array[@]:1}"; do
            echo "Starting Ray worker on $worker_node"
            srun --overlap --nodes=1 --ntasks=1 -w "$worker_node" ray stop --force || true
            srun --overlap --nodes=1 --ntasks=1 -w "$worker_node" \
                ray start \
                --address="$ip_head" \
                --num-gpus="$GPUS_PER_NODE" \
                --num-cpus="$SLURM_CPUS_PER_TASK" \
                --temp-dir="$RAY_TMPDIR" \
                --block &
        done
    fi

    expected_nodes="$SLURM_JOB_NUM_NODES"
    echo "Waiting for $expected_nodes Ray node(s) to join..."
    for attempt in $(seq 1 60); do
        live_nodes=$(python3 - <<'PY'
import os
import ray

ray.init(address=os.environ["RAY_ADDRESS"], ignore_reinit_error=True, logging_level="ERROR")
print(sum(1 for node in ray.nodes() if node.get("Alive")))
ray.shutdown()
PY
)
        echo "Ray live nodes: $live_nodes/$expected_nodes"
        if (( live_nodes >= expected_nodes )); then
            break
        fi
        if (( attempt == 60 )); then
            echo "Only $live_nodes/$expected_nodes Ray nodes joined."
            exit 1
        fi
        sleep 5
    done

    ray status --address="$ip_head"
    echo "Ray cluster started successfully."

    if [[ "$GPU_PREFLIGHT" == "1" ]]; then
        echo "Running Ray GPU visibility preflight with $VLLM_CONCURRENCY GPU task(s)..."
        python3 - <<'PY'
import json
import os
import sys

import ray

expected = int(os.environ["VLLM_CONCURRENCY"])
ray.init(address=os.environ["RAY_ADDRESS"], ignore_reinit_error=True, logging_level="ERROR")


@ray.remote(num_gpus=1)
def check_gpu():
    import os
    import socket

    import torch

    return {
        "host": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_device_count": torch.cuda.device_count(),
    }


results = ray.get([check_gpu.remote() for _ in range(expected)])
ray.shutdown()

print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))

bad = [
    row
    for row in results
    if not row["torch_cuda_available"] or row["torch_cuda_device_count"] < 1
]

if bad:
    print("GPU preflight failed: at least one Ray GPU task cannot see CUDA.", file=sys.stderr)
    sys.exit(1)

hosts = sorted({row["host"] for row in results})
print(f"GPU preflight passed: {len(results)} task(s) across {len(hosts)} host(s): {hosts}")
PY
    fi
else
    echo "Skipping Ray cluster startup for API eval backend."
fi

find_latest_success() {
    python3 - "$OUT" <<'PY'
import sys
from pathlib import Path

out = Path(sys.argv[1])
candidates = [path for path in (out / "runs").glob("*/success") if path.is_dir()]
if not candidates:
    raise SystemExit(f"No success output found under {out / 'runs'}")
print(max(candidates, key=lambda path: path.stat().st_mtime))
PY
}

run_generate() {
    cmd=(python3 -u generate/pipeline.py --backend "$GENERATE_BACKEND" --data "$DATA" --out "$OUT")
    if [[ -n "$ONLY_IDS" ]]; then
        cmd+=(--only-ids "$ONLY_IDS")
    fi
    if [[ -n "$MAX_ROWS" && "$GENERATE_BACKEND" == "api" ]]; then
        cmd+=(--max-rows "$MAX_ROWS")
    fi
    if [[ "$GENERATE_BACKEND" == "api" ]]; then
        cmd+=(--model "$GENERATE_MODEL" --concurrency "$API_CONCURRENCY" --timeout "$API_TIMEOUT" --retries "$API_RETRIES" --max-tokens "$GENERATE_MAX_TOKENS")
        if [[ -n "$API_BASE_URL" ]]; then
            cmd+=(--base-url "$API_BASE_URL")
        fi
    else
        export VLLM_BATCH_SIZE="${GENERATE_VLLM_BATCH_SIZE:-${USER_VLLM_BATCH_SIZE:-64}}"
        export MAX_OUTPUT_TOKENS="${GENERATE_MAX_OUTPUT_TOKENS:-${USER_MAX_OUTPUT_TOKENS:-4096}}"
    fi
    echo "Running generation command: ${cmd[*]}"
    "${cmd[@]}"
}

run_eval() {
    local input_path="$1"
    cmd=(python3 -u eval/pipeline.py --backend "$EVAL_BACKEND" --input "$input_path" --out "$EVAL_OUT")
    if [[ -n "$MAX_ROWS" ]]; then
        cmd+=(--max-rows "$MAX_ROWS")
    fi
    if [[ "$EVAL_BACKEND" == "api" ]]; then
        cmd+=(--model "$EVAL_MODEL" --concurrency "$API_CONCURRENCY" --timeout "$API_TIMEOUT" --retries "$API_RETRIES")
        if [[ -n "$API_BASE_URL" ]]; then
            cmd+=(--base-url "$API_BASE_URL")
        fi
    else
        export VLLM_BATCH_SIZE="${EVAL_VLLM_BATCH_SIZE:-${USER_VLLM_BATCH_SIZE:-32}}"
        export MAX_OUTPUT_TOKENS="${EVAL_MAX_OUTPUT_TOKENS:-${USER_MAX_OUTPUT_TOKENS:-2048}}"
    fi
    echo "Running eval command: ${cmd[*]}"
    "${cmd[@]}"
}

case "$MODE" in
    generate)
        run_generate
        ;;
    eval)
        if [[ -z "$EVAL_INPUT" ]]; then
            echo "EVAL_INPUT is required when MODE=eval."
            exit 1
        fi
        run_eval "$EVAL_INPUT"
        ;;
    all)
        run_generate
        eval_input="${EVAL_INPUT:-$(find_latest_success)}"
        run_eval "$eval_input"
        ;;
esac

echo "=================================================="
echo "Pipeline job finished at: $(date)"
echo "=================================================="
