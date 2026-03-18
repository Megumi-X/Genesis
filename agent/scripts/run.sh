#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

MODEL="${MODEL:-gpt-5.4}"
CRITIC_MODEL="${CRITIC_MODEL:-}"
REASONING_EFFORT="${REASONING_EFFORT:-xhigh}"
CRITIC_REASONING_EFFORT="${CRITIC_REASONING_EFFORT:-}"
BACKEND="${BACKEND:-gpu}"
MAX_OPT_ROUNDS="${MAX_OPT_ROUNDS:-8}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-12}"
XML_MAX_ATTEMPTS="${XML_MAX_ATTEMPTS:-4}"
SAMPLE_EVERY_SEC="${SAMPLE_EVERY_SEC:-0.5}"
MAX_FRAMES="${MAX_FRAMES:-24}"
TIMEOUT_SEC="${TIMEOUT_SEC:-600}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-agent/runs/opt_multibody_suite/${RUN_TS}}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --critic-model)
      CRITIC_MODEL="$2"
      shift 2
      ;;
    --reasoning-effort)
      REASONING_EFFORT="$2"
      shift 2
      ;;
    --critic-reasoning-effort)
      CRITIC_REASONING_EFFORT="$2"
      shift 2
      ;;
    --gpu)
      BACKEND="gpu"
      shift
      ;;
    --cpu)
      BACKEND="cpu"
      shift
      ;;
    --backend)
      BACKEND="$2"
      shift 2
      ;;
    --run-root)
      RUN_ROOT="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--model MODEL] [--critic-model MODEL] [--reasoning-effort EFFORT] [--critic-reasoning-effort EFFORT] [--gpu|--cpu|--backend cpu|gpu] [--run-root PATH]" >&2
      exit 2
      ;;
  esac
done

if [[ "$BACKEND" != "cpu" && "$BACKEND" != "gpu" ]]; then
  echo "Invalid BACKEND: $BACKEND" >&2
  exit 2
fi

LOCAL_PYTHON=""
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  LOCAL_PYTHON="$ROOT_DIR/.venv/bin/python"
fi

if [[ -n "${APPTAINER_SIF:-}" ]]; then
  if [[ -n "$LOCAL_PYTHON" ]]; then
    PYTHON_CMD=(apptainer exec --nv "$APPTAINER_SIF" "$LOCAL_PYTHON")
  else
    PYTHON_CMD=(apptainer exec --nv "$APPTAINER_SIF" uv run python)
  fi
else
  if [[ -n "$LOCAL_PYTHON" ]]; then
    PYTHON_CMD=("$LOCAL_PYTHON")
  else
    PYTHON_CMD=(uv run python)
  fi
fi

run_cmd() {
  echo "+ $*"
  "$@"
}

run_case() {
  local case_id="$1"
  local task="$2"

  local case_dir="$RUN_ROOT/$case_id"
  mkdir -p "$case_dir"
  printf '%s\n' "$task" > "$case_dir/task.txt"

  local opt_cmd=(
    "${PYTHON_CMD[@]}" -m agent.opt.cli optimize
    --task "$task"
    --model "$MODEL"
    --backend "$BACKEND"
    --max-opt-rounds "$MAX_OPT_ROUNDS"
    --max-attempts "$MAX_ATTEMPTS"
    --xml-max-attempts "$XML_MAX_ATTEMPTS"
    --timeout-sec "$TIMEOUT_SEC"
    --sample-every-sec "$SAMPLE_EVERY_SEC"
    --max-frames "$MAX_FRAMES"
    --out-dir "$case_dir"
    --out "$case_dir/summary.json"
  )
  if [[ -n "$CRITIC_MODEL" ]]; then
    opt_cmd+=(--critic-model "$CRITIC_MODEL")
  fi
  if [[ -n "$REASONING_EFFORT" ]]; then
    opt_cmd+=(--reasoning-effort "$REASONING_EFFORT")
  fi
  if [[ -n "$CRITIC_REASONING_EFFORT" ]]; then
    opt_cmd+=(--critic-reasoning-effort "$CRITIC_REASONING_EFFORT")
  fi

  echo "==> [$case_id] optimize"
  run_cmd "${opt_cmd[@]}"

  local round_dir
  while IFS= read -r round_dir; do
    [[ -z "$round_dir" ]] && continue
    if [[ -f "$round_dir/ir.validated.json" ]]; then
      echo "==> [$case_id] compile $(basename "$round_dir")"
      run_cmd "${PYTHON_CMD[@]}" -m agent.cli compile \
        --ir "$round_dir/ir.validated.json" \
        --out "$round_dir/compiled_genesis.py"
    fi
  done < <(find "$case_dir" -maxdepth 1 -type d -name 'round_*' | sort)
}

mkdir -p "$RUN_ROOT"

echo "Run root: $RUN_ROOT"
echo "Model: $MODEL"
echo "Critic model: ${CRITIC_MODEL:-<same as model>}"
echo "Reasoning effort: ${REASONING_EFFORT:-<default>}"
echo "Critic reasoning effort: ${CRITIC_REASONING_EFFORT:-<same as generator>}"
echo "Backend: $BACKEND"
echo "Python: ${PYTHON_CMD[*]}"

while IFS='|' read -r case_id task; do
  [[ -z "$case_id" ]] && continue
  [[ "${case_id:0:1}" == "#" ]] && continue
  run_case "$case_id" "$task"
done <<'CASES'
drawer|Create a desk with two drawers, and open the left drawer gradually until it is fully open over 8s.
CASES

echo "Done. Results are under: $RUN_ROOT"
