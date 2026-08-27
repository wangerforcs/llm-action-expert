#!/usr/bin/env bash
# Default full experiment: frozen Qwen3 KV backbone + AR Action Expert.
# Run from any directory: bash DUAL_LLM/run.sh
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"

# Override any of these without editing the script, e.g.:
#   EXPERT_ARCH=custom BATCH_SIZE=8 RUN_NAME=kv-bs8 bash DUAL_LLM/run.sh
# BATCH_SIZE is per GPU. Two GPUs therefore use global batch size 2*BATCH_SIZE.
data_path="${DATA_PATH:-/data/datasets/datasets-hf/APIGen-MT-5k/apigen-mt_5k.json}"
backbone_path="${BACKBONE_PATH:-/data/datasets/models-hf/Qwen3-4B}"
expert_arch="${EXPERT_ARCH:-qwen3}"
run_name="${RUN_NAME:-qwen3-4b-${expert_arch}-expert-36x1024}"
output_dir="${OUTPUT_DIR:-${script_dir}/runs/${run_name}}"
batch_size="${BATCH_SIZE:-16}"
epochs="${EPOCHS:-8}"
resume_from="${RESUME_FROM:-}"
num_gpus="${NUM_GPUS:-2}"
wandb_project="${WANDB_PROJECT:-dual-llm-kv-action}"
wandb_entity="${WANDB_ENTITY:-}"

if [[ ! -f "${data_path}" ]]; then
  echo "Dataset missing: ${data_path}" >&2
  exit 1
fi
if [[ ! -f "${backbone_path}/config.json" ]]; then
  echo "Backbone missing or incomplete: ${backbone_path}" >&2
  exit 1
fi
if ! [[ "${num_gpus}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_GPUS must be a positive integer, got: ${num_gpus}" >&2
  exit 1
fi
if [[ "${expert_arch}" != "qwen3" && "${expert_arch}" != "custom" ]]; then
  echo "EXPERT_ARCH must be qwen3 or custom, got: ${expert_arch}" >&2
  exit 1
fi

cmd=(
  torchrun --standalone --nproc_per_node "${num_gpus}" "${script_dir}/scripts/train.py"
  --data "${data_path}"
  --backbone "${backbone_path}"
  --output "${output_dir}"
  --max-context-tokens all
  --max-action-tokens 192
  --expert-width 1024
  --expert-layers 36
  --expert-heads 8
  --expert-arch "${expert_arch}"
  --kv-layers all
  --kv-tokens all
  --representation kv
  --batch-size "${batch_size}"
  --epochs "${epochs}"
  --lr 2e-4
  --warmup-steps 200
  --min-lr-ratio 0.1
  --wandb-project "${wandb_project}"
  --wandb-run-name "${run_name}"
  --wandb-log-interval 25
)

if [[ -n "${wandb_entity}" ]]; then
  cmd+=(--wandb-entity "${wandb_entity}")
fi
if [[ -n "${resume_from}" ]]; then
  if [[ "${resume_from}" != /* ]]; then
    resume_from="${script_dir}/${resume_from}"
  fi
  if [[ ! -f "${resume_from}" ]]; then
    echo "Resume checkpoint missing: ${resume_from}" >&2
    exit 1
  fi
  cmd+=(--resume-from "${resume_from}")
fi

cd "${script_dir}"
echo "Starting ${run_name}; expert=${expert_arch}; GPUs=${num_gpus}; per-GPU batch=${batch_size}; global batch=$((num_gpus * batch_size)); output=${output_dir}"
exec "${cmd[@]}"
