#!/usr/bin/env bash
set -euo pipefail

# Optional: allow overriding GPU id and episode range
GPU_ID="${GPU_ID:-4}"
START_EP="${START_EP:-9}"
END_EP="${END_EP:-100}"

CONFIG="pi05_xv_finetune"
EXP_NAME="0224_s2"
STEP="20000"
REPO="/share/chenshuaiwen-local/.cache/hf_home/fastumi/pick_place_long"
BASE_OUT="eval_results/relative/0224_s2"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.9"

for ep in $(seq "${START_EP}" "${END_EP}"); do
  out_dir="${BASE_OUT}/${ep}"
  mkdir -p "${out_dir}"

  echo "===== Running episode ${ep} -> ${out_dir} ====="
  uv run python ./umi_scripts_csw/eval_relative_new.py \
    --config "${CONFIG}" \
    --exp-name "${EXP_NAME}" \
    --step "${STEP}" \
    --repo "${REPO}" \
    --episode "${ep}" \
    --out-dir "${out_dir}" \
    --dims 10 \
    --use-time-axis
done

echo "All done: episodes ${START_EP}..${END_EP}"
