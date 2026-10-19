#!/usr/bin/env bash
set -euo pipefail

# Run only after the A--E formal training has stopped. All outputs stay under this isolated directory.
PROJECT_ROOT="${PROJECT_ROOT:-/home/wenhao/pathmnist_course}"
BRANCH_DIR="${PROJECT_ROOT}/experiments/resume_v2"
DATA_DIR="${PROJECT_ROOT}/data"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ACCEPTANCE_DIR="${BRANCH_DIR}/runs/gpu_acceptance_${STAMP}"
FULL_DIR="${ACCEPTANCE_DIR}/continuous_4"
RESUME_DIR="${ACCEPTANCE_DIR}/two_plus_resume"

test -f "${DATA_DIR}/manifest.json"
test -f "${BRANCH_DIR}/config.gpu-smoke.json"
mkdir -p "${ACCEPTANCE_DIR}"
cd "${BRANCH_DIR}"
export PYTHONPATH="${BRANCH_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

python -m pathmnist_resume.cli start --data "${DATA_DIR}" --run "${FULL_DIR}" --config config.gpu-smoke.json --device cuda --workers 4 --smoke
python -m pathmnist_resume.cli start --data "${DATA_DIR}" --run "${RESUME_DIR}" --config config.gpu-smoke.json --device cuda --workers 4 --smoke --stop-after-epoch 2
python -m pathmnist_resume.cli resume --data "${DATA_DIR}" --run "${RESUME_DIR}" --device cuda --workers 4 --smoke
python -m pathmnist_resume.compare --full "${FULL_DIR}" --resumed "${RESUME_DIR}" --output "${ACCEPTANCE_DIR}/comparison.json"
python -m pathmnist_resume.monitor --run "${RESUME_DIR}" --once
printf 'GPU acceptance results: %s\n' "${ACCEPTANCE_DIR}/comparison.json"
