#!/usr/bin/env bash
#SBATCH --job-name=g4gpu-v7-shake
#SBATCH --account=lu2026-2-51
#SBATCH --partition=gpua40
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --output=validation/v7_nnbar_event/output/v7_shakedown_%j.out
#SBATCH --error=validation/v7_nnbar_event/output/v7_shakedown_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p validation/v7_nnbar_event/output output

module load GCC/13.2.0 CUDA/12.8.0 CMake/3.27.6 2>/dev/null || \
  module load GCC/13.2.0 CUDA/12.8.0

PYTHON_BIN="${G4GPU_V7_PYTHON:-python3}"
"${PYTHON_BIN}" validation/v7_nnbar_event/run_v7.py \
  --candidate output/v7_g4gpu.root \
  --reference output/v7_geant4_ref.root \
  --output output/v7_summary.json
