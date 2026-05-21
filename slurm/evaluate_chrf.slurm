#!/bin/bash
#SBATCH --job-name=eval_chrf
#SBATCH --account=ec403
#SBATCH --mail-type=END,FAIL
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=normal
#SBATCH --output=slurm_logs/evaluate_chrf-%j.out

# Full development chrF over the curated Oracle + Embedding outputs:
#   sbatch slurm/evaluate_chrf.slurm
#
# One-file/debug chrF:
#   sbatch --export=ALL,INPUT_JSONL=/cluster/work/projects/ec403/$USER/cusqa-rag-2026/runs/embedding_rag_tiny_aya_water_e5_full_dev_retrieval_top5_1perlang.jsonl,ALLOW_NONFULL=1 \
#     slurm/evaluate_chrf.slurm

set -o errexit
set -o nounset

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm_logs

CUSQA_WORK="${CUSQA_WORK:-/cluster/work/projects/ec403/${USER:-user}/cusqa-rag-2026}"
OUTPUT_DIR="${OUTPUT_DIR:-$CUSQA_WORK/eval}"
COMBINED_JSON="${COMBINED_JSON:-$OUTPUT_DIR/chrf_dev_summary.json}"
COMBINED_CSV="${COMBINED_CSV:-$OUTPUT_DIR/chrf_dev_summary.csv}"
EXPECTED_COUNT="${EXPECTED_COUNT:-1408}"
INPUT_JSONL="${INPUT_JSONL:-}"
ALLOW_NONFULL="${ALLOW_NONFULL:-0}"

PYTHON_ARGS=(
  scripts/evaluate_chrf.py
  --output-dir "$OUTPUT_DIR"
  --combined-json "$COMBINED_JSON"
  --combined-csv "$COMBINED_CSV"
  --expected-count "$EXPECTED_COUNT"
)

if [[ -n "$INPUT_JSONL" ]]; then
  PYTHON_ARGS+=(--input-jsonl "$INPUT_JSONL")
fi

if [[ "$ALLOW_NONFULL" == "1" ]]; then
  PYTHON_ARGS+=(--allow-nonfull)
fi

python "${PYTHON_ARGS[@]}"
