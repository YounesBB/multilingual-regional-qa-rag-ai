#!/bin/bash
#SBATCH --job-name=oracle_full_dev
#SBATCH --account=ec403
#SBATCH --mail-type=END,FAIL
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --output=slurm_logs/oracle_full_dev-%j.out

# Llama full-dev baseline:
#   sbatch --gpus=rtx30:1 --mem=32G --time=01:30:00 \
#     --export=ALL,MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct,MODEL_TAG=llama31_8b \
#     slurm/oracle_full_dev.slurm
#
# Tiny Aya Water, European/Asia-Pacific Tiny Aya variant:
#   sbatch --gpus=rtx30:1 --mem=32G --time=01:30:00 \
#     --export=ALL,MODEL_NAME=CohereLabs/tiny-aya-water,MODEL_TAG=tiny_aya_water \
#     slurm/oracle_full_dev.slurm
#
# Qwen3 MoE upper-bound candidate, compatible with Transformers >=4.51:
#   sbatch --gpus=a100_80:1 --mem=80G --time=03:30:00 \
#     --export=ALL,MODEL_NAME=Qwen/Qwen3-30B-A3B,MODEL_TAG=qwen3_30b_a3b \
#     slurm/oracle_full_dev.slurm
#
# Gemma full-dev is blocked on the current Fox Transformers/vLLM module stack:
#   sbatch --gpus=a100_80:1 --mem=80G --time=03:30:00 \
#     --export=ALL,MODEL_NAME=google/gemma-4-26B-A4B-it,MODEL_TAG=gemma4_26b_a4b \
#     slurm/oracle_full_dev.slurm

set -o errexit
set -o nounset

module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-nlptools/01-foss-2024a-Python-3.12.3
module load nlpl-pytorch/2.6.0-foss-2024a-cuda-12.6.0-Python-3.12.3
module load nlpl-transformers/4.55.4-foss-2024a-Python-3.12.3

cd "${SLURM_SUBMIT_DIR:-$PWD}"

MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
MODEL_TAG="${MODEL_TAG:-$(printf '%s' "$MODEL_NAME" | tr '[:upper:]' '[:lower:]' | sed -E 's#[^a-z0-9]+#_#g; s#^_+|_+$##g')}"
LIMIT_QUESTIONS_PER_LANG="${LIMIT_QUESTIONS_PER_LANG:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
GENERATION_TOP_K="${GENERATION_TOP_K:-0}"
TOP_K="${TOP_K:-5}"
CHUNK_TOKENS="${CHUNK_TOKENS:-512}"
OVERLAP_TOKENS="${OVERLAP_TOKENS:-128}"

CUSQA_WORK="${CUSQA_WORK:-/cluster/work/projects/ec403/${USER:-user}/cusqa-rag-2026}"
INPUT_DIR="$CUSQA_WORK/inputs"
RUN_DIR="$CUSQA_WORK/runs"
SUMMARY_DIR="$CUSQA_WORK/summaries"
ORACLE_CACHE_DIR="${ORACLE_CACHE_DIR:-$CUSQA_WORK/oracle_cache}"
NEW_MODEL_CACHE="${NEW_MODEL_CACHE:-$CUSQA_WORK/hf_models}"
LLAMA_MODEL_CACHE="${LLAMA_MODEL_CACHE:-/fp/projects01/ec403/hf_models}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-$NEW_MODEL_CACHE}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-}"

if [[ "$MODEL_NAME" == "meta-llama/Llama-3.1-8B-Instruct" || "$MODEL_NAME" == "meta-llama/Meta-Llama-3.1-8B-Instruct" ]]; then
  MODEL_CACHE_DIR="$LLAMA_MODEL_CACHE"
  LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
fi
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"

QUESTIONS_JSONL="$INPUT_DIR/cusqa_dev_all.jsonl"
OUTPUT_JSONL="$RUN_DIR/oracle_full_dev_${MODEL_TAG}.jsonl"
SUMMARY_JSON="$SUMMARY_DIR/oracle_full_dev_${MODEL_TAG}_summary.json"
FULL_INPUT_JSONL="$INPUT_DIR/cusqa_dev_all.jsonl"

mkdir -p "$INPUT_DIR" "$RUN_DIR" "$SUMMARY_DIR" "$ORACLE_CACHE_DIR"
mkdir -p "$NEW_MODEL_CACHE" "$CUSQA_WORK/hf_datasets" "$CUSQA_WORK/cache" slurm_logs

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="$MODEL_CACHE_DIR"
export HF_DATASETS_CACHE="$CUSQA_WORK/hf_datasets"
export TRANSFORMERS_CACHE="$MODEL_CACHE_DIR"
export XDG_CACHE_HOME="$CUSQA_WORK/cache/xdg"
export TORCHDYNAMO_DISABLE=1

echo "MODEL_NAME=$MODEL_NAME"
echo "MODEL_TAG=$MODEL_TAG"
echo "MODEL_CACHE_DIR=$MODEL_CACHE_DIR"
echo "LOCAL_FILES_ONLY=$LOCAL_FILES_ONLY"
echo "QUESTIONS_JSONL=$QUESTIONS_JSONL"
echo "OUTPUT_JSONL=$OUTPUT_JSONL"
echo "SUMMARY_JSON=$SUMMARY_JSON"

python scripts/prepare_full_dev.py \
  --output-jsonl "$FULL_INPUT_JSONL" \
  --limit-per-lang "$LIMIT_QUESTIONS_PER_LANG"

PYTHON_ARGS=(
  scripts/oracle_rag.py
  --questions-jsonl "$QUESTIONS_JSONL"
  --output-jsonl "$OUTPUT_JSONL"
  --oracle-cache-dir "$ORACLE_CACHE_DIR"
  --limit-questions-per-lang 0
  --chunk-tokens "$CHUNK_TOKENS"
  --overlap-tokens "$OVERLAP_TOKENS"
  --top-k "$TOP_K"
  --model-name "$MODEL_NAME"
  --cache-dir "$MODEL_CACHE_DIR"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --generation-top-k "$GENERATION_TOP_K"
)

if [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]]; then
  PYTHON_ARGS+=(--trust-remote-code)
fi

if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
  PYTHON_ARGS+=(--local-files-only)
fi

python "${PYTHON_ARGS[@]}"

python scripts/validate_oracle_full_dev.py \
  --input-jsonl "$OUTPUT_JSONL" \
  --expected-jsonl "$QUESTIONS_JSONL" \
  --summary-json "$SUMMARY_JSON" \
  --model-name "$MODEL_NAME"
