#!/bin/bash
#SBATCH --job-name=oracle_model_compare
#SBATCH --account=ec403
#SBATCH --mail-type=END,FAIL
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --output=slurm_logs/oracle_model_compare-%j.out

# Example submissions:
# Tiny Aya:
#   sbatch --qos=devel --time=00:30:00 --export=ALL,MODEL_NAME=CohereLabs/tiny-aya-global,MODEL_TAG=tiny_aya oracle_model_compare.slurm
# Llama 3.1 8B, using the existing shared cache:
#   sbatch --export=ALL,MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct,MODEL_TAG=llama31_8b oracle_model_compare.slurm
# Gemma 4 26B A4B, first run with a larger GPU:
#   sbatch --gpus=a100_80:1 --mem=96G --time=04:00:00 --export=ALL,MODEL_NAME=google/gemma-4-26B-A4B-it,MODEL_TAG=gemma4_26b_a4b oracle_model_compare.slurm
# Optional Gemma model-card sampling experiment:
#   --export=ALL,MODEL_NAME=google/gemma-4-26B-A4B-it,MODEL_TAG=gemma4_26b_a4b,TEMPERATURE=1.0,TOP_P=0.95,GENERATION_TOP_K=64

set -o errexit
set -o nounset

module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-nlptools/01-foss-2024a-Python-3.12.3
module load nlpl-pytorch/2.6.0-foss-2024a-cuda-12.6.0-Python-3.12.3
module load nlpl-transformers/4.55.4-foss-2024a-Python-3.12.3

cd "${SLURM_SUBMIT_DIR:-$PWD}"

MODEL_NAME="${MODEL_NAME:-CohereLabs/tiny-aya-global}"
MODEL_TAG="${MODEL_TAG:-$(printf '%s' "$MODEL_NAME" | tr '[:upper:]' '[:lower:]' | sed -E 's#[^a-z0-9]+#_#g; s#^_+|_+$##g')}"
LIMIT_QUESTIONS_PER_LANG="${LIMIT_QUESTIONS_PER_LANG:-10}"
QUESTIONS_JSONL="${QUESTIONS_JSONL:-data/smoke/cusqa_dev_10_per_lang.jsonl}"
OUTPUT_JSONL="${OUTPUT_JSONL:-data/runs/oracle_rag_${MODEL_TAG}_${LIMIT_QUESTIONS_PER_LANG}_per_lang.jsonl}"
ORACLE_CACHE_DIR="${ORACLE_CACHE_DIR:-data/oracle_cache}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.8}"
GENERATION_TOP_K="${GENERATION_TOP_K:-0}"
TOP_K="${TOP_K:-5}"
CHUNK_TOKENS="${CHUNK_TOKENS:-512}"
OVERLAP_TOKENS="${OVERLAP_TOKENS:-128}"

CUSQA_WORK="${CUSQA_WORK:-/cluster/work/projects/ec403/${USER:-user}/cusqa-rag-2026}"
NEW_MODEL_CACHE="${NEW_MODEL_CACHE:-$CUSQA_WORK/hf_models}"
LLAMA_MODEL_CACHE="${LLAMA_MODEL_CACHE:-/fp/projects01/ec403/hf_models}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-$NEW_MODEL_CACHE}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-}"

if [[ "$MODEL_NAME" == "meta-llama/Llama-3.1-8B-Instruct" || "$MODEL_NAME" == "meta-llama/Meta-Llama-3.1-8B-Instruct" ]]; then
  MODEL_CACHE_DIR="$LLAMA_MODEL_CACHE"
  LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
fi

LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"

mkdir -p "$NEW_MODEL_CACHE" "$CUSQA_WORK/hf_datasets" "$CUSQA_WORK/cache" "$CUSQA_WORK/runs"
mkdir -p data/runs "$ORACLE_CACHE_DIR" slurm_logs

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
echo "OUTPUT_JSONL=$OUTPUT_JSONL"

PYTHON_ARGS=(
  scripts/oracle_rag.py
  --questions-jsonl "$QUESTIONS_JSONL"
  --output-jsonl "$OUTPUT_JSONL"
  --oracle-cache-dir "$ORACLE_CACHE_DIR"
  --limit-questions-per-lang "$LIMIT_QUESTIONS_PER_LANG"
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
