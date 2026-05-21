#!/bin/bash
#SBATCH --job-name=embedding_rag_gen
#SBATCH --account=ec403
#SBATCH --mail-type=END,FAIL
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --partition=accel
#SBATCH --gpus=a100_80:1
#SBATCH --output=slurm_logs/embedding_rag_generate-%j.out

# Tiny Aya Water debug generation, 1 question per language:
#   sbatch --gpus=1 --mem=32G --time=00:30:00 \
#     --export=ALL,MODEL_NAME=CohereLabs/tiny-aya-water,MODEL_TAG=tiny_aya_water,LIMIT_RECORDS_PER_LANG=1 \
#     slurm/embedding_rag_generate.slurm
#
# Llama 3.1 8B debug generation, 1 question per language:
#   sbatch --gpus=rtx30:1 --mem=32G --time=00:30:00 \
#     --export=ALL,MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct,MODEL_TAG=llama31_8b,LIMIT_RECORDS_PER_LANG=1 \
#     slurm/embedding_rag_generate.slurm
#
# Qwen3-30B-A3B full Embedding RAG generation:
#   sbatch --gpus=a100_80:1 --mem=80G --time=04:00:00 \
#     --export=ALL,MODEL_NAME=Qwen/Qwen3-30B-A3B,MODEL_TAG=qwen3_30b_a3b,LIMIT_RECORDS_PER_LANG=0 \
#     slurm/embedding_rag_generate.slurm

set -o errexit
set -o nounset

module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-nlptools/01-foss-2024a-Python-3.12.3
module load nlpl-pytorch/2.6.0-foss-2024a-cuda-12.6.0-Python-3.12.3
module load nlpl-transformers/4.55.4-foss-2024a-Python-3.12.3

cd "${SLURM_SUBMIT_DIR:-$PWD}"

CUSQA_WORK="${CUSQA_WORK:-/cluster/work/projects/ec403/${USER:-user}/cusqa-rag-2026}"
RUN_DIR="$CUSQA_WORK/runs"
SUMMARY_DIR="$CUSQA_WORK/summaries"
NEW_MODEL_CACHE="${NEW_MODEL_CACHE:-$CUSQA_WORK/hf_models}"
LLAMA_MODEL_CACHE="${LLAMA_MODEL_CACHE:-/fp/projects01/ec403/hf_models}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-30B-A3B}"
MODEL_TAG="${MODEL_TAG:-$(printf '%s' "$MODEL_NAME" | tr '[:upper:]' '[:lower:]' | sed -E 's#[^a-z0-9]+#_#g; s#^_+|_+$##g')}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-$NEW_MODEL_CACHE}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"

if [[ "$MODEL_NAME" == "meta-llama/Llama-3.1-8B-Instruct" || "$MODEL_NAME" == "meta-llama/Meta-Llama-3.1-8B-Instruct" ]]; then
  MODEL_CACHE_DIR="$LLAMA_MODEL_CACHE"
  LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
fi

RETRIEVAL_TAG="${RETRIEVAL_TAG:-e5_full_dev_retrieval_top5}"
RETRIEVAL_JSONL="${RETRIEVAL_JSONL:-$RUN_DIR/${RETRIEVAL_TAG}.jsonl}"
LIMIT_RECORDS="${LIMIT_RECORDS:-0}"
LIMIT_RECORDS_PER_LANG="${LIMIT_RECORDS_PER_LANG:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
GENERATION_TOP_K="${GENERATION_TOP_K:-0}"
RESUME="${RESUME:-1}"

OUTPUT_TAG="embedding_rag_${MODEL_TAG}_${RETRIEVAL_TAG}"
if [[ "$LIMIT_RECORDS_PER_LANG" != "0" ]]; then
  OUTPUT_TAG="${OUTPUT_TAG}_${LIMIT_RECORDS_PER_LANG}perlang"
elif [[ "$LIMIT_RECORDS" != "0" ]]; then
  OUTPUT_TAG="${OUTPUT_TAG}_${LIMIT_RECORDS}records"
fi

OUTPUT_JSONL="${OUTPUT_JSONL:-$RUN_DIR/${OUTPUT_TAG}.jsonl}"
SUMMARY_JSON="${SUMMARY_JSON:-$SUMMARY_DIR/${OUTPUT_TAG}_summary.json}"

mkdir -p "$RUN_DIR" "$SUMMARY_DIR" "$NEW_MODEL_CACHE" "$CUSQA_WORK/cache" "$CUSQA_WORK/hf_datasets" slurm_logs

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
echo "RETRIEVAL_JSONL=$RETRIEVAL_JSONL"
echo "LIMIT_RECORDS=$LIMIT_RECORDS"
echo "LIMIT_RECORDS_PER_LANG=$LIMIT_RECORDS_PER_LANG"
echo "OUTPUT_JSONL=$OUTPUT_JSONL"
echo "SUMMARY_JSON=$SUMMARY_JSON"

PYTHON_ARGS=(
  scripts/generate_embedding_rag.py
  --retrieval-jsonl "$RETRIEVAL_JSONL"
  --output-jsonl "$OUTPUT_JSONL"
  --summary-json "$SUMMARY_JSON"
  --model-name "$MODEL_NAME"
  --cache-dir "$MODEL_CACHE_DIR"
  --limit-records "$LIMIT_RECORDS"
  --limit-records-per-lang "$LIMIT_RECORDS_PER_LANG"
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

if [[ "$RESUME" == "1" ]]; then
  PYTHON_ARGS+=(--resume)
fi

python "${PYTHON_ARGS[@]}"
