#!/bin/bash
#SBATCH --job-name=e5_full_retrieval
#SBATCH --account=ec403
#SBATCH --mail-type=END,FAIL
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=accel
#SBATCH --gpus=rtx30:1
#SBATCH --output=slurm_logs/retrieve_finewiki_dev-%j.out

# Debug retrieval over the full index:
#   sbatch --gpus=rtx30:1 --mem=24G --time=00:10:00 \
#     --export=ALL,LIMIT_QUESTIONS_PER_LANG=3 \
#     slurm/retrieve_finewiki_dev.slurm
#
# Full development retrieval:
#   sbatch --gpus=rtx30:1 --mem=24G --time=00:15:00 \
#     --export=ALL,LIMIT_QUESTIONS_PER_LANG=0 \
#     slurm/retrieve_finewiki_dev.slurm

set -o errexit
set -o nounset

module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-nlptools/01-foss-2024a-Python-3.12.3
module load nlpl-pytorch/2.6.0-foss-2024a-cuda-12.6.0-Python-3.12.3
module load nlpl-transformers/4.55.4-foss-2024a-Python-3.12.3

cd "${SLURM_SUBMIT_DIR:-$PWD}"

CUSQA_WORK="${CUSQA_WORK:-/cluster/work/projects/ec403/${USER:-user}/cusqa-rag-2026}"
INDEX_TAG="${INDEX_TAG:-e5_full}"
INDEX_DIR="${INDEX_DIR:-$CUSQA_WORK/indexes/$INDEX_TAG}"
MODEL_NAME="${MODEL_NAME:-intfloat/multilingual-e5-large}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-$CUSQA_WORK/hf_models}"
QUESTIONS_JSONL="${QUESTIONS_JSONL:-$CUSQA_WORK/inputs/cusqa_dev_all.jsonl}"
TOP_K="${TOP_K:-5}"
LIMIT_QUESTIONS_PER_LANG="${LIMIT_QUESTIONS_PER_LANG:-0}"
TOKENIZER_MAX_LENGTH="${TOKENIZER_MAX_LENGTH:-512}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-16}"
FAISS_VENV="${FAISS_VENV:-$CUSQA_WORK/venvs/faiss-cpu}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
OUTPUT_TAG="${OUTPUT_TAG:-}"

if [[ -z "$OUTPUT_TAG" ]]; then
  if [[ "$LIMIT_QUESTIONS_PER_LANG" == "0" ]]; then
    OUTPUT_TAG="${INDEX_TAG}_dev_retrieval_top${TOP_K}"
  else
    OUTPUT_TAG="${INDEX_TAG}_dev_retrieval_top${TOP_K}_${LIMIT_QUESTIONS_PER_LANG}perlang"
  fi
fi

RUN_DIR="$CUSQA_WORK/runs"
SUMMARY_DIR="$CUSQA_WORK/summaries"
OUTPUT_JSONL="$RUN_DIR/${OUTPUT_TAG}.jsonl"
SUMMARY_JSON="$SUMMARY_DIR/${OUTPUT_TAG}_summary.json"

mkdir -p "$RUN_DIR" "$SUMMARY_DIR" "$CUSQA_WORK/cache" "$CUSQA_WORK/hf_models" "$CUSQA_WORK/hf_datasets" slurm_logs

export HF_HOME="${HF_HOME:-$CUSQA_WORK/cache}"
export HF_HUB_CACHE="$MODEL_CACHE_DIR"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$CUSQA_WORK/hf_datasets}"
export TRANSFORMERS_CACHE="$MODEL_CACHE_DIR"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CUSQA_WORK/cache/xdg}"
export TORCHDYNAMO_DISABLE=1

if [[ -d "$FAISS_VENV" ]]; then
  source "$FAISS_VENV/bin/activate"
fi

if ! python -c "import faiss" >/dev/null 2>&1; then
  echo "FAISS is not available. Create the project-local FAISS venv used by slurm/build_finewiki_index.slurm."
  exit 2
fi

echo "INDEX_TAG=$INDEX_TAG"
echo "INDEX_DIR=$INDEX_DIR"
echo "MODEL_NAME=$MODEL_NAME"
echo "MODEL_CACHE_DIR=$MODEL_CACHE_DIR"
echo "QUESTIONS_JSONL=$QUESTIONS_JSONL"
echo "TOP_K=$TOP_K"
echo "LIMIT_QUESTIONS_PER_LANG=$LIMIT_QUESTIONS_PER_LANG"
echo "OUTPUT_JSONL=$OUTPUT_JSONL"
echo "SUMMARY_JSON=$SUMMARY_JSON"
echo "LOCAL_FILES_ONLY=$LOCAL_FILES_ONLY"

PYTHON_ARGS=(
  scripts/retrieve_finewiki_dev.py
  --questions-jsonl "$QUESTIONS_JSONL"
  --index-dir "$INDEX_DIR"
  --index-tag "$INDEX_TAG"
  --output-jsonl "$OUTPUT_JSONL"
  --summary-json "$SUMMARY_JSON"
  --limit-questions-per-lang "$LIMIT_QUESTIONS_PER_LANG"
  --top-k "$TOP_K"
  --model-name "$MODEL_NAME"
  --model-cache-dir "$MODEL_CACHE_DIR"
  --tokenizer-max-length "$TOKENIZER_MAX_LENGTH"
  --embedding-batch-size "$EMBEDDING_BATCH_SIZE"
)

if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
  PYTHON_ARGS+=(--local-files-only)
fi

python "${PYTHON_ARGS[@]}"
