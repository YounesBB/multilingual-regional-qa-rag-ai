#!/bin/bash
#SBATCH --job-name=rerank_retrieval
#SBATCH --account=ec403
#SBATCH --mail-type=END,FAIL
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=accel
#SBATCH --gpus=rtx30:1
#SBATCH --output=slurm_logs/rerank_retrieval-%j.out

# Rerank existing E5 top-10 retrieval into top-5 and top-10 candidate sets:
#   sbatch --gpus=rtx30:1 --mem=32G --time=01:00:00 \
#     --export=ALL,INPUT_TAG=e5_full_dev_retrieval_top10,OUTPUT_TOP_KS="5 10" \
#     slurm/rerank_retrieval.slurm
#
# Then run Tiny Aya Water generation with, for example:
#   sbatch --gpus=1 --mem=32G --time=01:30:00 \
#     --export=ALL,MODEL_NAME=CohereLabs/tiny-aya-water,MODEL_TAG=tiny_aya_water,RETRIEVAL_TAG=e5_full_dev_retrieval_top10_rerank_bge_top5,LIMIT_RECORDS_PER_LANG=0 \
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
INPUT_TAG="${INPUT_TAG:-e5_full_dev_retrieval_top10}"
INPUT_JSONL="${INPUT_JSONL:-$RUN_DIR/${INPUT_TAG}.jsonl}"
RERANKER_MODEL="${RERANKER_MODEL:-BAAI/bge-reranker-v2-m3}"
RERANKER_TAG="${RERANKER_TAG:-bge}"
RERANKER_CACHE_DIR="${RERANKER_CACHE_DIR:-$CUSQA_WORK/hf_models}"
OUTPUT_TOP_KS="${OUTPUT_TOP_KS:-5 10}"
CANDIDATE_TOP_K="${CANDIDATE_TOP_K:-0}"
LIMIT_RECORDS="${LIMIT_RECORDS:-0}"
LIMIT_RECORDS_PER_LANG="${LIMIT_RECORDS_PER_LANG:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TOKENIZER_MAX_LENGTH="${TOKENIZER_MAX_LENGTH:-512}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"

mkdir -p "$RUN_DIR" "$SUMMARY_DIR" "$CUSQA_WORK/cache" "$CUSQA_WORK/hf_datasets" slurm_logs

export HF_HOME="${HF_HOME:-$CUSQA_WORK/cache}"
export HF_HUB_CACHE="$RERANKER_CACHE_DIR"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$CUSQA_WORK/hf_datasets}"
export TRANSFORMERS_CACHE="$RERANKER_CACHE_DIR"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CUSQA_WORK/cache/xdg}"
export TORCHDYNAMO_DISABLE=1

echo "INPUT_JSONL=$INPUT_JSONL"
echo "RERANKER_MODEL=$RERANKER_MODEL"
echo "RERANKER_TAG=$RERANKER_TAG"
echo "RERANKER_CACHE_DIR=$RERANKER_CACHE_DIR"
echo "OUTPUT_TOP_KS=$OUTPUT_TOP_KS"
echo "CANDIDATE_TOP_K=$CANDIDATE_TOP_K"
echo "LIMIT_RECORDS=$LIMIT_RECORDS"
echo "LIMIT_RECORDS_PER_LANG=$LIMIT_RECORDS_PER_LANG"
echo "LOCAL_FILES_ONLY=$LOCAL_FILES_ONLY"

read -r -a OUTPUT_TOP_K_ARRAY <<< "$OUTPUT_TOP_KS"
for OUTPUT_TOP_K in "${OUTPUT_TOP_K_ARRAY[@]}"; do
  OUTPUT_TAG="${INPUT_TAG}_rerank_${RERANKER_TAG}_top${OUTPUT_TOP_K}"
  if [[ "$LIMIT_RECORDS_PER_LANG" != "0" ]]; then
    OUTPUT_TAG="${OUTPUT_TAG}_${LIMIT_RECORDS_PER_LANG}perlang"
  elif [[ "$LIMIT_RECORDS" != "0" ]]; then
    OUTPUT_TAG="${OUTPUT_TAG}_${LIMIT_RECORDS}records"
  fi

  OUTPUT_JSONL="$RUN_DIR/${OUTPUT_TAG}.jsonl"
  SUMMARY_JSON="$SUMMARY_DIR/${OUTPUT_TAG}_summary.json"
  echo "OUTPUT_TOP_K=$OUTPUT_TOP_K"
  echo "OUTPUT_JSONL=$OUTPUT_JSONL"
  echo "SUMMARY_JSON=$SUMMARY_JSON"

  PYTHON_ARGS=(
    scripts/rerank_retrieval.py
    --input-jsonl "$INPUT_JSONL"
    --output-jsonl "$OUTPUT_JSONL"
    --summary-json "$SUMMARY_JSON"
    --reranker-model "$RERANKER_MODEL"
    --reranker-tag "$RERANKER_TAG"
    --cache-dir "$RERANKER_CACHE_DIR"
    --candidate-top-k "$CANDIDATE_TOP_K"
    --output-top-k "$OUTPUT_TOP_K"
    --limit-records "$LIMIT_RECORDS"
    --limit-records-per-lang "$LIMIT_RECORDS_PER_LANG"
    --batch-size "$BATCH_SIZE"
    --tokenizer-max-length "$TOKENIZER_MAX_LENGTH"
  )

  if [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]]; then
    PYTHON_ARGS+=(--trust-remote-code)
  fi

  if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
    PYTHON_ARGS+=(--local-files-only)
  fi

  python "${PYTHON_ARGS[@]}"
done
