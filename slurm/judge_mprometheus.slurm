#!/bin/bash
#SBATCH --job-name=mprometheus_eval
#SBATCH --account=ec403
#SBATCH --mail-type=END,FAIL
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --output=slurm_logs/judge_mprometheus-%j.out

# Smoke judge on Qwen shortlist, 1 example per language:
#   sbatch --gpus=1 --mem=32G --time=00:30:00 \
#     --export=ALL,LIMIT_RECORDS_PER_LANG=1 \
#     slurm/judge_mprometheus.slurm
#
# Full judge on Qwen shortlist:
#   sbatch --gpus=1 --mem=32G --time=05:00:00 \
#     --export=ALL,LIMIT_RECORDS_PER_LANG=0 \
#     slurm/judge_mprometheus.slurm
#
# Judge every completed Oracle and Embedding RAG model:
#   sbatch --gpus=rtx30:1 --mem=32G --time=24:00:00 \
#     --export=ALL,INPUT_GROUP=all,LIMIT_RECORDS_PER_LANG=0 \
#     slurm/judge_mprometheus.slurm
#
# Judge one system in isolation:
#   sbatch --gpus=rtx30:1 --mem=32G --time=02:00:00 \
#     --export=ALL,INPUT_GROUP=single,SYSTEM_ID=embedding_qwen3_30b_a3b_top5 \
#     slurm/judge_mprometheus.slurm

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
INPUT_GROUP="${INPUT_GROUP:-qwen}"
SYSTEM_ID="${SYSTEM_ID:-}"
if [[ -z "${OUTPUT_DIR:-}" ]]; then
  case "$INPUT_GROUP" in
    all) OUTPUT_DIR="$CUSQA_WORK/eval/mprometheus_v2_all" ;;
    single) OUTPUT_DIR="$CUSQA_WORK/eval/mprometheus_v2_${SYSTEM_ID}" ;;
    *) OUTPUT_DIR="$CUSQA_WORK/eval/mprometheus_v2" ;;
  esac
fi
if [[ -z "${COMBINED_JSON:-}" ]]; then
  case "$INPUT_GROUP" in
    all) COMBINED_JSON="$CUSQA_WORK/eval/mprometheus_v2_all_dev_summary.json" ;;
    single) COMBINED_JSON="$CUSQA_WORK/eval/mprometheus_v2_${SYSTEM_ID}_summary.json" ;;
    *) COMBINED_JSON="$CUSQA_WORK/eval/mprometheus_v2_dev_summary.json" ;;
  esac
fi
if [[ -z "${COMBINED_CSV:-}" ]]; then
  case "$INPUT_GROUP" in
    all) COMBINED_CSV="$CUSQA_WORK/eval/mprometheus_v2_all_dev_summary.csv" ;;
    single) COMBINED_CSV="$CUSQA_WORK/eval/mprometheus_v2_${SYSTEM_ID}_summary.csv" ;;
    *) COMBINED_CSV="$CUSQA_WORK/eval/mprometheus_v2_dev_summary.csv" ;;
  esac
fi
MODEL_NAME="${MODEL_NAME:-Unbabel/M-Prometheus-7B}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/fp/projects01/ec403/hf_models}"
LIMIT_RECORDS="${LIMIT_RECORDS:-0}"
LIMIT_RECORDS_PER_LANG="${LIMIT_RECORDS_PER_LANG:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-192}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
RESUME="${RESUME:-1}"
RETRY_UNPARSED="${RETRY_UNPARSED:-0}"

mkdir -p "$OUTPUT_DIR" "$CUSQA_WORK/cache" "$CUSQA_WORK/hf_datasets" slurm_logs

export HF_HOME="${HF_HOME:-$CUSQA_WORK/cache}"
export HF_HUB_CACHE="$MODEL_CACHE_DIR"
export HF_DATASETS_CACHE="$CUSQA_WORK/hf_datasets"
export TRANSFORMERS_CACHE="$MODEL_CACHE_DIR"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CUSQA_WORK/cache/xdg}"
export TORCHDYNAMO_DISABLE=1

case "$INPUT_GROUP" in
  qwen)
    INPUTS=(
      "$RUN_DIR/oracle_full_dev_qwen3_30b_a3b.jsonl"
      "$RUN_DIR/embedding_rag_qwen3_30b_a3b_e5_full_dev_retrieval_top5.jsonl"
      "$RUN_DIR/embedding_rag_qwen3_30b_a3b_e5_full_dev_retrieval_top10.jsonl"
    )
    ;;
  all)
    INPUTS=(
      "$RUN_DIR/oracle_full_dev_tiny_aya.jsonl"
      "$RUN_DIR/oracle_full_dev_tiny_aya_water.jsonl"
      "$RUN_DIR/oracle_full_dev_llama31_8b.jsonl"
      "$RUN_DIR/oracle_full_dev_qwen3_30b_a3b.jsonl"
      "$RUN_DIR/embedding_rag_tiny_aya_global_e5_full_dev_retrieval_top5.jsonl"
      "$RUN_DIR/embedding_rag_tiny_aya_global_e5_full_dev_retrieval_top10.jsonl"
      "$RUN_DIR/embedding_rag_tiny_aya_water_e5_full_dev_retrieval_top5.jsonl"
      "$RUN_DIR/embedding_rag_tiny_aya_water_e5_full_dev_retrieval_top10.jsonl"
      "$RUN_DIR/embedding_rag_llama31_8b_e5_full_dev_retrieval_top5.jsonl"
      "$RUN_DIR/embedding_rag_llama31_8b_e5_full_dev_retrieval_top10.jsonl"
      "$RUN_DIR/embedding_rag_qwen3_30b_a3b_e5_full_dev_retrieval_top5.jsonl"
      "$RUN_DIR/embedding_rag_qwen3_30b_a3b_e5_full_dev_retrieval_top10.jsonl"
    )
    ;;
  single)
    case "$SYSTEM_ID" in
      oracle_tiny_aya_global)
        INPUTS=("$RUN_DIR/oracle_full_dev_tiny_aya.jsonl")
        ;;
      oracle_tiny_aya_water)
        INPUTS=("$RUN_DIR/oracle_full_dev_tiny_aya_water.jsonl")
        ;;
      oracle_llama31_8b)
        INPUTS=("$RUN_DIR/oracle_full_dev_llama31_8b.jsonl")
        ;;
      oracle_qwen3_30b_a3b)
        INPUTS=("$RUN_DIR/oracle_full_dev_qwen3_30b_a3b.jsonl")
        ;;
      embedding_tiny_aya_global_top5)
        INPUTS=("$RUN_DIR/embedding_rag_tiny_aya_global_e5_full_dev_retrieval_top5.jsonl")
        ;;
      embedding_tiny_aya_global_top10)
        INPUTS=("$RUN_DIR/embedding_rag_tiny_aya_global_e5_full_dev_retrieval_top10.jsonl")
        ;;
      embedding_tiny_aya_water_top5)
        INPUTS=("$RUN_DIR/embedding_rag_tiny_aya_water_e5_full_dev_retrieval_top5.jsonl")
        ;;
      embedding_tiny_aya_water_top10)
        INPUTS=("$RUN_DIR/embedding_rag_tiny_aya_water_e5_full_dev_retrieval_top10.jsonl")
        ;;
      embedding_tiny_aya_water_rerank_bge_top5)
        INPUTS=("$RUN_DIR/embedding_rag_tiny_aya_water_e5_full_dev_retrieval_top10_rerank_bge_top5.jsonl")
        ;;
      embedding_tiny_aya_water_rerank_bge_top10)
        INPUTS=("$RUN_DIR/embedding_rag_tiny_aya_water_e5_full_dev_retrieval_top10_rerank_bge_top10.jsonl")
        ;;
      embedding_llama31_8b_top5)
        INPUTS=("$RUN_DIR/embedding_rag_llama31_8b_e5_full_dev_retrieval_top5.jsonl")
        ;;
      embedding_llama31_8b_top10)
        INPUTS=("$RUN_DIR/embedding_rag_llama31_8b_e5_full_dev_retrieval_top10.jsonl")
        ;;
      embedding_qwen3_30b_a3b_top5)
        INPUTS=("$RUN_DIR/embedding_rag_qwen3_30b_a3b_e5_full_dev_retrieval_top5.jsonl")
        ;;
      embedding_qwen3_30b_a3b_top10)
        INPUTS=("$RUN_DIR/embedding_rag_qwen3_30b_a3b_e5_full_dev_retrieval_top10.jsonl")
        ;;
      *)
        echo "Unknown SYSTEM_ID=$SYSTEM_ID for INPUT_GROUP=single." >&2
        echo "Valid SYSTEM_ID values:" >&2
        echo "  oracle_tiny_aya_global oracle_tiny_aya_water oracle_llama31_8b oracle_qwen3_30b_a3b" >&2
        echo "  embedding_tiny_aya_global_top5 embedding_tiny_aya_global_top10" >&2
        echo "  embedding_tiny_aya_water_top5 embedding_tiny_aya_water_top10" >&2
        echo "  embedding_tiny_aya_water_rerank_bge_top5 embedding_tiny_aya_water_rerank_bge_top10" >&2
        echo "  embedding_llama31_8b_top5 embedding_llama31_8b_top10" >&2
        echo "  embedding_qwen3_30b_a3b_top5 embedding_qwen3_30b_a3b_top10" >&2
        exit 2
        ;;
    esac
    ;;
  *)
    echo "Unknown INPUT_GROUP=$INPUT_GROUP. Use INPUT_GROUP=qwen, INPUT_GROUP=all, or INPUT_GROUP=single." >&2
    exit 2
    ;;
esac

PYTHON_ARGS=(
  scripts/judge_mprometheus.py
  --output-dir "$OUTPUT_DIR"
  --combined-json "$COMBINED_JSON"
  --combined-csv "$COMBINED_CSV"
  --model-name "$MODEL_NAME"
  --cache-dir "$MODEL_CACHE_DIR"
  --limit-records "$LIMIT_RECORDS"
  --limit-records-per-lang "$LIMIT_RECORDS_PER_LANG"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
)

for input in "${INPUTS[@]}"; do
  PYTHON_ARGS+=(--input-jsonl "$input")
done

if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
  PYTHON_ARGS+=(--local-files-only)
fi

if [[ "$RESUME" == "1" ]]; then
  PYTHON_ARGS+=(--resume)
fi

if [[ "$RETRY_UNPARSED" == "1" ]]; then
  PYTHON_ARGS+=(--retry-unparsed)
fi

python "${PYTHON_ARGS[@]}"
