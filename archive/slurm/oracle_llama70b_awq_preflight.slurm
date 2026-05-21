#!/bin/bash
#SBATCH --job-name=llama70b_awq_preflight
#SBATCH --account=ec403
#SBATCH --mail-type=END,FAIL
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=accel
#SBATCH --gpus=a100_80:1
#SBATCH --output=slurm_logs/oracle_llama70b_awq_preflight-%j.out

# Preferred preflight:
#   sbatch --gpus=a100_80:1 --mem=96G --time=01:00:00 \
#     oracle_llama70b_awq_preflight.slurm
#
# Riskier queue-time experiment:
#   sbatch --gpus=rtx30:2 --mem=96G --time=01:00:00 \
#     --export=ALL,TENSOR_PARALLEL_SIZE=2 \
#     oracle_llama70b_awq_preflight.slurm

set -o errexit
set -o nounset

module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-nlptools/01-foss-2024a-Python-3.12.3
module load nlpl-pytorch/2.6.0-foss-2024a-cuda-12.6.0-Python-3.12.3
module load nlpl-transformers/4.55.4-foss-2024a-Python-3.12.3

cd "${SLURM_SUBMIT_DIR:-$PWD}"

MODEL_NAME="${MODEL_NAME:-casperhansen/llama-3.3-70b-instruct-awq}"
MODEL_TAG="${MODEL_TAG:-llama33_70b_awq_preflight}"
LIMIT_QUESTIONS_PER_LANG="${LIMIT_QUESTIONS_PER_LANG:-1}"
MAX_TOKENS="${MAX_TOKENS:-64}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
VLLM_TOP_K="${VLLM_TOP_K:--1}"
TOP_K="${TOP_K:-5}"
CHUNK_TOKENS="${CHUNK_TOKENS:-512}"
OVERLAP_TOKENS="${OVERLAP_TOKENS:-128}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DTYPE="${DTYPE:-auto}"
VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-awq_marlin}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
SEED="${SEED:-0}"

CUSQA_WORK="${CUSQA_WORK:-/cluster/work/projects/ec403/${USER:-user}/cusqa-rag-2026}"
INPUT_DIR="$CUSQA_WORK/inputs"
RUN_DIR="$CUSQA_WORK/runs"
SUMMARY_DIR="$CUSQA_WORK/summaries"
ORACLE_CACHE_DIR="${ORACLE_CACHE_DIR:-$CUSQA_WORK/oracle_cache}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-$CUSQA_WORK/hf_models}"

PREFLIGHT_INPUT_JSONL="$INPUT_DIR/cusqa_dev_${MODEL_TAG}.jsonl"
PROMPT_JSONL="$RUN_DIR/oracle_${MODEL_TAG}_prompts.jsonl"
OUTPUT_JSONL="$RUN_DIR/oracle_${MODEL_TAG}.jsonl"
SUMMARY_JSON="$SUMMARY_DIR/oracle_${MODEL_TAG}_summary.json"

mkdir -p "$INPUT_DIR" "$RUN_DIR" "$SUMMARY_DIR" "$ORACLE_CACHE_DIR"
mkdir -p "$MODEL_CACHE_DIR" "$CUSQA_WORK/hf_datasets" "$CUSQA_WORK/cache" slurm_logs

export HF_HOME="${HF_HOME:-$CUSQA_WORK/cache}"
export HF_HUB_CACHE="$MODEL_CACHE_DIR"
export HF_DATASETS_CACHE="$CUSQA_WORK/hf_datasets"
export TRANSFORMERS_CACHE="$MODEL_CACHE_DIR"
export XDG_CACHE_HOME="$CUSQA_WORK/cache/xdg"
export TORCHDYNAMO_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

echo "MODEL_NAME=$MODEL_NAME"
echo "MODEL_TAG=$MODEL_TAG"
echo "MODEL_CACHE_DIR=$MODEL_CACHE_DIR"
echo "PREFLIGHT_INPUT_JSONL=$PREFLIGHT_INPUT_JSONL"
echo "PROMPT_JSONL=$PROMPT_JSONL"
echo "OUTPUT_JSONL=$OUTPUT_JSONL"
echo "SUMMARY_JSON=$SUMMARY_JSON"
echo "LIMIT_QUESTIONS_PER_LANG=$LIMIT_QUESTIONS_PER_LANG"
echo "MAX_MODEL_LEN=$MAX_MODEL_LEN"
echo "GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION"
echo "MAX_NUM_SEQS=$MAX_NUM_SEQS"
echo "MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS"
echo "TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "VLLM_QUANTIZATION=$VLLM_QUANTIZATION"
echo "ENFORCE_EAGER=$ENFORCE_EAGER"
echo "VLLM_WORKER_MULTIPROC_METHOD=$VLLM_WORKER_MULTIPROC_METHOD"

python scripts/prepare_full_dev.py \
  --output-jsonl "$PREFLIGHT_INPUT_JSONL" \
  --cache-dir "$HF_DATASETS_CACHE" \
  --limit-per-lang "$LIMIT_QUESTIONS_PER_LANG"

python scripts/oracle_rag.py \
  --questions-jsonl "$PREFLIGHT_INPUT_JSONL" \
  --output-jsonl "$PROMPT_JSONL" \
  --oracle-cache-dir "$ORACLE_CACHE_DIR" \
  --limit-questions-per-lang 0 \
  --chunk-tokens "$CHUNK_TOKENS" \
  --overlap-tokens "$OVERLAP_TOKENS" \
  --top-k "$TOP_K" \
  --model-name "$MODEL_NAME" \
  --cache-dir "$MODEL_CACHE_DIR" \
  --max-new-tokens "$MAX_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --generation-top-k 0 \
  --skip-generation

module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-vllm/0.8.2-foss-2024a-Python-3.12.3

VLLM_ARGS=(
  archive/scripts/vllm_generate_oracle_records.py
  --input-jsonl "$PROMPT_JSONL"
  --output-jsonl "$OUTPUT_JSONL"
  --model-name "$MODEL_NAME"
  --cache-dir "$MODEL_CACHE_DIR"
  --max-tokens "$MAX_TOKENS"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --top-k "$VLLM_TOP_K"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --batch-size "$BATCH_SIZE"
  --dtype "$DTYPE"
  --quantization "$VLLM_QUANTIZATION"
  --seed "$SEED"
  --allow-multimodal-inputs
)

if [[ "$ENFORCE_EAGER" == "1" ]]; then
  VLLM_ARGS+=(--enforce-eager)
fi

if [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]]; then
  VLLM_ARGS+=(--trust-remote-code)
fi

if [[ "${LOCAL_FILES_ONLY:-0}" == "1" ]]; then
  VLLM_ARGS+=(--local-files-only)
fi

python "${VLLM_ARGS[@]}"

python scripts/validate_oracle_full_dev.py \
  --input-jsonl "$OUTPUT_JSONL" \
  --expected-jsonl "$PREFLIGHT_INPUT_JSONL" \
  --summary-json "$SUMMARY_JSON" \
  --model-name "$MODEL_NAME"
