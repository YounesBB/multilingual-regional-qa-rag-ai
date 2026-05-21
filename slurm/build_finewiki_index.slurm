#!/bin/bash
#SBATCH --job-name=finewiki_e5_index
#SBATCH --account=ec403
#SBATCH --mail-type=END,FAIL
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=accel
#SBATCH --gpus=rtx30:1
#SBATCH --output=slurm_logs/build_finewiki_index-%j.out

# Debug index:
#   sbatch --gpus=rtx30:1 --mem=16G --time=00:15:00 \
#     --export=ALL,INDEX_TAG=e5_debug,LIMIT_DOCS_PER_LANG=2000 \
#     slurm/build_finewiki_index.slurm
#
# Full index after the debug run is validated:
#   sbatch --gpus=h100nv:1 --mem=32G --time=12:00:00 \
#     --export=ALL,INDEX_TAG=e5_full,LIMIT_DOCS_PER_LANG=0 \
#     slurm/build_finewiki_index.slurm
#
# One-time FAISS setup if the job reports that faiss is unavailable:
#   module purge
#   module use -a /fp/projects01/ec30/software/easybuild/modules/all/
#   module load nlpl-nlptools/01-foss-2024a-Python-3.12.3
#   module load nlpl-pytorch/2.6.0-foss-2024a-cuda-12.6.0-Python-3.12.3
#   module load nlpl-transformers/4.55.4-foss-2024a-Python-3.12.3
#   python -m venv --system-site-packages /cluster/work/projects/ec403/$USER/cusqa-rag-2026/venvs/faiss-cpu
#   source /cluster/work/projects/ec403/$USER/cusqa-rag-2026/venvs/faiss-cpu/bin/activate
#   python -m pip install --upgrade pip
#   python -m pip install faiss-cpu

set -o errexit
set -o nounset

module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-nlptools/01-foss-2024a-Python-3.12.3
module load nlpl-pytorch/2.6.0-foss-2024a-cuda-12.6.0-Python-3.12.3
module load nlpl-transformers/4.55.4-foss-2024a-Python-3.12.3

cd "${SLURM_SUBMIT_DIR:-$PWD}"

CUSQA_WORK="${CUSQA_WORK:-/cluster/work/projects/ec403/${USER:-user}/cusqa-rag-2026}"
INDEX_TAG="${INDEX_TAG:-e5_debug}"
INDEX_DIR="${INDEX_DIR:-$CUSQA_WORK/indexes/$INDEX_TAG}"
MODEL_NAME="${MODEL_NAME:-intfloat/multilingual-e5-large}"
LIMIT_DOCS_PER_LANG="${LIMIT_DOCS_PER_LANG:-2000}"
LIMIT_CHUNKS_PER_LANG="${LIMIT_CHUNKS_PER_LANG:-0}"
CHUNK_TOKENS="${CHUNK_TOKENS:-512}"
OVERLAP_TOKENS="${OVERLAP_TOKENS:-128}"
TOKENIZER_MAX_LENGTH="${TOKENIZER_MAX_LENGTH:-512}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-16}"
INDEX_BATCH_SIZE="${INDEX_BATCH_SIZE:-256}"
PROGRESS_EVERY="${PROGRESS_EVERY:-500}"
FAISS_VENV="${FAISS_VENV:-$CUSQA_WORK/venvs/faiss-cpu}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"

mkdir -p "$INDEX_DIR" "$CUSQA_WORK/cache" "$CUSQA_WORK/hf_models" "$CUSQA_WORK/hf_datasets" slurm_logs

export HF_HOME="${HF_HOME:-$CUSQA_WORK/cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$CUSQA_WORK/hf_models}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$CUSQA_WORK/hf_datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$CUSQA_WORK/hf_models}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CUSQA_WORK/cache/xdg}"
export TORCHDYNAMO_DISABLE=1

if [[ -d "$FAISS_VENV" ]]; then
  source "$FAISS_VENV/bin/activate"
fi

if ! python -c "import faiss" >/dev/null 2>&1; then
  echo "FAISS is not available."
  echo "Create the project-local FAISS venv once, then resubmit:"
  echo "  module purge"
  echo "  module use -a /fp/projects01/ec30/software/easybuild/modules/all/"
  echo "  module load nlpl-nlptools/01-foss-2024a-Python-3.12.3"
  echo "  module load nlpl-pytorch/2.6.0-foss-2024a-cuda-12.6.0-Python-3.12.3"
  echo "  module load nlpl-transformers/4.55.4-foss-2024a-Python-3.12.3"
  echo "  python -m venv --system-site-packages $FAISS_VENV"
  echo "  source $FAISS_VENV/bin/activate"
  echo "  python -m pip install --upgrade pip"
  echo "  python -m pip install faiss-cpu"
  exit 2
fi

echo "INDEX_TAG=$INDEX_TAG"
echo "INDEX_DIR=$INDEX_DIR"
echo "MODEL_NAME=$MODEL_NAME"
echo "LIMIT_DOCS_PER_LANG=$LIMIT_DOCS_PER_LANG"
echo "LIMIT_CHUNKS_PER_LANG=$LIMIT_CHUNKS_PER_LANG"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "FAISS_VENV=$FAISS_VENV"

PYTHON_ARGS=(
  scripts/build_finewiki_index.py
  --index-tag "$INDEX_TAG"
  --output-dir "$INDEX_DIR"
  --model-name "$MODEL_NAME"
  --model-cache-dir "$HF_HUB_CACHE"
  --cache-dir "$HF_DATASETS_CACHE"
  --limit-docs-per-lang "$LIMIT_DOCS_PER_LANG"
  --limit-chunks-per-lang "$LIMIT_CHUNKS_PER_LANG"
  --chunk-tokens "$CHUNK_TOKENS"
  --overlap-tokens "$OVERLAP_TOKENS"
  --tokenizer-max-length "$TOKENIZER_MAX_LENGTH"
  --embedding-batch-size "$EMBEDDING_BATCH_SIZE"
  --index-batch-size "$INDEX_BATCH_SIZE"
  --progress-every "$PROGRESS_EVERY"
)

if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
  PYTHON_ARGS+=(--local-files-only)
fi

if [[ "$FORCE_REBUILD" == "1" ]]; then
  PYTHON_ARGS+=(--force)
fi

python "${PYTHON_ARGS[@]}"
