#!/bin/bash
#SBATCH --job-name=oracle_rag_qwen3
#SBATCH --account=ec403
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=younesb@uio.no
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --qos=devel
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --output=slurm_logs/oracle_rag-%j.out

set -o errexit
set -o nounset

module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-nlptools/01-foss-2024a-Python-3.12.3
module load nlpl-pytorch/2.6.0-foss-2024a-cuda-12.6.0-Python-3.12.3
module load nlpl-transformers/4.55.4-foss-2024a-Python-3.12.3

cd "${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p data/hf_cache data/oracle_cache data/runs slurm_logs

export HF_HOME="$PWD/data/hf_cache"
export HF_DATASETS_CACHE="$PWD/data/hf_cache"
export TRANSFORMERS_CACHE="$PWD/data/hf_cache"
export XDG_CACHE_HOME="$PWD/data/hf_cache"

python scripts/oracle_rag.py \
  --questions-jsonl data/smoke/cusqa_dev_10_per_lang.jsonl \
  --output-jsonl data/runs/oracle_rag_qwen3_0p6b.jsonl \
  --oracle-cache-dir data/oracle_cache \
  --limit-questions-per-lang 1 \
  --chunk-tokens 512 \
  --overlap-tokens 128 \
  --top-k 5 \
  --model-name Qwen/Qwen3-0.6B \
  --cache-dir data/hf_cache \
  --max-new-tokens 64 \
  --temperature 0.7 \
  --top-p 0.8
