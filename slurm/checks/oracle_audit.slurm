#!/bin/bash
#SBATCH --job-name=oracle_audit
#SBATCH --account=ec403
#SBATCH --mail-type=END,FAIL
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --qos=devel
#SBATCH --output=slurm_logs/oracle_audit-%j.out

set -o errexit
set -o nounset

module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-nlptools/01-foss-2024a-Python-3.12.3

cd "${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p data/oracle_cache data/runs slurm_logs

python scripts/oracle_audit.py \
  --questions-jsonl data/smoke/cusqa_dev_10_per_lang.jsonl \
  --output-jsonl data/runs/oracle_audit_10_per_lang.jsonl \
  --oracle-cache-dir data/oracle_cache \
  --limit-questions-per-lang 10 \
  --chunk-tokens 512 \
  --overlap-tokens 128 \
  --top-k 5
