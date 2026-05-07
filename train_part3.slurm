#!/bin/bash
#SBATCH --job-name=oblig3_p3_attn
#SBATCH --account=ec403
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=younesb@uio.no
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=accel
#SBATCH --gpus=rtx30:1

set -o errexit
set -o nounset

module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-nlptools/01-foss-2024a-Python-3.12.3
module load nlpl-pytorch/2.6.0-foss-2024a-cuda-12.6.0-Python-3.12.3
module load nlpl-transformers/4.55.4-foss-2024a-Python-3.12.3
module load nlpl-trl/0.22.2-foss-2024a-Python-3.12.3

CACHE_DIR=/fp/projects01/ec403/hf_cache/${USER}
mkdir -p "$CACHE_DIR" "$CACHE_DIR/datasets" "$CACHE_DIR/transformers" "$CACHE_DIR/metrics" "$CACHE_DIR/xdg"
export HF_HOME="$CACHE_DIR"
export HF_DATASETS_CACHE="$CACHE_DIR/datasets"
export HF_METRICS_CACHE="$CACHE_DIR/metrics"
export XDG_CACHE_HOME="$CACHE_DIR/xdg"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p data bin plots slurm_logs


python train_part3.py \
  --model_name "HuggingFaceTB/SmolLM2-1.7B" \
  --model_alias smollm2_1p7b_lora_attn \
  --cache_dir /fp/projects01/ec403/hf_models \
  --train_data /fp/projects01/ec403/IN5550/obligatories/3/obli3_train.tsv.gz \
  --test_data /fp/projects01/ec403/IN5550/obligatories/3/obli3_test.tsv.gz \
  --output_dir bin \
  --data_out_dir data \
  --num_train_epochs 2 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --skip_search \
  --learning_rates "1e-4" \
  --weight_decays "0.0" \
  --warmup_ratios "0.03" \
  --lora_target_sets "attn" \
  --lora_ranks "16" \
  --lora_alphas "32" \
  --lora_dropouts "0.05"

