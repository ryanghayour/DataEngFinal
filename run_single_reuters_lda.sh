#!/bin/bash
#SBATCH --job-name=lts-single
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:1
#SBATCH --time=0-06:00:00
#SBATCH --output=/gpfs/scratch/np3106/DE_Project/DataEngFinal/LTS/single_%j.out
#SBATCH --error=/gpfs/scratch/np3106/DE_Project/DataEngFinal/LTS/single_%j.err

PROJECT_DIR=/gpfs/scratch/np3106/DE_Project/DataEngFinal/LTS
cd "$PROJECT_DIR"

export HF_HOME=/gpfs/scratch/np3106/hf_cache

module purge 2>/dev/null || true
source ~/.bashrc 2>/dev/null || true

CONDA_BASE=$(conda info --base 2>/dev/null || echo "/gpfs/share/apps/anaconda3/gpu/2023.09")
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate /gpfs/scratch/np3106/venvs/lts

export LD_LIBRARY_PATH="/gpfs/scratch/np3106/venvs/lts/lib:${LD_LIBRARY_PATH:-}"

# Clean state
rm -f data_use_cases/data_reuters_crude_data_labeled.csv
rm -f data_use_cases/data_reuters_crude_training_data.csv
rm -f data_use_cases/data_reuters_crude_model_results.json
rm -f data_use_cases/data_reuters_crude_lda.csv
rm -f positive_data.csv
rm -f selected_ids.txt
rm -f wins.txt
rm -f losses.txt
rm -rf models/

echo "===== [$(date '+%H:%M:%S')] Starting lda_balanceTrue_run1 (filter_label=True) ====="

python main_cluster.py \
    -sample_size 200 \
    -filename data_use_cases/data_reuters_crude \
    -val_path data_use_cases/reuters_crude_validation.csv \
    -sampling thompson \
    -filter_label True \
    -balance True \
    -model_finetune bert-base-uncased \
    -labeling huggingface \
    -hf_model Qwen/Qwen2.5-3B-Instruct \
    -model text \
    -baseline 0.5 \
    -metric f1 \
    -cluster_size 10 \
    -dataset reuters \
    -clustering lda \
    > single_lda_reuters_filterlabel.log 2>&1

echo "===== [$(date '+%H:%M:%S')] Finished ====="
echo ""
echo "Cluster cache saved to: data_use_cases/data_reuters_crude_lda.csv"
echo "Results saved to: data_use_cases/data_reuters_crude_model_results.json"
echo "Log: single_lda_reuters_filterlabel.log"
