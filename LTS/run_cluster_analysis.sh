#!/bin/bash
#SBATCH --job-name=lts-clusters
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:1
#SBATCH --time=0-04:00:00
#SBATCH --output=/gpfs/scratch/np3106/DE_Project/DataEngFinal/LTS/clusters_%j.out
#SBATCH --error=/gpfs/scratch/np3106/DE_Project/DataEngFinal/LTS/clusters_%j.err

PROJECT_DIR=/gpfs/scratch/np3106/DE_Project/DataEngFinal/LTS
cd "$PROJECT_DIR"

export HF_HOME=/gpfs/scratch/np3106/hf_cache

module purge 2>/dev/null || true
source ~/.bashrc 2>/dev/null || true

CONDA_BASE=$(conda info --base 2>/dev/null || echo "/gpfs/share/apps/anaconda3/gpu/2023.09")
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate /gpfs/scratch/np3106/venvs/lts

export LD_LIBRARY_PATH="/gpfs/scratch/np3106/venvs/lts/lib:${LD_LIBRARY_PATH:-}"

echo "===== [$(date '+%H:%M:%S')] Starting cluster analysis ====="
python analyze_clusters.py
echo "===== [$(date '+%H:%M:%S')] Done ====="
echo ""
echo "Download cluster_sizes.json and run generate_cluster_plot.py locally."
