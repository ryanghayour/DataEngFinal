#!/bin/bash
#SBATCH --job-name=lts-sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --partition=gpu4_medium
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=sweep_%j.out
#SBATCH --error=sweep_%j.err

PROJECT_DIR=$SLURM_SUBMIT_DIR
cd "$PROJECT_DIR"

export HF_HOME=/gpfs/scratch/$USER/hf_cache

module purge 2>/dev/null || true
source ~/.bashrc 2>/dev/null || true

# Robust conda activation: source conda.sh directly instead of relying on .bashrc hooks
CONDA_BASE=$(conda info --base 2>/dev/null || echo "/gpfs/share/apps/anaconda3/gpu/2023.09")
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate /gpfs/scratch/$USER/venvs/lts

# Compute nodes don't pick up the env's libstdc++ from .bashrc, so prepend it
# explicitly. Without this, sklearn fails with "GLIBCXX_3.4.29 not found".
export LD_LIBRARY_PATH="/gpfs/scratch/$USER/venvs/lts/lib:${LD_LIBRARY_PATH:-}"

# Diagnostic output: prove we're using the right Python before the sweep starts
echo "===== Environment diagnostics ====="
echo "which python: $(which python)"
python -c "import sys; print('python:', sys.executable)"
python -c "import scipy; print('scipy:', scipy.__version__)"
python -c "import numpy; print('numpy:', numpy.__version__)"
python -c "from transformers import AutoTokenizer; print('transformers: OK')"
python -c "from top2vec import Top2Vec; print('top2vec: OK')"
echo "==================================="

# Select sweep script based on DATASET env var (default: leather)
# Usage: sbatch --export=DATASET=reuters run_sweep.sh
DATASET=${DATASET:-leather}

if [ "$DATASET" = "leather_filterlabel" ]; then
    echo "Running leather sweep (filter_label=True)..."
    bash sweep_experiments_leather_filterlabel.sh
elif [ "$DATASET" = "reuters" ]; then
    echo "Running Reuters crude sweep (filter_label=False)..."
    bash sweep_experiments_reuters.sh
elif [ "$DATASET" = "reuters_filterlabel" ]; then
    echo "Running Reuters crude sweep (filter_label=True)..."
    bash sweep_experiments_reuters_filterlabel.sh
else
    echo "Running leather sweep (filter_label=False)..."
    bash sweep_experiments_leather.sh
fi
