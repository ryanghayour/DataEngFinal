#!/bin/bash
#SBATCH --job-name=lts-hf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/scratch/np3106/DE_Project/DataEngFinal/LTS/lts_%j.out
#SBATCH --error=/gpfs/scratch/np3106/DE_Project/DataEngFinal/LTS/lts_%j.err

PROJECT_DIR=/gpfs/scratch/np3106/DE_Project/DataEngFinal/LTS
cd $PROJECT_DIR

module purge
source ~/.bashrc
conda activate /gpfs/scratch/np3106/venvs/lts

# Run LTS with HuggingFace model on shark dataset
python main_cluster.py \
    -sample_size 200 \
    -filename "data_use_cases/shark_trophy" \
    -val_path "data_use_cases/validation_sharks.csv" \
    -balance False \
    -sampling "thompson" \
    -filter_label True \
    -model_finetune "bert-base-uncased" \
    -labeling "huggingface" \
    -hf_model "mistralai/Mistral-7B-Instruct-v0.3" \
    -model "text" \
    -baseline 0.5 \
    -metric "f1" \
    -cluster_size 10
