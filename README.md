# LTS — Learn to Sample

Comparing LDA, BERTopic+KMeans, and Top2Vec as clustering backends for the LTS active learning framework on the Leather (wildlife trade) and Reuters crude oil datasets.

---

## Running on HPC (BigPurple)

### Step 1 — Clone the repo

```bash
ssh <netid>@bigpurple.nyumc.org
cd /gpfs/scratch/$USER
git clone https://github.com/ryanghayour/DataEngFinal.git
cd DataEngFinal
```

### Step 2 — Create the conda environment (first time only)

Some packages (e.g. `hdbscan`) require compiling C extensions, which the login node will kill. Request an interactive compute node first:

```bash
srun --pty --mem=16GB --cpus-per-task=4 --time=01:00:00 bash
```

Then inside that session:

```bash
conda create -p /gpfs/scratch/$USER/venvs/lts python=3.10 -y
conda activate /gpfs/scratch/$USER/venvs/lts
conda install -c conda-forge hdbscan -y   # pre-built binary; pip build fails on GCC 15
pip install -r requirements.txt
exit  # return to login node when done
```

### Step 3 — Download HuggingFace models (first time only)

```bash
export HF_HOME=/gpfs/scratch/$USER/hf_cache
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('bert-base-uncased')"
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B-Instruct')"
```

### Step 4 — Submit sweep jobs

Run from the repo root. Each sweep runs 18 experiments (3 clustering methods × 2 balance settings × 3 runs) on a single GPU node (~24–48 h on `gpu4_medium`).

```bash
# Leather — filter_label=False (main results)
sbatch run_sweep.sh

# Leather — filter_label=True
sbatch --export=DATASET=leather_filterlabel run_sweep.sh

# Reuters — filter_label=False
sbatch --export=DATASET=reuters run_sweep.sh

# Reuters — filter_label=True
sbatch --export=DATASET=reuters_filterlabel run_sweep.sh
```

### Step 5 — Monitor progress

```bash
squeue -u $USER
tail -f sweep_<jobid>.out
```

Results are saved to a timestamped directory (e.g. `sweep_results_leather_20250501_120000/`) with one `.log` and one `_model_results.json` per run.

---

## File Reference

| File | Description |
|------|-------------|
| `run_sweep.sh` | SLURM job script — submit this with `sbatch`. Selects the right sweep script based on the `DATASET` env var. |
| `sweep_experiments_leather.sh` | Runs 18 experiments on the Leather dataset (`filter_label=False`). |
| `sweep_experiments_leather_filterlabel.sh` | Same as above with `filter_label=True`. |
| `sweep_experiments_reuters.sh` | Runs 18 experiments on the Reuters crude oil dataset (`filter_label=False`). |
| `sweep_experiments_reuters_filterlabel.sh` | Same as above with `filter_label=True`. |
| `main_cluster.py` | Entry point for a single experiment run. |
| `prepare_reuters_crude.py` | Preprocesses Reuters-21578 into the training pool and validation set. |
| `thompson_sampling.py` | Thompson Sampling bandit that selects clusters and tracks rewards. |
| `fine_tune.py` | BERT fine-tuning and inference logic. |
| `labeling.py` | LLM labeling via HuggingFace (Qwen). |
| `LDA.py` | LDA clustering backend. |
| `bertopic_cluster.py` | BERTopic + KMeans clustering backend. |
| `top2vec_cluster.py` | Top2Vec clustering backend. |
| `text_cluster.py` | Shared clustering interface. |
| `text_embedding.py` | Text embedding utilities. |
| `preprocessing.py` | Text cleaning and tokenization. |
| `model_sampling.py` | Sampling logic (Thompson, random). |
| `analyze_clusters.py` | Standalone script to inspect cluster size distributions. |
| `data_use_cases/` | Leather and Reuters training pools and validation sets. |
