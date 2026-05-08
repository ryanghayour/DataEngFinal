# LTS — Learn to Sample

Comparing LDA, BERTopic+KMeans, and Top2Vec as clustering backends for the LTS active learning framework on the Leather (wildlife trade) and Reuters crude oil datasets.

---

## Running the Sweep Experiments

All sweeps are submitted via `run_sweep.sh` on NYU HPC (BigPurple). Each sweep runs 18 experiments (3 clustering methods × 2 balance settings × 3 runs) on a single GPU node.

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

Results are saved to a timestamped directory (e.g. `sweep_results_leather_20250501_120000/`) with one `.log` and one `_model_results.json` per run.

### Prerequisites
- Conda environment at `/gpfs/scratch/$USER/venvs/lts` with packages from `requirements.txt`
- HuggingFace cache at `/gpfs/scratch/$USER/hf_cache` with `bert-base-uncased` and `Qwen/Qwen2.5-3B-Instruct` pre-downloaded
- Run `sbatch` from the repo root (scripts use `$SLURM_SUBMIT_DIR` to find themselves)
- Reuters data is already preprocessed in `data_use_cases/` — `prepare_reuters_crude.py` is only needed if regenerating from raw Reuters-21578 files

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
