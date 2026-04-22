#!/bin/bash
# sweep_experiments.sh
# Runs all 12 LTS experiments sequentially: 4 configs x 3 runs each.
# Cleans state files between every run so they don't contaminate each other.
# Archives per-run logs and model_results.json into a timestamped directory.

set -u  # error on undefined vars (but not on command failures — we want to continue)

PROJECT_DIR=/gpfs/scratch/yw5653/de_final/DataEngFinal/LTS
cd "$PROJECT_DIR"

RESULTS_DIR="$PROJECT_DIR/sweep_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"
echo "Sweep results will be saved to: $RESULTS_DIR"

COMMON_ARGS=(
    -sample_size 200
    -filename data_use_cases/data_leather
    -val_path data_use_cases/leather_validation.csv
    -sampling thompson
    -filter_label True
    -model_finetune bert-base-uncased
    -labeling huggingface
    -hf_model Qwen/Qwen2.5-3B-Instruct
    -model text
    -baseline 0.5
    -metric f1
    -cluster_size 10
)

clean_state() {
    rm -f data_use_cases/data_leather_data_labeled.csv
    rm -f data_use_cases/data_leather_training_data.csv
    rm -f data_use_cases/data_leather_model_results.json
    rm -f positive_data.csv
    rm -f selected_ids.txt
    rm -f wins.txt
    rm -f losses.txt
    rm -rf models/
}

run_experiment() {
    local clustering=$1
    local balance=$2
    local run_num=$3

    local run_name="${clustering}_balance${balance}_run${run_num}"
    local log_file="$RESULTS_DIR/${run_name}.log"
    local results_file="$RESULTS_DIR/${run_name}_model_results.json"

    echo ""
    echo "===== [$(date '+%H:%M:%S')] Starting $run_name ====="

    clean_state

    # Both LDA and BERTopic are stochastic now — delete cache for both
    # so each run gets fresh clusters (apples-to-apples variance comparison).
    rm -f data_use_cases/data_leather_lda.csv
    # rm -f data_use_cases/data_leather_bertopic.csv
    rm -f data_use_cases/data_leather_top2vec.csv

    python main_cluster.py \
        "${COMMON_ARGS[@]}" \
        -clustering "$clustering" \
        -balance "$balance" \
        > "$log_file" 2>&1

    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "  WARNING: $run_name exited with code $exit_code (see $log_file)"
    fi

    if [ -f "data_use_cases/data_leather_model_results.json" ]; then
        cp data_use_cases/data_leather_model_results.json "$results_file"
    fi

    echo "===== [$(date '+%H:%M:%S')] Finished $run_name ====="
}

# 6 experiments: 2 configs x 3 runs
# balance=False is dropped because BERT collapses on extreme imbalance
# without upsampling, which triggers an infinite loop in the original
# thompson_sampling.py (no positives -> while data.empty never exits).

# for run in 1 2 3; do run_experiment lda      True $run; done
# for run in 1 2 3; do run_experiment top2vec  True $run; done

# for run in 1 2 3; do run_experiment lda      False $run; done
for run in 1 2 3; do run_experiment top2vec  False $run; done

echo ""
echo "All experiments complete. Results in $RESULTS_DIR"
ls -la "$RESULTS_DIR"
