## LTS overview



1\. LDA clusters entire dataset into 10 topic clusters (once, before loop)



Loop 10 times:

&#x20;   2. Thompson Sampling picks a cluster based on win/loss history

&#x20;

&#x20;   3. Sample 200 from that cluster:

&#x20;      - First iteration: random 200

&#x20;      - Later iterations (if filter\_label=True and BERT exists):

&#x20;        BERT scans the entire cluster, predicts labels,

&#x20;        then picks 200 from the predicted-positive samples

&#x20;

&#x20;   4. Qwen labels the 200 samples → "relevant animal" or "not"

&#x20;

&#x20;   5. Add these 200 to ALL previously l abeled data

&#x20;

&#x20;   6. Balance if needed (undersample majority class)

&#x20;

&#x20;   7. Train BERT on ALL accumulated labeled data

&#x20;

&#x20;   8. Evaluate BERT on validation set (leather\_validation.csv)

&#x20;

&#x20;   9. Compare F1 to previous best:

&#x20;      - Improved → save model, record "win" for this cluster

&#x20;      - Didn't improve → keep old model, record "loss"

&#x20;

&#x20;   10. Thompson updates probabilities → clusters with wins

&#x20;       are more likely to be picked next iteration

\-------------------------------------------------------------------------

### main\_cluster.py



#### Algorithm



When you run main\_cluster.py, it follows this exact path:



&#x09;***Setup***: Loads the unlabeled data and validation data. Cleans the text.



&#x09;***Clustering***: Groups the text into topics using LDA.



&#x09;

&#x09;The Loop (Runs 10 times):



&#x09;	***Sample***: The Thompson Sampler picks a cluster to pull data from.



&#x09;	***Label***: The LLM annotates this small batch of data.



&#x09;	***Balance***: It checks if it needs to mix in previous positive samples to keep the training data balanced.



&#x09;	***Train***: The BERT model trains on this new data.



&#x09;	***Evaluate***: The script checks if the newly trained BERT model beats the previous baseline accuracy/F1 score.



&#x09;	***Reward***: If the model improved, the script saves the model, updates the baseline, and gives the Thompson Sampler a "win" for that cluster. If it got worse, it throws away 		the new model, reverts to the old one, and gives the sampler a "loss."





#### Prints



If you execute main\_cluster.py, your console will look very busy. Based on the code, here is the sequence of what will be printed:



Initialization:



&#x09;--"Creating LDA" (if it doesn't find a saved CSV) followed by the number of clusters (e.g., 10) and "LDA created".



&#x09;--"Loading HuggingFace model: ..." and "HuggingFace model loaded" (if using local Hugging Face for labels).



Inside the Loop (Repeats 10 times):



&#x09;--"Chosen bandit \[X]" (The cluster ID selected).



&#x09;--"length of bendit \[Y]" (How much data is in that cluster).



&#x09;--"df for inference created" followed by a progress bar: Labeling samples: 100%|██████████| ...



&#x09;--The value counts of the labels (e.g., 0: 25, 1: 5).



&#x09;--"adding positive data: ..." or "Balanced data: ..." (if the balancing logic triggers).



&#x09;--"using model \[bert-base-uncased or previous best]"



&#x09;--"Starting with metric \[metric] baseline \[X.XX]" or "previous model \[metric] metric baseline of: \[X.XX]"



&#x09;--"Starting training" followed by "Unbalanced? True/False".



&#x09;--If unbalanced: "using modified loss function".



&#x09;--Hugging Face Trainer Logs: You will see a standard progress bar for training steps, followed by a printed dictionary of evaluation results (e.g., {'eval\_loss': 0.45, 'eval\_accuracy': 0.82, ...}).



The Verdict:  

&#x09;--If it did well: "Model improved with \[0.05]"



&#x09;--If it did poorly, it stays quiet on this step but saves positive data.



End of Script:



&#x09;--"Bendt with highest expected improvement: \[X]"



&#x09;--An array of the total wins for each cluster (e.g., \[2. 0. 1. 5. ...]).



&#x09;--An array of the total losses for each cluster.

\-------------------------------------------------------------------------

## Results



The run completed all 10 iterations. Final results:



Iter	Bandit	Positives Best F1 Result

0	8	1	0.607	Win

1	5	26	0.265	Loss

2	8	1+27	0.680	Win

3	8	2	0.665	Loss

4	9	7	0.646	Loss

5	9	5	0.647	Loss

6	8	3	0.531	Loss

7	5	34	0.638	Loss

8	0	17	0.647	Loss

9	2	6+74	0.685	Win

Final best F1: 0.685 (iteration 9, bandit 2)



The Thompson Sampling results at the end:



Wins: bandit 2 (0.99), bandit 8 (1.83)

Losses: bandit 0 (0.98), bandit 5 (1.88), bandit 8 (1.89), bandit 9 (1.89)

Best bandit: 8 had the highest win ratio overall

