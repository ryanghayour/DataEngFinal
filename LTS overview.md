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

## Results



The run completed all 10 iterations. Final results:



Iter	Bandit	Positives	Best F1	Result

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

