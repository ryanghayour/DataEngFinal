"""
generate_cluster_plot.py

Run locally after downloading cluster_sizes.json from the HPC.
Generates figures/cluster_sizes.pdf with real cluster distributions.
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('figures', exist_ok=True)

with open('cluster_sizes.json') as f:
    data = json.load(f)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'lda':      '#E07070',
    'bertopic': '#70B070',
    'top2vec':  '#7090D0',
}
METHOD_LABELS = {
    'lda':      'LDA',
    'bertopic': 'BERTopic+KMeans',
    'top2vec':  'Top2Vec',
}
DATASET_LABELS = {
    'leather': 'Leather (152,495 docs)',
    'reuters': 'Reuters (8,761 docs)',
}

datasets = list(data.keys())
methods  = ['lda', 'bertopic', 'top2vec']

fig, axes = plt.subplots(len(datasets), len(methods),
                          figsize=(12, 5), sharey='row')
fig.suptitle('Cluster Size Distributions by Method and Dataset',
             fontweight='bold', y=1.02)

for row, dataset in enumerate(datasets):
    for col, method in enumerate(methods):
        ax = axes[row][col]
        dist = data[dataset][method]
        cluster_ids = sorted(dist.keys(), key=lambda x: int(x))
        sizes = [dist[k] for k in cluster_ids]
        x = np.arange(len(cluster_ids))

        bars = ax.bar(x, sizes, color=COLORS[method],
                      edgecolor='white', linewidth=0.5)

        # Annotate min/max
        min_s, max_s = min(sizes), max(sizes)
        ax.axhline(np.mean(sizes), color='black',
                   linestyle='--', linewidth=0.8, alpha=0.6, label='Mean')

        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in range(len(cluster_ids))],
                           fontsize=8)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)

        title = METHOD_LABELS[method]
        ratio = max_s / min_s if min_s > 0 else float('inf')
        ax.set_title(f'{title}\nmin={min_s:,}  max={max_s:,}  ratio={ratio:.1f}x',
                     fontsize=9, fontweight='bold')

        if col == 0:
            ax.set_ylabel(f'{DATASET_LABELS[dataset]}\nDocs per cluster',
                          fontsize=9)
        if row == len(datasets) - 1:
            ax.set_xlabel('Cluster ID', fontsize=9)

plt.tight_layout()
plt.savefig('figures/cluster_sizes.pdf', bbox_inches='tight')
plt.savefig('figures/cluster_sizes.png', bbox_inches='tight', dpi=150)
plt.close()
print('Saved figures/cluster_sizes.pdf and figures/cluster_sizes.png')
