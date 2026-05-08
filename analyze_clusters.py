"""
analyze_clusters.py

Standalone cluster analysis — no LTS, no labeling, no BERT.
Runs LDA, BERTopic+KMeans, and Top2Vec on both datasets and
reports real cluster size distributions.

Usage:
    python analyze_clusters.py

Outputs:
    - Prints cluster size table for each (dataset, method) pair
    - Saves cluster cache CSVs to data_use_cases/
    - Saves cluster_sizes.json summary for plotting
"""

import json
import os
import pandas as pd
from preprocessing import TextPreprocessor

preprocessor = TextPreprocessor()

DATASETS = {
    'leather': {
        'path':     'data_use_cases/data_leather.csv',
        'text_col': 'clean_title',   # preprocessor creates this from title
    },
    'reuters': {
        'path':     'data_use_cases/data_reuters_crude.csv',
        'text_col': 'clean_text',    # preprocessor creates this from text
    },
}

METHODS = ['lda', 'bertopic', 'top2vec']
CLUSTER_SIZE = 10

summary = {}   # {dataset: {method: {cluster_id: count}}}


def report(dataset_name, method, topics, texts_count):
    s = pd.Series(topics).value_counts().sort_index()
    print(f'\n  {dataset_name.upper()} x {method.upper()}  '
          f'({texts_count} docs, {s.nunique()} clusters)')
    print(f'  {"Cluster":<10} {"Count":>8}  {"% of corpus":>12}')
    print(f'  {"-"*34}')
    for cluster_id, count in s.items():
        pct = count / texts_count * 100
        bar = '#' * int(pct / 2)
        print(f'  {cluster_id:<10} {count:>8}  {pct:>11.1f}%  {bar}')
    print(f'  Min: {s.min()}  Max: {s.max()}  '
          f'Ratio: {s.max()/s.min():.1f}x  Std: {s.std():.0f}')
    return s.to_dict()


# ── Run all combinations ──────────────────────────────────────────────────────

for dataset_name, cfg in DATASETS.items():
    summary[dataset_name] = {}
    print(f'\n{"="*60}')
    print(f'  Dataset: {dataset_name}')
    print(f'{"="*60}')

    # Load and preprocess
    data = pd.read_csv(cfg['path'])
    print(f'  Loaded {len(data):,} documents')
    data = preprocessor.preprocess_df(data)
    text_col = cfg['text_col']
    texts = data[text_col].fillna('').tolist()

    for method in METHODS:
        cache_file = f'data_use_cases/data_{dataset_name}_crude_{method}.csv' \
                     if dataset_name == 'reuters' \
                     else f'data_use_cases/data_{dataset_name}_{method}.csv'

        # Use cached file if it exists
        if os.path.exists(cache_file):
            print(f'\n  [{method}] Loading from cache: {cache_file}')
            cached = pd.read_csv(cache_file)
            topics = cached['label_cluster'].tolist()
        else:
            print(f'\n  [{method}] Clustering {len(texts):,} documents...')
            if method == 'lda':
                from LDA import LDATopicModel
                model = LDATopicModel(num_topics=CLUSTER_SIZE)
                topics = model.fit_transform(texts)
            elif method == 'bertopic':
                from bertopic_cluster import BERTopicModel
                model = BERTopicModel(nr_topics=CLUSTER_SIZE)
                topics = model.fit_transform(texts)
            else:  # top2vec
                from top2vec_cluster import Top2VecModel
                model = Top2VecModel(speed='learn', workers=8,
                                     cluster_size=CLUSTER_SIZE)
                topics = model.fit_transform(texts)

            # Save cache
            data['label_cluster'] = topics
            data.to_csv(cache_file, index=False)
            print(f'  [{method}] Saved cache: {cache_file}')

        cluster_dist = report(dataset_name, method, topics, len(texts))
        summary[dataset_name][method] = cluster_dist


# ── Save summary JSON for plotting ───────────────────────────────────────────

with open('cluster_sizes.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f'\n\nSaved cluster_sizes.json')
print('\nDone. Use cluster_sizes.json to generate the figure.')
