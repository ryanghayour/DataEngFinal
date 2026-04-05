import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from text_embedding import BertTextEmbedder


class KMeansTextClusterer:
    def __init__(self, num_topics=10, model_name='bert-base-uncased', batch_size=100, random_state=42):
        self.num_topics = num_topics
        self.random_state = random_state
        self.batch_size = batch_size
        self.embedder = BertTextEmbedder(model_name=model_name, save_embedding=True)
        self.kmeans = KMeans(n_clusters=self.num_topics, random_state=self.random_state, n_init='auto')
        self.embeddings = None

    def _get_embeddings(self, texts):
        """Generate normalized BERT embeddings for a list of texts."""
        embeddings = self.embedder.get_bert_embeddings(texts)
        return normalize(embeddings)

    def fit(self, texts):
        self.embeddings = self._get_embeddings(texts)
        self.kmeans.fit(self.embeddings)

    def transform(self, texts):
        embeddings = self._get_embeddings(texts)
        return self.kmeans.predict(embeddings).tolist()

    def fit_transform(self, texts):
        self.embeddings = self._get_embeddings(texts)
        labels = self.kmeans.fit_predict(self.embeddings)
        return labels.tolist()
