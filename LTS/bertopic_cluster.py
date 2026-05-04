from bertopic import BERTopic
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer


class BERTopicModel:
    def __init__(self, nr_topics=10):
        self.nr_topics = nr_topics
        self.model = None

    def fit_transform(self, texts):
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # KMeans guarantees exactly nr_topics clusters with balanced sizes.
        # No outlier (-1) bucket, no mega-cluster problem.
        # No random_state — keeps clustering stochastic for fair variance
        # comparison against LDA (which is also stochastic).
        kmeans_model = KMeans(n_clusters=self.nr_topics, n_init=10)

        self.model = BERTopic(
            embedding_model=embedding_model,
            hdbscan_model=kmeans_model,
            verbose=True,
        )
        topics, _ = self.model.fit_transform(texts)

        print(f"BERTopic created {len(set(topics))} topics")
        topic_info = self.model.get_topic_info()
        print(topic_info[["Topic", "Count", "Name"]].to_string())

        return topics