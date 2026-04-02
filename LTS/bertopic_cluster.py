from bertopic import BERTopic
from sentence_transformers import SentenceTransformer


class BERTopicModel:
    def __init__(self, nr_topics=10):
        self.nr_topics = nr_topics
        self.model = None

    def fit_transform(self, texts):
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.model = BERTopic(
            embedding_model=embedding_model,
            nr_topics=self.nr_topics,
            verbose=True,
        )
        topics, _ = self.model.fit_transform(texts)

        # BERTopic assigns -1 to outliers. Reassign them to the nearest topic.
        topics = self.model.reduce_outliers(texts, topics, strategy="embeddings")

        print(f"BERTopic created {len(set(topics))} topics")
        topic_info = self.model.get_topic_info()
        print(topic_info[["Topic", "Count", "Name"]].to_string())

        return topics