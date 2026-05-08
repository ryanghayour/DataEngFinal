from top2vec import Top2Vec
import os

class Top2VecModel:
    def __init__(self, speed="learn", workers=8, cluster_size=None):
        self.speed = speed
        self.workers = workers
        self.cluster_size = cluster_size
        self.model = None

    def fit_transform(self, documents):
        print(f"Training Top2Vec model with speed='{self.speed}' and workers={self.workers}...")
        
        self.model = Top2Vec(documents=documents, speed=self.speed, workers=self.workers)
        
        if self.cluster_size is not None:
            original_num_topics = self.model.get_num_topics()
            if original_num_topics > int(self.cluster_size):
                print(f"Reducing Top2Vec topics hierarchically from {original_num_topics} to {self.cluster_size}...")
                self.model.hierarchical_topic_reduction(num_topics=int(self.cluster_size))
                return self.model.doc_top_reduced
            else:
                return self.model.doc_top
                
        return self.model.doc_top

    def get_topic_info(self):
        if self.model is None:
            raise ValueError("Model is not trained yet. Call fit_transform first.")
            
        topic_words, word_scores, topic_nums = self.model.get_topics()
        return topic_words, word_scores, topic_nums

    def save_model(self, file_path):
        if self.model:
            self.model.save(file_path)
            print(f"Top2Vec model saved successfully to {file_path}")
        else:
            print("No model to save.")

    @classmethod
    def load_model(cls, file_path):
        instance = cls()
        instance.model = Top2Vec.load(file_path)
        print(f"Top2Vec model loaded from {file_path}")
        return instance