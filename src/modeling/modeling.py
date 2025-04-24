import numpy as np
import os
import json
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
chunk_dir = os.path.join(PROJECT_ROOT, "data", "processed")
vector_db_dir = os.path.join(PROJECT_ROOT, "data", "vector_db")

def load_data_and_embeddings():
    """Load processed chunks and pre-computed embeddings"""
    # Load embeddings
    novel_embeddings = np.load(os.path.join(vector_db_dir, 'novel_embeddings.npy'))
    script_embeddings = np.load(os.path.join(vector_db_dir, 'script_embeddings.npy'))

    # Load text chunks
    def load_chunks(file_path):
        with open(file_path, 'r') as f:
            return [json.loads(line)["text"] for line in f]

    novel_texts = load_chunks(os.path.join(chunk_dir, "doyle_novels", "novel_chunks.jsonl"))
    script_texts = load_chunks(os.path.join(chunk_dir, "cbs_elementary", "elementary_chunks.jsonl"))

    return {
        "novel_embeddings": novel_embeddings,
        "script_embeddings": script_embeddings,
        "novel_texts": novel_texts,
        "script_texts": script_texts
    }

def create_topic_model(novel_texts, script_texts, novel_embeddings, script_embeddings):
    """Create and train optimized BERTopic model"""
    # Configuration
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )
    
    hdbscan_model = HDBSCAN(
        min_cluster_size=15,
        min_samples=5,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )
    
    vectorizer_model = CountVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.85
    )
    
    # Initialize model
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=True,
        verbose=True
    )
    
    # Prepare data
    all_texts = novel_texts + script_texts
    all_embeddings = np.vstack([novel_embeddings, script_embeddings])
    doc_labels = ["novel"] * len(novel_texts) + ["script"] * len(script_texts)

    # Fit model
    topic_model.fit(all_texts, embeddings=all_embeddings)
    
    # Post-processing
    topic_model.reduce_topics(all_texts, nr_topics=20)
    topic_model.generate_topic_labels(nr_words=3, separator=", ")

    # Get results
    topics, probs = topic_model.transform(all_texts, all_embeddings)

    # Save topics
    topic_model.get_topic_info().to_csv(os.path.join(PROJECT_ROOT, "data", "processed", "topics.csv"), index=False)
    
    return topic_model, topics, doc_labels, probs

def main():
    """Pipeline execution"""
    print("Loading data...")
    data = load_data_and_embeddings()
    
    print("Creating topic model...")
    topic_model, topics, doc_labels, probs = create_topic_model(
        data["novel_texts"],
        data["script_texts"],
        data["novel_embeddings"],
        data["script_embeddings"]
    )
    
    # Save model
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    topic_model.save(os.path.join(models_dir, "sherlock_topic_model"))
    
    print("✅ Topic modeling complete!")

if __name__ == "__main__":
    main()
