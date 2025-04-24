import sys
import os
import numpy as np
import json
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
sys.path.insert(0, PROJECT_ROOT)

# Now import local modules
from src.modeling.vector_db import load_vector_database

# Rest of your code remains the same...
chunk_dir = os.path.join(PROJECT_ROOT, "data", "processed")

def load_data_and_embeddings():
    """Load processed chunks and embeddings from Chroma"""
    from src.modeling.vector_db import load_vector_database
    
    # Get Chroma collection
    collection = load_vector_database()
    
    # Retrieve all documents with embeddings
    results = collection.get(
        include=["documents", "metadatas", "embeddings"]
    )
    
    # Separate novels and scripts using metadata
    novel_texts = []
    novel_embeddings = []
    script_texts = []
    script_embeddings = []
    
    for doc, embedding, meta in zip(results["documents"], 
                                  results["embeddings"], 
                                  results["metadatas"]):
        if meta["source"] == "novel":
            novel_texts.append(doc)
            novel_embeddings.append(embedding)
        else:
            script_texts.append(doc)
            script_embeddings.append(embedding)
    
    return {
        "novel_embeddings": np.array(novel_embeddings),
        "script_embeddings": np.array(script_embeddings),
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
    all_embeddings = all_embeddings = np.concatenate([novel_embeddings, script_embeddings])
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
