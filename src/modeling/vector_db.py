# src/modeling/vector_db.py
import json
import os
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import tqdm
from tqdm import tqdm as tdm

# Configure paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
chroma_dir = os.path.join(PROJECT_ROOT, "data", "chroma_db")
os.makedirs(chroma_dir, exist_ok=True)

# Data paths
novel_file = os.path.join(PROJECT_ROOT, "data", "processed", "doyle_novels", "novel_chunks.jsonl")
script_file = os.path.join(PROJECT_ROOT, "data", "processed", "cbs_elementary", "elementary_chunks.jsonl")

def create_chroma_collection():
    """Create and persist Chroma collection with all data"""
    client = chromadb.PersistentClient(path=chroma_dir)
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    collection = client.get_or_create_collection(
        name="sherlock_holmes",
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine"}
    )

    # Process novels
    with open(novel_file, 'r') as f:
        novel_chunks = [json.loads(line) for line in f]
    
    novel_ids = [f"novel_{chunk['chunk_id']}" for chunk in novel_chunks]
    novel_texts = [chunk["text"] for chunk in novel_chunks]
    novel_metadatas = [{
        "source": "novel",
        "novel": chunk.get("novel", "Unknown"),
        "chunk_id": chunk["chunk_id"]
    } for chunk in novel_chunks]

    # Process scripts 
    with open(script_file, 'r') as f:
        script_chunks = [json.loads(line) for line in f]
    
    script_ids = [f"script_{chunk['chunk_id']}" for chunk in script_chunks]
    script_texts = [chunk["text"] for chunk in script_chunks]
    script_metadatas = [{
        "source": "script",
        "episode": chunk.get("episode", "Unknown"),
        "season": chunk.get("season", "Unknown"),
        "chunk_id": chunk["chunk_id"]
    } for chunk in script_chunks]

    # Add all documents to Chroma
    print("Adding documents to Chroma...")
    batch_size = 1000
    for i in tdm(range(0, len(novel_texts), batch_size)):
        end_idx = i + batch_size
        collection.add(
            ids=novel_ids[i:end_idx],
            documents=novel_texts[i:end_idx],
            metadatas=novel_metadatas[i:end_idx]
        )

    for i in tdm(range(0, len(script_texts), batch_size)):
        end_idx = i + batch_size
        collection.add(
            ids=script_ids[i:end_idx],
            documents=script_texts[i:end_idx],
            metadatas=script_metadatas[i:end_idx]
        )

    return collection

def load_vector_database():
    """Load persisted Chroma collection"""
    client = chromadb.PersistentClient(path=chroma_dir)
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return client.get_collection(
        name="sherlock_holmes",
        embedding_function=embedder
    )

# Create the collection if not exists
if __name__ == "__main__":
    print("Building Chroma database...")
    collection = create_chroma_collection()
    print(f"✅ Chroma database ready with {collection.count()} documents")
