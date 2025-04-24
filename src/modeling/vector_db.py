# src/modeling/vector_db.py
import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Create vector database directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
output_dir = os.path.join(PROJECT_ROOT, "data", "vector_db")
os.makedirs(output_dir, exist_ok=True)

# file names for saving embeddings and metadata
novel_embs = output_dir + '/novel_embeddings.npy'
script_embs = output_dir + '/script_embeddings.npy'
novel_metadata_file = output_dir + '/novel_metadata.json'
script_metadata_file = output_dir + '/script_metadata.json'
novel_faiss_file = output_dir + '/novel_index.faiss'
script_faiss_file = output_dir + '/script_index.faiss'

#
novel_dir = os.path.join(PROJECT_ROOT, "data", "processed", "doyle_novels")
novel_file = os.path.join(novel_dir, "novel_chunks.jsonl")
script_dir = os.path.join(PROJECT_ROOT, "data", "processed", "cbs_elementary")
script_file = os.path.join(script_dir, "elementary_chunks.jsonl")

def load_vector_database():
    """Load the vector database and all necessary metadata."""
    # Load embeddings
    novel_embeddings = np.load(novel_embs)
    script_embeddings = np.load(script_embs)
    
    # Load FAISS indices
    novel_index = faiss.read_index(novel_faiss_file)
    script_index = faiss.read_index(script_faiss_file)
    
    # Load metadata
    with open(novel_metadata_file, 'r') as f:
        novel_metadata = json.load(f)
    with open(script_metadata_file, 'r') as f:
        script_metadata = json.load(f)
    
    # Load original chunks (for text retrieval)
    with open(novel_file, 'r') as f:
        novel_chunks = [json.loads(line) for line in f]
    with open(script_file, 'r') as f:
        script_chunks = [json.loads(line) for line in f]
    
    # Load embedding model (must match what you used for encoding)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    return {
        "novel_index": novel_index,
        "script_index": script_index,
        "novel_metadata": novel_metadata,
        "script_metadata": script_metadata,
        "novel_chunks": novel_chunks,
        "script_chunks": script_chunks,
        "model": model
    }

# Load your preprocessed data
with open(novel_file, 'r') as f:
    novel_chunks = [json.loads(line) for line in f]

with open(script_file, 'r') as f:
    script_chunks = [json.loads(line) for line in f]

# Extract text and track metadata
novel_texts = [chunk["text"] for chunk in novel_chunks]
novel_metadata = [{"id": chunk["chunk_id"], "source": "novel", "novel": chunk.get("novel", "")} for chunk in novel_chunks]

script_texts = [chunk["text"] for chunk in script_chunks]
script_metadata = [{"id": chunk["chunk_id"], "source": "script", "episode": chunk.get("episode", "")} for chunk in script_chunks]

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dimension vectors

# Create and save novel embeddings
print("Creating novel embeddings...")
novel_embeddings = model.encode(novel_texts, show_progress_bar=True)
novel_embs = output_dir + '/novel_embeddings.npy'
np.save(novel_embs, novel_embeddings)

# Create and save script embeddings
print("Creating script embeddings...")
script_embeddings = model.encode(script_texts, show_progress_bar=True)
script_embs = output_dir + '/script_embeddings.npy'
np.save(script_embs, script_embeddings)

# Save metadata
print("Saving metadata...")
# novel_metadata_file = output_dir + '/novel_metadata.json'
with open(novel_metadata_file, 'w') as f:
    json.dump(novel_metadata, f)

# script_metadata_file = output_dir + '/script_metadata.json'
with open(script_metadata_file, 'w') as f:
    json.dump(script_metadata, f)

# Build novel index
print("Building vector database...")

print("Creating FAISS index for novel embeddings...")
dimension = novel_embeddings.shape[1]  # Should be 384
novel_index = faiss.IndexFlatL2(dimension)
novel_index.add(novel_embeddings.astype('float32'))
faiss.write_index(novel_index, novel_faiss_file)


# Build script index
print("Creating FAISS index for script embeddings...")
script_index = faiss.IndexFlatL2(dimension)
script_index.add(script_embeddings.astype('float32'))
faiss.write_index(script_index, script_faiss_file)

print(f"✅ Vector database built: {novel_index.ntotal} novel chunks and {script_index.ntotal} script chunks indexed")
