import numpy as np

def semantic_search(query: str, db: dict, corpus: str = "both", top_k: int = 5) -> list:
    """Search across Sherlock Holmes novels and Elementary scripts"""
    query_embedding = db["model"].encode([query])[0].astype('float32')
    results = []
    if corpus in ["novels", "both"]:
        D, I = db["novel_index"].search(np.array([query_embedding]), top_k)
        for i, idx in enumerate(I[0]):
            if 0 <= idx < len(db["novel_chunks"]):
                results.append({
                    "text": db["novel_chunks"][idx]["text"],
                    "metadata": db["novel_metadata"][idx],
                    "score": 1 - D[0][i],
                    "source": "novel"
                })
    if corpus in ["scripts", "both"]:
        D, I = db["script_index"].search(np.array([query_embedding]), top_k)
        for i, idx in enumerate(I[0]):
            if 0 <= idx < len(db["script_chunks"]):
                results.append({
                    "text": db["script_chunks"][idx]["text"],
                    "metadata": db["script_metadata"][idx],
                    "score": 1 - D[0][i],
                    "source": "script"
                })
    return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
