def chroma_search(query, collection, corpus="both", top_k=3):
    """Search Chroma collection with filters"""
    filters = {"source": corpus} if corpus in ["novel", "script"] else None
    
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        where=filters,
        include=["documents", "metadatas", "distances"]
    )
    
    return [{
        "text": doc,
        "score": 1 - dist,  # Convert distance to similarity
        "source": meta.get("source", "unknown"),
        "metadata": meta
    } for doc, dist, meta in zip(results["documents"][0], 
                               results["distances"][0], 
                               results["metadatas"][0])]
