import torch
torch.classes.__path__ = []

import os
import streamlit as st
from src.modeling.vector_db import load_vector_database
from src.inference.chroma_search import chroma_search
from src.llm.answer_generation import get_rag_answer

# Streamlit config
st.set_page_config(page_title="Sherlock Holmes RAG", layout="wide")
st.title("🔎 Decoding the Detective")

# Load Chroma collection
@st.cache_resource
def get_db():
    return load_vector_database()

collection = get_db()

# introduction
st.markdown("""
This app digs into how CBS's *Elementary* – the modern Sherlock Holmes reboot 
with a sober, tattooed detective in New York – stacks up against Arthur Conan Doyle's 
original 19th-century stories. You might be surprised what 130 years (and one 
gender-swapped Watson) can change... or what stays timeless about this iconic detective duo.
""")

# Image comparison section
col1, col2 = st.columns((1, 1.25))

with col1:
    st.image(
        "https://tvseriesfinale.com/wp-content/uploads/2015/05/elementary26.jpg",
        caption="CBS' Elementary Cast: Jonny Lee Miller as Sherlock Holmes, Lucy Liu as Joan Watson",
        use_container_width=True,  # Updated parameter
    )
    st.caption("Source: [TV Series Finale](https://tvseriesfinale.com)")

with col2:
    st.image(
        "https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEgNW8z0RHwr_LxOZ6REotwQhk_mbniLeVkvm_5nI82ljv49bJTjByOHqVnNFXf4qNLJJVhvvhYm8czrlxGM0DbkVv9UhsRRTgwQtIKsbYLh7v1d_UwHQEwHeKCe3RWAX6vTyctdVnv8pWB_/w1200-h630-p-k-no-nu/STUD+-+publication+date.PNG",
        caption="First Sherlock Holmes Novel: 'A Study in Scarlet' (1887)",
        use_container_width=True,  # Updated parameter
    )
    st.caption("Source: [I Hear of Sherlock](https://www.ihearofsherlock.com/2016/12/the-real-first-publication-date-of.html)")

st.divider()  # Modern replacement for markdown horizontal rule


# Example questions
st.markdown("#### Example questions:")
st.markdown("- Are there semantically similar paragraphs in two different works?")
st.markdown("- In which novel does Sherlock Holmes mention his cocaine use?")
st.markdown("- How does Joan Watson describe her transition from doctor to detective?")
st.markdown("- What role does addiction play in both versions of Sherlock Holmes?")

# Search interface
query = st.text_input("Ask a question about Sherlock Holmes or Elementary scripts:")
top_k = st.slider("Number of passages to retrieve:", 1, 20, 3)

if st.button("Search") or query:
    # Search execution
    with st.spinner("Retrieving relevant passages..."):
        results = chroma_search(query, collection, corpus="both", top_k=top_k)
    
    # Display results
    if results:
        st.markdown("### Retrieved Passages")
        for i, result in enumerate(results):
            st.write(f"**Result {i+1} | Source:** {result['source'].capitalize()}")
            st.write(f"**Score:** {result['score']:.4f}")
            
            metadata = result.get('metadata', {})
            if result['source'] == 'novel':
                st.write(f"**Novel:** {metadata.get('novel', 'Unknown')}")
            else:
                st.write(f"**Episode:** {metadata.get('episode', 'Unknown')}")
            
            st.write(result['text'])
            st.markdown("---")
        
        # Generate RAG answer
        with st.spinner("Generating answer with LLM..."):
            answer = get_rag_answer(query, [r['text'] for r in results])
        st.markdown("### LLM Answer")
        st.write(answer)
    else:
        st.warning("No results found. Try refining your search terms.")