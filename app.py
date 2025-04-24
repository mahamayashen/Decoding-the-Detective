import streamlit as st
from src.modeling.vector_db import load_vector_database
from src.inference import semantic_search
from src.llm.answer_generation import get_rag_answer


st.set_page_config(page_title="Sherlock Holmes RAG", layout="wide")
st.title("🔎 Sherlock Holmes RAG Explorer")

# Load vector DB once
@st.cache_resource
def get_db():
    return load_vector_database()
    
db = get_db()

st.markdown("#### Example questions:")
st.markdown("- How does Elementary modernize Sherlock's investigation methods?")
st.markdown("- Compare the portrayal of Watson in novels vs TV scripts")
st.markdown("- What forensic technologies appear in Elementary but not the original stories?")
st.markdown("- Show passages where Sherlock struggles emotionally")

query = st.text_input("Ask a question about Sherlock Holmes or Elementary scripts:")

top_k = st.slider("Number of passages to retrieve:", 1, 10, 3)

if st.button("Search") or query:
    with st.spinner("Retrieving relevant passages..."):
        results = semantic_search(query, db, corpus="both", top_k=top_k)
    st.markdown("### Retrieved Passages")
    for i, result in enumerate(results):
        st.write(f"**Result {i+1} | Source:** {result['source'].capitalize()}")
        st.write(f"**Score:** {result['score']:.4f}")
        if result['source'] == 'novel':
            st.write(f"**Novel:** {result['metadata'].get('novel', '')}")
        else:
            st.write(f"**Episode:** {result['metadata'].get('episode', '')}")
        st.write(result['text'])
        st.markdown("---")
    # RAG answer
    with st.spinner("Generating answer with LLM..."):
        answer = get_rag_answer(query, [r['text'] for r in results])
    st.markdown("### LLM Answer")
    st.write(answer)
