from transformers import pipeline, AutoTokenizer
from huggingface_hub import login
import os

login(token=os.environ["HUGGINGFACE_TOKEN"])

# Initialize the pipeline once (not on every query)
qa_pipeline = pipeline(
    "text-generation",
    model="gpt2",
    device_map="auto"
)

def get_rag_answer(question: str, contexts: list[str], max_new_tokens: int = 256) -> str:
    """Generate answers using retrieved context"""
    try:
        context_text = "\n\n".join(contexts)
        prompt = f"""Answer the question using the context about Sherlock Holmes adaptations.
        Context:
        {context_text}
        Question: {question}
        Answer:"""
        
        response = qa_pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.7  # Balance creativity/factuality
        )
        return response[0]['generated_text'].split("Answer:")[-1].strip()
    
    except Exception as e:
        return f"Error generating answer: {str(e)}"
