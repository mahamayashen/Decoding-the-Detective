from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import os
import torch

login(token=os.environ["HUGGINGFACE_TOKEN"])

# Initialize model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    torch_dtype=torch.float16,
    token=os.environ["HUGGINGFACE_TOKEN"]
)
tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    padding_side="left",
    token=os.environ["HUGGINGFACE_TOKEN"]
)

def get_rag_answer(question: str, contexts: list[str], max_new_tokens: int = 256) -> str:
    """Generate answers using your original prompt format"""
    try:
        context_text = "\n\n".join(contexts)
        prompt = f"""Answer the question using the context about Sherlock Holmes adaptations.
        Context:
        {context_text}
        Question: {question}
        Answer:"""
        
        # Encode with Mistral's requirements
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            return_attention_mask=True
        ).to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True
        )
        
        # Return full answer (no splitting needed)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    except Exception as e:
        return f"Error generating answer: {str(e)}"
