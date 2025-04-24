# src/llm/answer_generation.py
import os
import torch
from threading import Lock
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_cpp import Llama

# Configuration
MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"
QUANTIZED_MODEL = os.path.abspath("Phi-3-mini-4k-instruct-q4.gguf")
llama_lock = Lock()  # Thread safety for GGUF model

def load_optimized_model():
    """Hardware-aware model loading with validation"""
    if not os.path.exists(QUANTIZED_MODEL):
        raise FileNotFoundError(f"GGUF model missing at {QUANTIZED_MODEL}")
    
    if torch.cuda.is_available():
        return AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True
        )
    else:
        # Optimized for Apple Silicon
        return Llama(
            model_path=QUANTIZED_MODEL,
            n_ctx=4096,
            n_threads=max(4, os.cpu_count()-2),
            n_gpu_layers=1 if torch.backends.mps.is_available() else 0,
            offload_kqv=True,
            use_mlock=True,
            verbose=False
        )

def validate_response(answer: str) -> str:
    """Ensure response adheres to data constraints"""
    forbidden_terms = ["BBC", "Benedict Cumberbatch", "Sherlock BBC"]
    if any(term in answer for term in forbidden_terms):
        return "I specialize in CBS' Elementary and original novels. Ask me about those!"
    return answer.split("<|end|>")[0].strip()

def get_rag_answer(question: str, contexts: list[str]) -> str:
    """Generate context-grounded answers with safeguards"""
    model = load_optimized_model()
    context_chunks = "\n".join([f"[[CONTEXT {i+1}]] {c}" for i,c in enumerate(contexts)])
    
    prompt = f"""<|system|>
    **Answering Rules**
    1. Base answers mainly on these contexts:
    {context_chunks}
    2. Never mention BBC productions<|end|>
    3. Make sure you sound passionate about the CBS show Elementary and the original novels<|end|>
    4. If the user ask you something outside of the scope of these two works, say "I don't know"<|end|>
    5. Make sure NOT TO HALLUCINATE<|end|>
    6. Make sure you complete your answer in 150 words or less<|end|>
    
    <|user|>
    {question}<|end|>
    <|assistant|>"""
    
    try:
        if isinstance(model, Llama):
            with llama_lock:  # Thread-safe execution
                response = model.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=256,
                    repeat_penalty=1.1
                )
            answer = response['choices'][0]['message']['content']
        else:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return validate_response(answer)
    
    except Exception as e:
        return f"Error generating answer: {str(e)}"
