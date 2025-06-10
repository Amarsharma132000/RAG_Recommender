# load_model.py - Simplified with better efficient models

from sentence_transformers import SentenceTransformer
import os

# Set your Hugging Face token
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN environment variable not set.")

def load_embedding_model():
    """
    Loads the best efficient embedding model available
    Options tried in order:
    1. BAAI/bge-small-en-v1.5 (384d, best balance)
    2. thenlper/gte-small (384d, great performance)
    3. sentence-transformers/all-MiniLM-L12-v2 (384d, fallback)
    """
    model_options = [
        # "BAAI/bge-small-en-v1.5",  # Best balance (384d)
        "thenlper/gte-small",      # Great performance (384d)
        "sentence-transformers/all-MiniLM-L12-v2"  # Reliable fallback
    ]
    
    for model_name in model_options:
        try:
            print(f"Attempting to load {model_name}...")
            model = SentenceTransformer(model_name)
            
            # Quick test embedding to verify it works
            test_embed = model.encode("test")
            if len(test_embed) == 384:  # All these models use 384 dimensions
                print(f"Successfully loaded {model_name}")
                return model
                
        except Exception as e:
            print(f"Failed to load {model_name}: {str(e)}")
            continue
    
    raise RuntimeError("Could not load any embedding model")

# For backward compatibility with your existing code
def load_finetuned_model():
    """Maintains same interface but uses better public models"""
    return load_embedding_model()