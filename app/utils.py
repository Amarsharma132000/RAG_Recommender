# utils.py — Updated with HNSW Index Support

import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
import os
import json
from langchain_core.documents import Document
import math

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_json_data(path: str) -> List[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def chunk_documents(docs: List[str], chunk_size=200, chunk_overlap=0) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "]
    )
    documents = [Document(page_content=doc) for doc in docs]
    chunks = splitter.split_documents(documents)
    return [chunk.page_content for chunk in chunks]

def build_faiss_index(embeddings: np.ndarray, dim: int, index_type: str = "hnsw"):
    """
    Builds a FAISS index with support for Flat, IVF, and HNSW.
    """
    index_type = index_type.lower()
    
    if index_type == "flat":
        index = faiss.IndexFlatL2(dim)
        
    elif index_type == "ivf":
        nlist = int(math.sqrt(embeddings.shape[0]))
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist)
        print(f"Training FAISS index of type 'ivf'...")
        index.train(embeddings)
        
    # --- IMPROVEMENT: Added HNSW support ---
    elif index_type == "hnsw":
        # HNSW does not require training, it's added directly.
        # M is the number of neighbors per vertex in the graph.
        M = 32 
        index = faiss.IndexHNSWFlat(dim, M)
        
    else:
        raise ValueError(f"Unknown index_type: {index_type}. Use 'flat', 'ivf', or 'hnsw'.")

    print(f"Adding embeddings to FAISS index of type '{index_type}'...")
    index.add(embeddings)
    print("Index is ready.")
    return index