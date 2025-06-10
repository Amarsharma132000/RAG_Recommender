# rag_engine.py — Updated with 3-Way Retrieval Fusion (BM25 + Flat + HNSW)

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from utils import cosine_sim, build_faiss_index, load_json_data, chunk_documents
from models.load_model import load_finetuned_model # Assuming you have this
from typing import List, Dict

class SHLRAGEngine:
    def __init__(self, data_path: str, semantic_index_type: str = "hnsw"):
        self.semantic_index_type = semantic_index_type
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.model = load_finetuned_model()
        self.assessments = load_json_data(data_path)

        # --- Data Preparation ---
        self.structured_chunks = []
        self.structured_lookup = []
        for item in self.assessments:
            structured_chunk = (
                f"Assessment Name: {item.get('Assessment Name', '')}\n"
                f"Test Types: {item.get('Test Types', '')}\n"
                f"Description: {item.get('Description', '')}\n"
                f"Remote Testing: {item.get('Remote Testing', '')}\n"
                f"Adaptive/IRT: {item.get('Adaptive/IRT', '')}\n"
                f"Duration: {item.get('Duration (minutes)', '')}"
            )
            self.structured_chunks.append(structured_chunk)
            self.structured_lookup.append({**item, 'provenance': 'structured'})

        self.semantic_chunks = []
        self.semantic_lookup = []
        for idx, item in enumerate(self.assessments):
            full_text = (
                f"Assessment Name: {item.get('Assessment Name', '')}\n"
                f"Test Types: {item.get('Test Types', '')}\n"
                f"Description: {item.get('Description', '')}\n"
                f"Remote Testing: {item.get('Remote Testing', '')}\n"
                f"Adaptive/IRT: {item.get('Adaptive/IRT', '')}\n"
                f"Duration: {item.get('Duration (minutes)', '')}"
            )
            for chunk in chunk_documents([full_text]):
                self.semantic_chunks.append(chunk)
                self.semantic_lookup.append({**item, 'provenance': 'semantic', 'parent_idx': idx})

        # --- Index Building ---
        print("Encoding embeddings...")
        self.structured_emb = np.array(self.model.encode(self.structured_chunks, show_progress_bar=True))
        self.semantic_emb = np.array(self.model.encode(self.semantic_chunks, show_progress_bar=True))
        
        dim = self.structured_emb.shape[1]

        # --- RETRIEVER 1: Flat Index for high-accuracy search on structured chunks ---
        print("Building Structured Data Index (Flat)...")
        self.index_struct_flat = build_faiss_index(self.structured_emb, dim, index_type="flat")
        
        # --- RETRIEVER 2: HNSW Index for fast, high-recall search on semantic chunks ---
        print(f"Building Semantic Chunk Index ({self.semantic_index_type.upper()})...")
        self.index_semantic_ann = build_faiss_index(self.semantic_emb, dim, index_type=self.semantic_index_type)

        # --- RETRIEVER 3: BM25 Index for keyword search on structured chunks ---
        print("Building BM25 Index...")
        self.bm25_corpus = [chunk.lower().split() for chunk in self.structured_chunks]
        self.bm25_model = BM25Okapi(self.bm25_corpus)
        
        print("All retrieval engines are ready.")

    def _search_index(self, index, emb: np.ndarray, lookup: List[Dict], chunks: List[str], top_k: int = 10) -> List[Dict]:
        _, I = index.search(emb, top_k)
        results = []
        for idx in I[0]:
            if 0 <= idx < len(lookup):
                item = dict(lookup[idx])
                item['retrieved_chunk'] = chunks[idx]
                results.append(item)
        return results

    def _bm25_search(self, query: str, top_k: int = 10) -> List[Dict]:
        tokenized_query = query.lower().split()
        scores = self.bm25_model.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            item = dict(self.structured_lookup[idx])
            item['retrieved_chunk'] = self.structured_chunks[idx]
            item['bm25_score'] = float(scores[idx])
            item['provenance'] = 'bm25'
            results.append(item)
        return results

    def _rerank(self, query: str, candidates: List[Dict], top_k: int = 10) -> List[Dict]:
        if not candidates: return []
            
        bm25_scores = np.array([item.get('bm25_score', 0.0) for item in candidates])
        if bm25_scores.max() > bm25_scores.min():
            bm25_scores_normalized = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min())
        else:
            bm25_scores_normalized = np.zeros_like(bm25_scores)
            
        for i, item in enumerate(candidates):
            item['bm25_norm_score'] = bm25_scores_normalized[i]

        rerank_inputs = [[query, item['retrieved_chunk']] for item in candidates]
        ce_scores = self.cross_encoder.predict(rerank_inputs, show_progress_bar=False)
        
        for i, item in enumerate(candidates):
            item['cross_score'] = float(ce_scores[i])
            item['final_score'] = (0.7 * item['cross_score']) + (0.3 * item['bm25_norm_score'])

        return sorted(candidates, key=lambda x: x['final_score'], reverse=True)[:top_k]

    def expand_query(self, query: str) -> List[str]:
        return [query]

    def recommend(self, query: str, top_k: int = 10) -> List[Dict]:
        queries = self.expand_query(query)
        all_results = []
        for q in queries:
            query_emb = self.model.encode([q])
            
            # --- FUSION: Get results from all 3 retrieval sources ---
            # 1. High-accuracy semantic search on structured data
            results_struct = self._search_index(self.index_struct_flat, query_emb, self.structured_lookup, self.structured_chunks, top_k=top_k)
            # 2. Fast, high-recall semantic search on detailed chunks
            results_semantic = self._search_index(self.index_semantic_ann, query_emb, self.semantic_lookup, self.semantic_chunks, top_k=top_k)
            # 3. Keyword search on structured data
            results_bm25 = self._bm25_search(q, top_k=top_k)
            
            combined = results_struct + results_semantic + results_bm25
            all_results.extend(combined)

        # Deduplicate results before sending to the expensive reranker
        seen_names = set()
        merged = []
        for item in all_results:
            # Deduplicate based on Assessment Name (or another unique field)
            name = item.get('Assessment Name', '').strip().lower()
            if name and name not in seen_names:
                seen_names.add(name)
                merged.append(item)

        reranked = self._rerank(query, merged, top_k=top_k*2)  # Get more for diversity selection
        # Final deduplication by URL (fallback to name)
        seen_final = set()
        unique_results = []
        for item in reranked:
            url = item.get('URL', '').strip().lower()
            name = item.get('Assessment Name', '').strip().lower()
            key = url if url else name
            if key and key not in seen_final:
                seen_final.add(key)
                unique_results.append(item)

        # --- MMR Diversity Filtering ---
        def mmr_select(items, embeddings, top_k, lambda_param=0.7):
            selected = []
            selected_idx = []
            if not items:
                return []
            # Compute pairwise cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            sim_matrix = cosine_similarity(embeddings)
            query_sim = sim_matrix[0]  # similarity to query embedding (first row)
            selected_idx.append(0)
            selected.append(items[0])
            while len(selected) < min(top_k, len(items)):
                remaining = [i for i in range(len(items)) if i not in selected_idx]
                mmr_scores = []
                for i in remaining:
                    max_sim = max([sim_matrix[i][j] for j in selected_idx])
                    mmr_score = lambda_param * query_sim[i] - (1 - lambda_param) * max_sim
                    mmr_scores.append((mmr_score, i))
                mmr_scores.sort(reverse=True)
                next_idx = mmr_scores[0][1]
                selected_idx.append(next_idx)
                selected.append(items[next_idx])
            return selected

        # Prepare embeddings for MMR (use cross_encoder score as proxy if no embedding)
        # Here, use semantic embedding for diversity
        if unique_results:
            from sentence_transformers import util
            query_emb = self.model.encode([query])
            item_embs = self.model.encode([item.get('retrieved_chunk', '') for item in unique_results])
            mmr_results = mmr_select(unique_results, np.vstack([query_emb, item_embs]), top_k)
        else:
            mmr_results = unique_results[:top_k]
        return mmr_results