# api/main.py — Improved for Production Use

import os
import asyncio
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import List, Dict

# Adjust sys.path to import from the project root
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.rag_engine import SHLRAGEngine
from app.extract_query import extract_query_text

# --- Centralized Application State ---
class AppState:
    rag_engine: SHLRAGEngine = None

app_state = AppState()

# --- FastAPI Lifespan Events to Load Models on Startup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events. The RAG engine is loaded once on startup.
    """
    # Load config from environment variables for flexibility
    # Default to 'hnsw' as the recommended index type
    data_path = os.getenv("DATA_PATH", "app/data/dataset.json")
    index_type = os.getenv("SEMANTIC_INDEX_TYPE", "hnsw")
    
    print("--- API Startup ---")
    logging.info(f"Loading RAG engine with data from '{data_path}' and semantic index type '{index_type}'...")
    
    app_state.rag_engine = SHLRAGEngine(data_path=data_path, semantic_index_type=index_type)
    
    logging.info("RAG engine loaded successfully.")
    yield
    print("--- API Shutdown ---")


# --- FastAPI App Initialization ---
app = FastAPI(
    title="SHL Assessment Recommender API", 
    version="1.2",
    description="An advanced API to recommend SHL assessments by fusing keyword and semantic search.",
    lifespan=lifespan
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# CORS middleware for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for Input/Output Validation ---
class QueryInput(BaseModel):
    query: str = Field(..., example="I need an assessment for a senior python developer role.", min_length=10)

class QueryOutput(BaseModel):
    processed_query: str
    recommendations: List[Dict]

# --- API Endpoints ---
@app.post("/recommend", response_model=QueryOutput)
async def recommend_assessments(input_data: QueryInput):
    """
    Accepts a query or job description URL, processes it, and returns relevant assessments.
    This endpoint is asynchronous and handles long-running tasks in a non-blocking way.
    """
    logging.info(f"Received query: '{input_data.query[:100]}...'")
    if not app_state.rag_engine:
        raise HTTPException(status_code=503, detail="RAG Engine is not available.")
        
    try:
        # Run potentially blocking I/O (like web scraping) in a separate thread
        processed_query = await asyncio.to_thread(extract_query_text, input_data.query)
        logging.info(f"Processed query: '{processed_query}'")

        # Run the CPU-bound recommendation engine in a separate thread
        results = await asyncio.to_thread(app_state.rag_engine.recommend, processed_query)
        logging.info(f"Found {len(results)} recommendations.")
        
        return {"processed_query": processed_query, "recommendations": results}
        
    except Exception as e:
        logging.error(f"An error occurred processing the request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred.")

@app.get("/health", summary="Health Check")
async def health_check():
    """Provides a simple health check to confirm the API is running."""
    return {"status": "ok"}

@app.get("/")
async def root():
    """Landing page for the API."""
    return {"message": "Welcome to the SHL Assessment Recommender API. Visit /docs for interactive documentation."}