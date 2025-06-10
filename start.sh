#!/bin/bash
# Start FastAPI (on port 8000) and Streamlit (on port 8501) in the background
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 &
streamlit run app/frontend/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
