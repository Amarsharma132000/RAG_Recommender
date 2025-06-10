#!/bin/bash
# Start Streamlit (on the Render-provided port) in the background
streamlit run app/frontend/streamlit_app.py --server.port $PORT --server.address 0.0.0.0
