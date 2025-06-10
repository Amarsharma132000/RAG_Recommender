# frontend/streamlit_app.py

import sys
import os
# Ensure the app directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import streamlit as st
from rag_engine import SHLRAGEngine
from extract_query import extract_query_text

# --- Page Configuration ---
st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="🤖",
    layout="wide"
)

# --- Engine Loading ---
@st.cache_resource
def load_engine():
    """
    Loads the RAG engine once and caches it for the session.
    Uses HNSW as the recommended index for best performance.
    """
    data_path = "app/data/dataset.json"
    if not os.path.exists(data_path):
        st.error(f"Error: Data file not found at '{data_path}'. Please ensure the path is correct.")
        return None
    
    engine = SHLRAGEngine(data_path=data_path, semantic_index_type="hnsw")
    return engine

engine = load_engine()

# --- UI Layout ---
st.title("🔍 SHL Assessment Recommender")
st.write("Enter a job description, query or LinkedIn job post URL to get relevant SHL assessments.")

query_input = st.text_area("Paste query or job description URL:", height=150)

if st.button("🔎 Recommend Assessments") and query_input:
    with st.spinner("Extracting query..."):
        processed_query = extract_query_text(query_input)
        st.markdown("**🔎 Processed Query for Recommender:**")
        st.code(processed_query, language="markdown")

    with st.spinner("Running retrieval & reranking..."):
        results = engine.recommend(processed_query,top_k=10)

    if not results:
        st.warning("No assessments found. Try refining your query.")
    else:
        table_rows = []
        for r in results[:10]:
            name = r.get("Assessment Name", "Unnamed")
            url = r.get("URL", "#")
            remote = "Yes" if r.get("Remote Testing", False) else "No"
            adaptive = "Yes" if r.get("Adaptive/IRT", False) else "No"
            duration = r.get("Duration (minutes)", "N/A")
            test_type = r.get("Test Types", "N/A")

            table_rows.append({
                "Assessment Name": f"[{name}]({url})",
                "Remote Support": remote,
                "Adaptive/IRT": adaptive,
                "Duration (min)": duration,
                "Test Type": test_type
            })

        df = pd.DataFrame(table_rows)
        st.markdown("### ✅ Recommended Assessments")
        st.markdown(df.to_markdown(index=False), unsafe_allow_html=True)
        st.markdown("You can click on the assessment names to view more details or start the assessment.")