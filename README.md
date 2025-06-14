# RAG_Recommender
SHL Assessment Recommender using RAG
This project is an advanced recommendation system designed to suggest relevant SHL assessments based on a user's query, which can be a simple text description or a URL to a LinkedIn job posting. It leverages a sophisticated Retrieval-Augmented Generation (RAG) pipeline with a hybrid search mechanism to provide accurate and contextually relevant recommendations.

✨ Core Features
Custom Data Scraping: The core dataset.json was built using a custom Python script (scraper3.py) with Selenium to gather comprehensive data directly from the SHL product catalog.

Hybrid Search Engine: Implements a 3-way retrieval fusion combining lexical search (BM25) with dense vector search (FAISS with HNSW) for robust and accurate retrieval.

LinkedIn Job Scraping: Automatically extracts key details (responsibilities, skills, qualifications) from a LinkedIn job URL to form a detailed query.

Advanced Reranking: Utilizes a Cross-Encoder model to rerank the retrieved results for higher relevance and applies Maximal Marginal Relevance (MMR) to ensure diversity in the final recommendations.

Interactive Web Interface: A user-friendly frontend built with Streamlit allows for easy interaction and visualization of the results.

Cloud Deployment: Successfully deployed and hosted on Streamlit Cloud, with Docker and Render configurations also available.

⚙️ How It Works: The RAG Pipeline
The system follows a multi-stage process to deliver recommendations:

Query Extraction (extract_query.py):

The user inputs either plain text or a LinkedIn job URL.

If a URL is detected, Selenium and BeautifulSoup are used to scrape the job description.

A language model then extracts structured information (key responsibilities, skills, education) from the raw text.

This structured data is compiled into a detailed, high-quality query.

Hybrid Retrieval (rag_engine.py):

The processed query is sent to the SHLRAGEngine.

Lexical Search: BM25Okapi performs a keyword-based search over the assessment data.

Semantic Search: A pre-trained sentence transformer (thenlper/gte-small) encodes the query into an embedding. This embedding is used to search two FAISS vector indexes:

IndexFlatL2: A flat index for exact, high-accuracy search on structured data chunks.

IndexHNSWFlat: An HNSW index for fast and efficient search on larger semantic chunks.

The results from all three retrievers are fused together.

Reranking and Diversification (rag_engine.py):

The combined list of candidates is passed to a CrossEncoder model (ms-marco-MiniLM-L-6-v2) which reranks them based on semantic similarity to the original query.

Maximal Marginal Relevance (MMR) is then applied to the reranked list to diversify the results and reduce redundancy.

Presentation (streamlit_app.py):

The final, ranked, and diversified list of SHL assessments is displayed to the user in a clean, interactive table in the Streamlit web app.

🚀 Setup and Installation
Follow these steps to get the project running on your local machine.

Prerequisites:

Python 3.8+

pip for package installation.

A Hugging Face account and an API token.

1. Clone the Repository:

git clone [https://github.com/Amarsharma132000/RAG_Recommender.git]
cd RAG_Recommender

2. Create a Virtual Environment (Recommended):

python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate

3. Install Dependencies:

pip install -r requirements.txt

4. Set Up Environment Variables:
You need a Hugging Face User Access Token. Create a .env file in the root of the project directory and add your token:

HF_TOKEN="your_hugging_face_token_here"

▶️ Usage
To run the interactive web application, execute the following command in your terminal:

streamlit run streamlit_app.py

This will start the Streamlit server and open the application in your default web browser.

☁️ Deployment
This project has been successfully deployed and is live on Streamlit Cloud.

Streamlit Cloud (Live): The application is configured for direct deployment from the GitHub repository via Streamlit's platform.

Docker/Render: The repository also includes a render.yaml for deployment on services like Render.

🛠️ Key Dependencies
streamlit: For creating the interactive web UI.

sentence-transformers: For loading and using the text embedding models.

faiss-cpu: Powers the efficient similarity search in the vector database.

rank_bm25: For the lexical (keyword-based) search component.

langchain: Provides the core framework and utilities for building the RAG pipeline.

selenium & beautifulsoup4: For scraping job descriptions and assessment data.
