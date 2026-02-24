# 📘 PostgreSQL Self-Corrective RAG

Architecture Overview:

User Question
   ↓
Retriever (FAISS)
   ↓
Generator (LLM)
   ↓
Evaluator (LLM-as-critic)
   ↓
Confidence-based Routing
        ├── Accept
        ├── Refine + Retry
        └── Reject

A production-grade **Self-Corrective Retrieval-Augmented Generation (RAG)** system built using:

- LangGraph
- FAISS
- HuggingFace Embeddings
- Groq LLM
- Streamlit

## ✨ Features

- Confidence-aware answers
- Automatic self-refinement
- Hallucination rejection
- PostgreSQL documentation grounding
- Streamlit UI
- Hugging Face Spaces ready

## 🚀 Run Locally

```bash
git clone https://github.com/<your-username>/postgresql-self-corrective-rag.git
cd postgresql-self-corrective-rag

python -m venv .venv
.venv\Scripts\activate   # Windows

pip install -r requirements.txt

streamlit run app.py