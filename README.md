# RAG Bot — LLM + Retrieval-Augmented Generation

Минимальный каркас проекта для чата по документам (PDF/HTML/MD) с RAG.

## Быстрый старт
```bash
conda create -n rag-bot python=3.11 -y
conda activate rag-bot
pip install fastapi uvicorn[standard] qdrant-client sentence-transformers pypdf trafilatura beautifulsoup4 clickhouse-connect openai transformers faiss-cpu streamlit pydantic-settings scikit-learn torch
```