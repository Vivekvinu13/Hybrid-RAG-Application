# Hybrid RAG: FastAPI + Streamlit

## Architecture
1. Upload file to FastAPI `/ingest`.
2. Text is chunked and embedded into Chroma (vector DB).
3. Entities and lightweight co-occurrence relationships are written to NetworkX JSON (knowledge graph).
4. `/query` searches Chroma and the graph independently.
5. Both contexts are passed to the LLM to produce one grounded answer.

## Run
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env
# add OPENAI_API_KEY to .env
uvicorn backend.app.main:app --reload --port 8000
streamlit run frontend/streamlit_app.py
```

## Production upgrades
- Replace regex graph extraction with structured LLM extraction.
- Use Neo4j or another graph DB for scale and graph traversal.
- Add document IDs, tenant isolation, authentication, citations, reranking, and background jobs.
