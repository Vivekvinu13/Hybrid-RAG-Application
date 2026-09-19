import streamlit as st, requests, os

API=os.getenv("BACKEND_URL","http://localhost:8000")
st.set_page_config(page_title="Hybrid RAG", page_icon="🔎")
st.title("🔎 Hybrid RAG: Vector DB + Knowledge Graph")
st.caption("Pipeline 1: upload and ingest → Pipeline 2: retrieve from both → generate one answer")

uploaded=st.file_uploader("Upload a document", type=["pdf","docx","txt","md","html"])
if uploaded and st.button("Ingest document"):
    with st.spinner("Running vector + graph ingestion..."):
        r=requests.post(f"{API}/ingest",files={"file":(uploaded.name,uploaded.getvalue())},timeout=180)
    if r.ok: st.success(r.json())
    else: st.error(r.text)

st.divider()
question=st.text_area("Ask a question about your uploaded documents", height=100)
if st.button("Ask") and question.strip():
    with st.spinner("Searching vector database and knowledge graph..."):
        r=requests.post(f"{API}/query",json={"question":question,"top_k":5},timeout=180)
    if r.ok:
        data=r.json(); st.subheader("Answer"); st.write(data["answer"])
        with st.expander("Vector evidence"): st.json(data["vector_context"])
        with st.expander("Knowledge graph evidence"): st.json(data["graph_context"])
    else: st.error(r.text)
