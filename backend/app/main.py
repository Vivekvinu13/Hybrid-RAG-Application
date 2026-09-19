from pathlib import Path
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile

from .ingestion import ingest

from .rag import (
    vector_search,
    graph_search,
    answer,
)

from .guardrails import apply_input_guardrails

from .schema import (
    QueryRequest,
    QueryResponse,
)


# --------------------------------------------------
# APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="Hybrid RAG API",
    description="Vector DB + Knowledge Graph RAG",
)


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok"
    }


# --------------------------------------------------
# INGEST DOCUMENT
# --------------------------------------------------

@app.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
):
    filename = file.filename or ""

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            delete=False,
        ) as temporary_file:

            temporary_file.write(file_bytes)

            temporary_path = Path(
                temporary_file.name
            )

        result = ingest(
            str(temporary_path),
            filename,
        )

        return {
            "filename": filename,
            "result": result,
        }

    except ValueError as error:

        print(
            f"Ingestion validation error: {error}"
        )

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:

        print(
            f"Ingestion error: {error}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {error}",
        )

    finally:

        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
            temporary_path.unlink()


# --------------------------------------------------
# QUERY DOCUMENTS
# --------------------------------------------------

@app.post(
    "/query",
    response_model=QueryResponse,
)
def query(request: QueryRequest):

    # ----------------------------------------------
    # INPUT GUARDRAILS
    # ----------------------------------------------

    try:

        question = apply_input_guardrails(
            request.question
        )

    except ValueError as error:

        print(
            f"Guardrail rejection: {error}"
        )

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    # ----------------------------------------------
    # RETRIEVAL AND ANSWER GENERATION
    # ----------------------------------------------

    try:

        vectors = vector_search(
            question=question,
            top_k=request.top_k,
        ) or []

        graph = graph_search(
            question
        ) or []

        print(
            "API vector results:",
            len(vectors),
        )

        print(
            "API graph results:",
            len(graph),
        )

        generated_answer = answer(
            question=question,
            vectors=vectors,
            graph=graph,
        )

        return QueryResponse(
            answer=generated_answer or "",
            vector_context=vectors,
            graph_context=graph,
        )

    except Exception as error:

        print(
            f"Query error: {error}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {error}",
        )