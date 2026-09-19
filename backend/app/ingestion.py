
import json
import re
import uuid

import chromadb
import networkx as nx

from openai import OpenAI
from pypdf import PdfReader

from .config import (
    OPENAI_API_KEY,
    OPENAI_CHAT_MODEL,
    OPENAI_EMBEDDING_MODEL,
    CHROMA_PATH,
    CHROMA_COLLECTION,
)

from .graph_store import merge_graph


client = OpenAI(
    api_key=OPENAI_API_KEY
)


chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)


collection = chroma_client.get_or_create_collection(
    name=CHROMA_COLLECTION,
    metadata={
        "hnsw:space": "cosine"
    }
)


def extract_text(
    file_path: str
) -> str:
    """
    Extract text from a PDF file.
    """

    reader = PdfReader(file_path)

    pages = []

    for page in reader.pages:

        text = page.extract_text() or ""

        if text.strip():

            pages.append(text)

    return "\n".join(pages)


def chunk_text(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 200
) -> list[str]:
    """
    Split text into overlapping chunks.
    """

    if overlap >= chunk_size:

        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    if not text:

        return []

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = min(
            start + chunk_size,
            text_length
        )

        chunk = text[start:end].strip()

        if chunk:

            chunks.append(chunk)

        if end >= text_length:

            break

        start = end - overlap

    return chunks


def extract_graph(
    text: str
) -> nx.DiGraph:
    """
    Extract standalone entities and relationships.
    """

    prompt = f"""
Extract meaningful entities and relationships
from the following document.

Return ONLY valid JSON in this format:

{{
  "entities": [
    "Kafka",
    "Redis",
    "API Gateway"
  ],
  "relationships": [
    {{
      "source": "entity A",
      "relation": "relationship",
      "target": "entity B"
    }}
  ]
}}

Rules:

- Extract ALL meaningful entities, even if they
  have no relationships.
- Include technologies, systems, services,
  databases, infrastructure, platforms,
  and architectural components.
- Extract technologies such as Kafka, Redis,
  Docker, Kubernetes, PostgreSQL, and API Gateway
  whenever they appear in the document.
- Extract product names, services, applications,
  and important architectural components.
- Normalize entity names consistently.
- Extract meaningful actions and relationships.
- Do not use "co_occurs" as a relationship.
- Do not invent information.
- Use only information found in the document.
- Keep relationship names short and descriptive.
- Every relationship source and target must also
  be included in the entities list.
- Do not include duplicate entities.

DOCUMENT:

{text}
"""

    response = client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,

        messages=[
            {
                "role": "system",
                "content": (
                    "You extract structured knowledge "
                    "graphs from documents. "
                    "Return valid JSON only."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0,

        response_format={
            "type": "json_object"
        }
    )

    content = (
        response.choices[0].message.content
        or "{}"
    )

    try:

        data = json.loads(content)

    except json.JSONDecodeError as error:

        raise ValueError(
            "The LLM returned invalid graph JSON"
        ) from error

    graph = nx.DiGraph()

    entities = data.get(
        "entities",
        []
    )

    if not isinstance(entities, list):

        entities = []

    for entity in entities:

        if not isinstance(entity, str):

            continue

        entity_name = entity.strip()

        if not entity_name:

            continue

        graph.add_node(
            entity_name,
            type="entity"
        )

    relationships = data.get(
        "relationships",
        []
    )

    if not isinstance(relationships, list):

        relationships = []

    for item in relationships:

        if not isinstance(item, dict):

            continue

        source = str(
            item.get(
                "source",
                ""
            )
        ).strip()

        relation = str(
            item.get(
                "relation",
                ""
            )
        ).strip()

        target = str(
            item.get(
                "target",
                ""
            )
        ).strip()

        if (
            not source
            or not relation
            or not target
        ):

            continue

        graph.add_node(
            source,
            type="entity"
        )

        graph.add_node(
            target,
            type="entity"
        )

        graph.add_edge(
            source,
            target,
            relation=relation
        )

    return graph


def store_vectors(
    chunks: list[str],
    filename: str
) -> int:
    """
    Generate embeddings and store chunks.
    """

    if not chunks:

        return 0

    response = client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=chunks
    )

    embeddings = [
        item.embedding
        for item in response.data
    ]

    ids = [
        str(uuid.uuid4())
        for _ in chunks
    ]

    metadatas = [
        {
            "filename": filename,
            "chunk_index": index
        }
        for index in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(chunks)


def ingest(
    file_path: str,
    filename: str
) -> dict:
    """
    Extract text, store vectors,
    and build the knowledge graph.
    """

    text = extract_text(
        file_path
    )

    if not text.strip():

        raise ValueError(
            "No text could be extracted "
            "from the document"
        )

    chunks = chunk_text(
        text
    )

    vector_count = store_vectors(
        chunks=chunks,
        filename=filename
    )

    graph = extract_graph(
        text
    )

    print(
        "Extracted graph:",
        graph
    )

    print(
        "Graph entities:",
        list(graph.nodes)
    )

    print(
        "Graph relationships:",
        list(
            graph.edges(
                data=True
            )
        )
    )

    merge_graph(
        graph
    )

    return {
        "chunks": vector_count,
        "entities": graph.number_of_nodes(),
        "relationships": graph.number_of_edges()
    }