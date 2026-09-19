import json
import re
import tempfile
from pathlib import Path

import networkx as nx

from .config import GRAPH_PATH


STOPWORDS = {
    "what",
    "is",
    "are",
    "the",
    "a",
    "an",
    "of",
    "to",
    "for",
    "in",
    "on",
    "and",
    "or",
    "how",
    "does",
    "do",
    "with",
    "about",
    "explain",
    "tell",
    "me",
    "feature",
    "features",
    "used",
    "use",
    "role",
    "system",
}


def normalize_text(
    text: str
) -> str:
    """
    Normalize text for matching.
    """

    text = str(text).lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def tokenize(
    text: str
) -> set[str]:
    """
    Convert text into meaningful tokens.
    """

    normalized = normalize_text(text)

    words = normalized.split()

    return {
        word
        for word in words
        if (
            word not in STOPWORDS
            and len(word) > 1
        )
    }


def load_graph() -> nx.DiGraph:
    """
    Load the persistent knowledge graph.

    Returns an empty graph if the file is missing
    or corrupted.
    """

    graph = nx.DiGraph()

    graph_path = Path(GRAPH_PATH)

    if not graph_path.exists():

        return graph

    try:

        data = json.loads(
            graph_path.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(data, dict):

            print(
                "Graph load warning: "
                "Invalid JSON format"
            )

            return graph

        graph = nx.node_link_graph(
            data,
            directed=True,
            edges="links"
        )

        return graph

    except Exception as error:

        print(
            f"Graph load warning: {error}"
        )

        return nx.DiGraph()


def save_graph(
    graph: nx.DiGraph
):
    """
    Save the graph atomically using
    a temporary file.
    """

    graph_path = Path(GRAPH_PATH)

    graph_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    data = nx.node_link_data(
        graph,
        edges="links"
    )

    serialized = json.dumps(
        data,
        indent=2,
        ensure_ascii=False
    )

    temporary_path = None

    try:

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=graph_path.parent,
            prefix=f"{graph_path.name}.",
            suffix=".tmp",
            delete=False
        ) as temporary_file:

            temporary_file.write(
                serialized
            )

            temporary_file.flush()

            temporary_path = Path(
                temporary_file.name
            )

        temporary_path.replace(
            graph_path
        )

    finally:

        if (
            temporary_path is not None
            and temporary_path.exists()
        ):

            temporary_path.unlink()


def merge_graph(
    new_graph: nx.DiGraph
):
    """
    Merge newly extracted entities and
    relationships into the persistent graph.
    """

    graph = load_graph()

    graph.add_nodes_from(
        new_graph.nodes(
            data=True
        )
    )

    graph.add_edges_from(
        new_graph.edges(
            data=True
        )
    )

    save_graph(graph)


def calculate_entity_score(
    question_tokens: set[str],
    question: str,
    entity: str,
    graph: nx.DiGraph
) -> int:
    """
    Calculate the relevance score of an entity.
    """

    entity_tokens = tokenize(
        entity
    )

    if not entity_tokens:

        return 0

    overlap = (
        question_tokens
        & entity_tokens
    )

    score = len(overlap) * 3

    normalized_entity = normalize_text(
        entity
    )

    normalized_question = normalize_text(
        question
    )

    # Exact entity phrase match
    if (
        normalized_entity
        and normalized_entity
        in normalized_question
    ):

        score += 5

    # Match relationship labels
    for source, target, data in graph.edges(
        data=True
    ):

        if (
            source == entity
            or target == entity
        ):

            relation = str(
                data.get(
                    "relation",
                    ""
                )
            )

            relation_tokens = tokenize(
                relation
            )

            relation_overlap = (
                question_tokens
                & relation_tokens
            )

            score += len(
                relation_overlap
            )

    return score


def get_neighbors(
    graph: nx.DiGraph,
    entity: str
) -> list[dict]:
    """
    Return incoming and outgoing relationships.
    """

    neighbors = []

    seen = set()

    # Outgoing relationships
    for neighbor in graph.successors(
        entity
    ):

        edge_data = (
            graph.get_edge_data(
                entity,
                neighbor
            )
            or {}
        )

        relation = edge_data.get(
            "relation",
            "related_to"
        )

        item = {
            "entity": str(neighbor),
            "relation": str(relation),
            "direction": "outgoing"
        }

        key = (
            item["entity"],
            item["relation"],
            item["direction"]
        )

        if key not in seen:

            seen.add(key)

            neighbors.append(item)

    # Incoming relationships
    for neighbor in graph.predecessors(
        entity
    ):

        edge_data = (
            graph.get_edge_data(
                neighbor,
                entity
            )
            or {}
        )

        relation = edge_data.get(
            "relation",
            "related_to"
        )

        item = {
            "entity": str(neighbor),
            "relation": str(relation),
            "direction": "incoming"
        }

        key = (
            item["entity"],
            item["relation"],
            item["direction"]
        )

        if key not in seen:

            seen.add(key)

            neighbors.append(item)

    return neighbors


def search_graph(
    question: str,
    limit: int = 10
) -> list[dict]:
    """
    Search the knowledge graph using
    normalized token matching.
    """

    graph = load_graph()

    if graph.number_of_nodes() == 0:

        return []

    question_tokens = tokenize(
        question
    )

    if not question_tokens:

        return []

    scored_entities = []

    for entity in graph.nodes:

        score = calculate_entity_score(
            question_tokens=question_tokens,
            question=question,
            entity=str(entity),
            graph=graph
        )

        if score > 0:

            scored_entities.append(
                (
                    entity,
                    score
                )
            )

    scored_entities.sort(
        key=lambda item: item[1],
        reverse=True
    )

    results = []

    for entity, score in scored_entities[:limit]:

        neighbors = get_neighbors(
            graph,
            entity
        )

        results.append(
            {
                "entity": str(entity),
                "score": score,
                "neighbors": neighbors
            }
        )

    # Debug output must be outside the loop
    print(
        "Graph search question:",
        question
    )

    print(
        "Graph search tokens:",
        question_tokens
    )

    print(
        "Graph search scored entities:",
        scored_entities
    )

    print(
        "Graph search results:",
        results
    )

    return results