from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):

    question: str

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )


class QueryResponse(BaseModel):

    answer: str

    vector_context: list[Any] = Field(
        default_factory=list
    )

    graph_context: list[Any] = Field(
        default_factory=list
    )