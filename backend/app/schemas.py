from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(min_length=2, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceReference(BaseModel):
    source_id: str
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    score: float
    excerpt: str


class AskResponse(BaseModel):
    query: str
    answer: str
    embedding_backend: str
    sources: list[SourceReference]


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    page_count: int
    chunk_count: int
    embedding_backend: str
    message: str


class DocumentSummary(BaseModel):
    document_id: str
    filename: str
    page_count: int
    chunk_count: int


class HealthResponse(BaseModel):
    status: str
    qdrant: str
    collection: str
