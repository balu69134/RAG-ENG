from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import get_settings
from backend.app.schemas import (
    AskRequest,
    AskResponse,
    DocumentSummary,
    HealthResponse,
    IngestResponse,
    SourceReference,
)
from backend.app.services.embeddings import EmbeddingService
from backend.app.services.generation import GenerationService
from backend.app.services.ingestion import (
    DocumentProcessingError,
    chunk_pages,
    extract_pdf_pages,
)
from backend.app.services.retrieval import RetrievalService
from backend.app.services.vector_store import VectorStore, VectorStoreError

settings = get_settings()
embeddings = EmbeddingService(settings)
vector_store = VectorStore(settings)
retrieval = RetrievalService(settings, embeddings, vector_store)
generation = GenerationService(settings)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Upload PDFs and ask questions grounded in their contents.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


@app.get("/", tags=["health"])
def root() -> dict[str, str]:
    return {"name": settings.app_name, "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    try:
        vector_store.check_connection()
    except VectorStoreError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    return HealthResponse(
        status="ok",
        qdrant="connected",
        collection=settings.qdrant_collection,
    )


@app.post(
    "/documents/upload",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["documents"],
)
async def upload_document(file: UploadFile = File(...)) -> IngestResponse:
    original_name = file.filename or "document.pdf"
    filename = original_name.replace("\\", "/").rsplit("/", 1)[-1].strip()
    if not filename or not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Upload a PDF file.",
        )

    pdf_bytes = await file.read(settings.max_upload_size_bytes + 1)
    await file.close()
    if len(pdf_bytes) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"PDFs must be {settings.max_upload_size_mb} MB or smaller.",
        )
    if not pdf_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded PDF is empty.",
        )

    try:
        pages = extract_pdf_pages(pdf_bytes)
        chunks = chunk_pages(pages, settings)
        vector_store.ensure_collection(embeddings.vector_size)
        vectors = embeddings.encode([chunk.text for chunk in chunks])
        document_id = str(uuid4())
        vector_store.upsert_chunks(
            document_id=document_id,
            filename=filename,
            page_count=len(pages),
            embedding_backend=embeddings.backend_name,
            chunks=chunks,
            vectors=vectors,
        )
    except DocumentProcessingError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except VectorStoreError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document processing failed. Check the server configuration and logs.",
        ) from exc

    return IngestResponse(
        document_id=document_id,
        filename=filename,
        page_count=len(pages),
        chunk_count=len(chunks),
        embedding_backend=embeddings.backend_name,
        message="PDF processed and indexed successfully.",
    )


@app.get("/documents", response_model=list[DocumentSummary], tags=["documents"])
def list_documents() -> list[DocumentSummary]:
    try:
        documents = vector_store.list_documents()
    except VectorStoreError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    return [DocumentSummary(**document) for document in documents]


@app.delete("/documents/{document_id}", tags=["documents"])
def delete_document(document_id: str) -> dict[str, str | bool]:
    try:
        vector_store.delete_document(document_id)
    except VectorStoreError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    return {"deleted": True, "document_id": document_id}


@app.post("/ask", response_model=AskResponse, tags=["questions"])
def ask_question(request: AskRequest) -> AskResponse:
    try:
        passages = retrieval.retrieve(request.query, request.top_k)
    except VectorStoreError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not create a query embedding. Check the embedding model setup.",
        ) from exc

    sources = [
        SourceReference(
            source_id=f"S{index}",
            document_id=passage["document_id"],
            filename=passage["filename"],
            page_number=passage["page_number"],
            chunk_index=passage["chunk_index"],
            score=round(passage["score"], 4),
            excerpt=passage["text"],
        )
        for index, passage in enumerate(passages, start=1)
    ]

    if not passages:
        return AskResponse(
            query=request.query,
            answer="No document passages were found. Upload a PDF before asking a question.",
            embedding_backend=embeddings.backend_name,
            sources=[],
        )

    try:
        answer = generation.answer(request.query, passages)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    return AskResponse(
        query=request.query,
        answer=answer,
        embedding_backend=embeddings.backend_name,
        sources=sources,
    )
