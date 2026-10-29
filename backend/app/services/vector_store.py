from functools import cached_property
from uuid import UUID, uuid5

from qdrant_client import QdrantClient, models

from backend.app.config import Settings
from backend.app.services.ingestion import TextChunk


class VectorStoreError(RuntimeError):
    """Raised when the vector database cannot complete an operation."""


class VectorStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @cached_property
    def client(self) -> QdrantClient:
        if self.settings.qdrant_local_path:
            return QdrantClient(path=self.settings.qdrant_local_path)

        options: dict[str, object] = {"url": self.settings.qdrant_url, "timeout": 30}
        if self.settings.qdrant_api_key:
            options["api_key"] = self.settings.qdrant_api_key
        return QdrantClient(**options)

    def check_connection(self) -> None:
        try:
            self.client.get_collections()
        except Exception as exc:
            raise VectorStoreError("Could not connect to Qdrant.") from exc

    def collection_exists(self) -> bool:
        try:
            return self.client.collection_exists(self.settings.qdrant_collection)
        except Exception as exc:
            raise VectorStoreError("Could not inspect the Qdrant collection.") from exc

    def ensure_collection(self, vector_size: int) -> None:
        try:
            if self.client.collection_exists(self.settings.qdrant_collection):
                info = self.client.get_collection(self.settings.qdrant_collection)
                vectors = info.config.params.vectors
                current_size = getattr(vectors, "size", None)
                if current_size is not None and current_size != vector_size:
                    raise VectorStoreError(
                        "The existing Qdrant collection uses a different embedding size. "
                        "Choose a new QDRANT_COLLECTION or clear the old collection."
                    )
                return

            self.client.create_collection(
                collection_name=self.settings.qdrant_collection,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
            self.client.create_payload_index(
                collection_name=self.settings.qdrant_collection,
                field_name="document_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=self.settings.qdrant_collection,
                field_name="embedding_backend",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError("Could not initialize the Qdrant collection.") from exc

    def upsert_chunks(
        self,
        document_id: str,
        filename: str,
        page_count: int,
        embedding_backend: str,
        chunks: list[TextChunk],
        vectors: list[list[float]],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Each text chunk must have exactly one embedding.")
        try:
            points = [
                models.PointStruct(
                    id=str(uuid5(UUID(document_id), str(chunk.chunk_index))),
                    vector=vector,
                    payload={
                        "document_id": document_id,
                        "filename": filename,
                        "page_count": page_count,
                        "embedding_backend": embedding_backend,
                        "page_number": chunk.page_number,
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                    },
                )
                for chunk, vector in zip(chunks, vectors)
            ]
            batch_size = 128
            for offset in range(0, len(points), batch_size):
                self.client.upsert(
                    collection_name=self.settings.qdrant_collection,
                    points=points[offset : offset + batch_size],
                    wait=True,
                )
        except Exception as exc:
            raise VectorStoreError("Could not save document chunks to Qdrant.") from exc

    def search(
        self, vector: list[float], limit: int, embedding_backend: str
    ) -> list[dict]:
        if not self.collection_exists():
            return []
        try:
            response = self.client.query_points(
                collection_name=self.settings.qdrant_collection,
                query=vector,
                limit=limit,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="embedding_backend",
                            match=models.MatchValue(value=embedding_backend),
                        )
                    ]
                ),
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            raise VectorStoreError("Could not search the Qdrant collection.") from exc

        results: list[dict] = []
        for point in response.points:
            payload = point.payload or {}
            results.append(
                {
                    "document_id": payload.get("document_id", ""),
                    "embedding_backend": payload.get("embedding_backend", ""),
                    "filename": payload.get("filename", "Unknown document"),
                    "page_number": int(payload.get("page_number", 0)),
                    "chunk_index": int(payload.get("chunk_index", 0)),
                    "text": payload.get("text", ""),
                    "score": float(point.score),
                }
            )
        return results

    def list_documents(self) -> list[dict]:
        if not self.collection_exists():
            return []

        summaries: dict[str, dict] = {}
        offset = None
        try:
            while True:
                records, offset = self.client.scroll(
                    collection_name=self.settings.qdrant_collection,
                    limit=256,
                    offset=offset,
                    with_payload=["document_id", "filename", "page_count"],
                    with_vectors=False,
                )
                for record in records:
                    payload = record.payload or {}
                    document_id = payload.get("document_id")
                    if not document_id:
                        continue
                    summary = summaries.setdefault(
                        document_id,
                        {
                            "document_id": document_id,
                            "filename": payload.get("filename", "Unknown document"),
                            "page_count": int(payload.get("page_count", 0)),
                            "chunk_count": 0,
                        },
                    )
                    summary["chunk_count"] += 1
                if offset is None:
                    break
        except Exception as exc:
            raise VectorStoreError("Could not list documents in Qdrant.") from exc
        return sorted(summaries.values(), key=lambda item: item["filename"].lower())

    def delete_document(self, document_id: str) -> bool:
        if not self.collection_exists():
            return False
        try:
            self.client.delete(
                collection_name=self.settings.qdrant_collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id",
                                match=models.MatchValue(value=document_id),
                            )
                        ]
                    )
                ),
                wait=True,
            )
            return True
        except Exception as exc:
            raise VectorStoreError("Could not delete the document from Qdrant.") from exc
