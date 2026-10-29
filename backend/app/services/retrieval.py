from backend.app.config import Settings
from backend.app.services.embeddings import EmbeddingService
from backend.app.services.vector_store import VectorStore


class RetrievalService:
    def __init__(
        self,
        settings: Settings,
        embeddings: EmbeddingService,
        vector_store: VectorStore,
    ) -> None:
        self.settings = settings
        self.embeddings = embeddings
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        vector = self.embeddings.encode([query])[0]
        return self.vector_store.search(
            vector=vector,
            limit=top_k or self.settings.default_top_k,
            embedding_backend=self.embeddings.backend_name,
        )
