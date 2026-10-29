import hashlib
import logging
import math
import re

from backend.app.config import Settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Sentence Transformers encoder with a dependency-light lexical fallback."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None
        self._backend_name: str | None = None
        self._vector_size = 384

    def _resolve_backend(self) -> None:
        if self._backend_name is not None:
            return
        if self.settings.embedding_backend == "hashed-lexical":
            self._backend_name = "hashed-lexical"
            return
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(self.settings.embedding_model)
            size = model.get_sentence_embedding_dimension()
            self._model = model
            self._vector_size = int(size or 384)
            self._backend_name = "sentence-transformers"
        except Exception as exc:
            if self.settings.embedding_backend == "sentence-transformers":
                raise RuntimeError(
                    "Sentence Transformers could not load. Check its installation and "
                    "the machine's native-library policy."
                ) from exc
            self._backend_name = "hashed-lexical"
            logger.warning(
                "Sentence Transformers unavailable (%s); using hashed lexical embeddings.",
                type(exc).__name__,
            )

    @property
    def backend_name(self) -> str:
        self._resolve_backend()
        return self._backend_name or "hashed-lexical"

    @property
    def vector_size(self) -> int:
        self._resolve_backend()
        return self._vector_size

    def _encode_lexical(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            tokens = re.findall(r"[\w]+", text.casefold())
            features: list[tuple[str, float]] = [(token, 1.0) for token in tokens]
            features.extend(
                (f"{left}_{right}", 1.5)
                for left, right in zip(tokens, tokens[1:])
            )
            for token in tokens:
                if len(token) >= 3:
                    padded = f"^{token}$"
                    features.extend(
                        (f"char{width}:{padded[index:index + width]}", 0.2)
                        for width in (3, 4)
                        for index in range(max(0, len(padded) - width + 1))
                    )

            vector = [0.0] * self._vector_size
            for feature, weight in features:
                digest = hashlib.blake2b(
                    feature.encode("utf-8"), digest_size=8, person=b"rag-eng"
                ).digest()
                index = int.from_bytes(digest[:4], "little") % self._vector_size
                sign = 1.0 if digest[4] & 1 else -1.0
                vector[index] += sign * weight

            norm = math.sqrt(sum(value * value for value in vector))
            if norm:
                vector = [value / norm for value in vector]
            vectors.append(vector)
        return vectors

    def encode(self, texts: list[str]) -> list[list[float]]:
        self._resolve_backend()
        if self.backend_name == "hashed-lexical":
            return self._encode_lexical(texts)

        try:
            import numpy as np

            vectors = self._model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=self.settings.embedding_batch_size,
            )
            return np.asarray(vectors, dtype=np.float32).tolist()
        except Exception as exc:
            if self.settings.embedding_backend == "sentence-transformers":
                raise RuntimeError("Sentence Transformers could not encode the text.") from exc
            logger.warning(
                "Sentence Transformers failed during encoding (%s); using hashed lexical embeddings.",
                type(exc).__name__,
            )
            self._model = None
            self._backend_name = "hashed-lexical"
            return self._encode_lexical(texts)
