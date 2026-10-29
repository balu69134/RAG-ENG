"""Small HTTP client for the RAG-ENG FastAPI service."""

from typing import Any

import requests


class APIClientError(RuntimeError):
    """A user-readable API connection or response error."""


class RagApiClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = requests.request(method, f"{self.base_url}{path}", **kwargs)
        except requests.Timeout as exc:
            raise APIClientError("The request timed out. Please try again.") from exc
        except requests.RequestException as exc:
            raise APIClientError(
                f"Could not reach the API at {self.base_url}. Check that the backend is running."
            ) from exc

        if not response.ok:
            try:
                payload = response.json()
                detail = payload.get("detail", payload)
            except ValueError:
                detail = response.text or f"Request failed with status {response.status_code}."
            raise APIClientError(str(detail))

        if not response.content:
            return None
        try:
            return response.json()
        except ValueError as exc:
            raise APIClientError("The backend returned an invalid response.") from exc

    def health(self) -> dict:
        return self._request("GET", "/health", timeout=(3, 5))

    def upload_pdf(self, filename: str, content: bytes) -> dict:
        return self._request(
            "POST",
            "/documents/upload",
            files={"file": (filename, content, "application/pdf")},
            timeout=(5, 300),
        )

    def ask(self, query: str, top_k: int) -> dict:
        return self._request(
            "POST",
            "/ask",
            json={"query": query, "top_k": top_k},
            timeout=(5, 300),
        )

    def list_documents(self) -> list[dict]:
        return self._request("GET", "/documents", timeout=(3, 20))

    def delete_document(self, document_id: str) -> dict:
        return self._request(
            "DELETE",
            f"/documents/{document_id}",
            timeout=(3, 20),
        )
