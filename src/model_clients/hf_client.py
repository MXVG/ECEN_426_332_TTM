from __future__ import annotations

from typing import Any, Dict

from .base import ModelClient


class HuggingFaceClient(ModelClient):
    """Client wrapper for Hugging Face text generation endpoints."""

    def __init__(self, api_token: str, model: str, endpoint: str, **settings: Any) -> None:
        super().__init__(name=model, api_token=api_token, model=model, endpoint=endpoint, **settings)
        self.api_token = api_token
        self.model = model
        self.endpoint = endpoint

    def generate(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError("Integrate Hugging Face client in this method.")
