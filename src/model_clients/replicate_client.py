from __future__ import annotations

from typing import Any, Dict

from .base import ModelClient


class ReplicateClient(ModelClient):
    """Client wrapper for Replicate hosted models."""

    def __init__(self, api_token: str, model: str, **settings: Any) -> None:
        super().__init__(name=model, api_token=api_token, model=model, **settings)
        self.api_token = api_token
        self.model = model

    def generate(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError("Integrate Replicate client in this method.")
