from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class ModelClient(ABC):
    """Common interface for model backends."""

    def __init__(self, name: str, **client_settings: Any) -> None:
        self.name = name
        self._settings = client_settings

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """Invoke the model and return the raw provider response."""

    def format_response(self, response: Dict[str, Any]) -> str:
        """Extract the assistant text from a provider response."""
        raise NotImplementedError("format_response must be implemented")
