from __future__ import annotations

import os
from typing import Any, Dict, List, MutableMapping

from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError, InferenceTimeoutError

from .base import ModelClient


_DEFAULT_CHAT_ENDPOINT = "https://router.huggingface.co/v1/chat/completions"


class HuggingFaceClient(ModelClient):
    """Client wrapper for the Hugging Face OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        api_token: str | None,
        model: str,
        endpoint: str | None = None,
        timeout: float = 30.0,
        **settings: Any,
    ) -> None:
        """Capture Hugging Face endpoint configuration details."""
        resolved_token = api_token or os.getenv("HF_API_TOKEN") or os.getenv("HF_TOKEN")
        if not resolved_token:
            raise ValueError("A Hugging Face API token must be provided via argument or HF_API_TOKEN env var.")

        resolved_endpoint = endpoint or _DEFAULT_CHAT_ENDPOINT
        default_request_fields = dict(settings.pop("request_params", {}) or {})

        client_kwargs: Dict[str, Any] = {
            "api_key": resolved_token,
            "timeout": timeout,
        }
        provider = settings.pop("provider", None)
        if provider:
            client_kwargs["provider"] = provider
        if endpoint:
            client_kwargs["base_url"] = endpoint

        super().__init__(
            name=model,
            api_key=resolved_token,
            model=model,
            endpoint=resolved_endpoint,
            timeout=timeout,
            **settings,
        )
        self.api_token = resolved_token
        self.model = model
        self.endpoint = resolved_endpoint
        self.timeout = timeout
        self._default_request_fields: Dict[str, Any] = default_request_fields
        self._client = InferenceClient(**client_kwargs)

    def generate(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """Invoke the Hugging Face chat completion endpoint via huggingface_hub."""
        payload = self._build_payload(prompt, **kwargs)
        try:
            response = self._client.chat.completions.create(**payload)
        except InferenceTimeoutError as exc:
            raise RuntimeError("Hugging Face request timed out") from exc
        except HfHubHTTPError as exc:
            detail = exc.response.text if getattr(exc, "response", None) is not None else str(exc)
            raise RuntimeError(f"Hugging Face request failed: {detail}") from exc

        if isinstance(response, MutableMapping):
            return dict(response)
        if isinstance(response, dict):
            return response
        raise RuntimeError("Unexpected Hugging Face response type (streaming is not supported)")

    def format_response(self, response: Dict[str, Any]) -> str:
        """Extract the assistant text from a chat completion response."""
        choices = response.get("choices")
        if not choices:
            return ""

        first = choices[0]
        message = first.get("message")
        return self._extract_message_content(message)

    def _build_payload(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        config = kwargs.pop("config", None)
        explicit_messages = kwargs.pop("messages", None)
        if explicit_messages is None:
            messages = self._prompt_to_messages(prompt)
        else:
            messages = explicit_messages

        payload: Dict[str, Any] = {
            "model": kwargs.pop("model", self.model),
            "messages": messages,
        }

        if config is not None:
            self._set_if_not_none(payload, "temperature", getattr(config, "temperature", None))
            self._set_if_not_none(payload, "max_tokens", getattr(config, "max_output_tokens", None))
            self._set_if_not_none(payload, "top_p", getattr(config, "top_p", None))

        payload.update({k: v for k, v in self._default_request_fields.items() if v is not None})
        payload.update({k: v for k, v in kwargs.items() if v is not None})

        return payload

    @staticmethod
    def _set_if_not_none(target: Dict[str, Any], key: str, value: Any) -> None:
        if value is not None:
            target.setdefault(key, value)

    @staticmethod
    def _prompt_to_messages(prompt: str) -> List[Dict[str, str]]:
        return [{"role": "user", "content": prompt}]

    @staticmethod
    def _extract_message_content(message: Any) -> str:
        """Handle OpenAI-style message objects with list or string content."""
        if isinstance(message, MutableMapping):
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                text_parts = [segment.get("text") if isinstance(segment, MutableMapping) else segment for segment in content]
                joined = "".join(part for part in text_parts if isinstance(part, str))
                if joined:
                    return joined
        if isinstance(message, str):
            return message
        if isinstance(message, list):
            merged = "".join(part for part in message if isinstance(part, str))
            if merged:
                return merged
        return ""
