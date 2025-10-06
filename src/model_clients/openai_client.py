
from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, MutableMapping, Optional
from urllib import error, request

from .base import ModelClient


_DEFAULT_ENDPOINT = "https://api.openai.com/v1/chat/completions"


class OpenAIClient(ModelClient):
    """Client wrapper for the OpenAI Chat Completions API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4-turbo",
        endpoint: str = _DEFAULT_ENDPOINT,
        timeout: float = 30.0,
        **default_params: Any,
    ) -> None:
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError("An OpenAI API key must be provided via argument or OPENAI_API_KEY env var.")

        self.api_key = resolved_key
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout
        self._default_params: Dict[str, Any] = dict(default_params)

        super().__init__(
            name=model,
            api_key=resolved_key,
            model=model,
            endpoint=endpoint,
            timeout=timeout,
            **default_params,
        )

    def generate(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        payload = self._build_payload(prompt, **kwargs)
        encoded_payload = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        req = request.Request(self.endpoint, data=encoded_payload, headers=headers)

        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp else exc.reason
            raise RuntimeError(f"OpenAI request failed with status {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"OpenAI request failed: {exc.reason}") from exc

        return json.loads(body)

    def format_response(self, response: Dict[str, Any]) -> str:
        choices = response.get("choices")
        if not choices:
            return ""

        first = choices[0]
        message = first.get("message")
        if isinstance(message, MutableMapping):
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, Iterable):
                # Handle tool message formats that return segments.
                return "".join(segment for segment in content if isinstance(segment, str))

        text = first.get("text")
        return text if isinstance(text, str) else ""

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

        params: Dict[str, Any] = {}
        params.update(self._default_params)
        if config is not None:
            params.setdefault("temperature", getattr(config, "temperature", None))
            params.setdefault("max_tokens", getattr(config, "max_output_tokens", None))
            params.setdefault("top_p", getattr(config, "top_p", None))

        params = {k: v for k, v in params.items() if v is not None}
        params.update({k: v for k, v in kwargs.items() if v is not None})

        payload.update(params)
        return payload

    @staticmethod
    def _prompt_to_messages(prompt: str) -> List[Dict[str, str]]:
        return [{"role": "user", "content": prompt}]
