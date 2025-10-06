from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ManifestRecord:
    """Representation of a single row in the manifest JSONL files."""
    identifier: str
    path: Path
    split: str
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class PromptTemplate:
    """Prompt template."""
    name: str
    text: str


@dataclass
class BatchRunConfig:
    """Settings that control how a batch of prompts is executed."""
    model_name: str
    temperature: float
    max_input_tokens: int
    max_output_tokens: int
    top_p: Optional[float] = None
    batch_size: int = 1
