from __future__ import annotations

from pathlib import Path
from typing import Dict

from .schema import PromptTemplate

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(filename: str) -> PromptTemplate:
    file_path = PROMPT_DIR / filename
    text = file_path.read_text(encoding="utf-8")
    return PromptTemplate(name=filename, text=text)


def list_prompts() -> Dict[str, Path]:
    return {path.stem: path for path in PROMPT_DIR.glob("*.txt")}
