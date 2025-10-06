from __future__ import annotations

from typing import Iterable, List


def chunk_text(text: str, max_chars: int, delimiter: str = "\n\n") -> List[str]:
    """Split text into chunks without breaking the given delimiter."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    pieces = text.split(delimiter)
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for piece in pieces:
        candidate_len = current_len + (len(delimiter) if current else 0) + len(piece)
        if candidate_len > max_chars and current:
            chunks.append(delimiter.join(current))
            current = [piece]
            current_len = len(piece)
        else:
            if current:
                current_len += len(delimiter) + len(piece)
            else:
                current_len = len(piece)
            current.append(piece)

    if current:
        chunks.append(delimiter.join(current))

    return chunks
