from __future__ import annotations

import gzip
import json
import hashlib
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, Iterator, Union

JsonDict = Dict[str, Any]
PathLike = Union[str, Path]


def read_jsonl(path: PathLike) -> Iterator[JsonDict]:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: PathLike, rows: Iterable[JsonDict]) -> None:
    file_path = Path(path)
    with file_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_gzip_jsonl(path: PathLike) -> Iterator[JsonDict]:
    file_path = Path(path)
    with gzip.open(file_path, "rt", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_gzip_jsonl(path: PathLike, rows: Iterable[JsonDict]) -> None:
    file_path = Path(path)
    with gzip.open(file_path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def hash_text(value: str) -> str:
    digest = hashlib.sha256()
    digest.update(value.encode("utf-8"))
    return digest.hexdigest()


@contextmanager
def measure_time():
    start = perf_counter()
    try:
        yield
    finally:
        end = perf_counter()
        print(f"Elapsed {end - start:.3f}s")
