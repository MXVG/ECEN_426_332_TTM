from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_EXTENSIONS = (".c", ".js", ".py")


def list_files(root: Path, extensions: Sequence[str] | None) -> Iterable[Path]:
    normalized = None if extensions is None else {ext.lower() for ext in extensions}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if normalized and path.suffix.lower() not in normalized:
            continue
        yield path


def sha256_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_lines(path: Path, encoding: str) -> int:
    with path.open("r", encoding=encoding, errors="ignore") as handle:
        return sum(1 for _ in handle)


def infer_language(path: Path) -> str:
    mapping = {
        ".c": "c",
        ".h": "c",
        ".js": "javascript",
        ".ts": "typescript",
        ".py": "python",
    }
    return mapping.get(path.suffix.lower(), path.suffix.lstrip(".").lower() or "text")


def compose_identifier(label: str, relative: Path, language: str, digest: str) -> str:
    sanitized = str(relative.with_suffix("")).replace("\\", "/")
    sanitized = sanitized.replace("/", "_").replace(".", "-")
    return f"{label}/{sanitized}_{language}_{digest[:8]}"


def pick_part(parts: Sequence[str], index: int | None, fallback: str | None) -> str | None:
    if index is not None and 0 <= index < len(parts):
        return parts[index]
    return fallback


def build_manifest(
    root: Path,
    output: Path,
    label: str,
    split: str,
    variant: str | None,
    variant_depth: int | None,
    obfuscation: str | None,
    obfuscation_depth: int | None,
    copy_variant_to_obfuscation: bool,
    extensions: Sequence[str] | None,
    encoding: str,
) -> None:
    root = root.resolve()
    records = []
    for file_path in list_files(root, extensions):
        relative = file_path.relative_to(root)
        parts = relative.parts
        language = infer_language(file_path)
        digest = sha256_digest(file_path)
        record_variant = pick_part(parts, variant_depth, variant) or "plain"
        if copy_variant_to_obfuscation and record_variant != "plain":
            record_obfuscation = record_variant
        else:
            record_obfuscation = pick_part(parts, obfuscation_depth, obfuscation)
        record = {
            "id": compose_identifier(label, relative, language, digest),
            "language": language,
            "path": file_path.as_posix(),
            "variant": record_variant,
            "obfuscation": record_obfuscation,
            "source": (root / parts[0]).as_posix() if parts else root.as_posix(),
            "split": split,
            "sha256": f"sha256:{digest}",
            "loc": count_lines(file_path, encoding),
        }
        records.append(record)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a manifest JSONL file from a directory of source files.")
    parser.add_argument("--root", required=True, help="Dataset root directory (e.g. data/raw/obfuscated/javascript-data).")
    parser.add_argument("--label", required=True, help="Identifier prefix, e.g. NONOBFUSCATED or OBFUSCATED.")
    parser.add_argument("--output", required=True, help="Destination JSONL path.")
    parser.add_argument("--split", default="train", help="Dataset split label to store in each record.")
    parser.add_argument("--variant", default=None, help="Static variant label (default: inferred).")
    parser.add_argument("--variant-depth", type=int, default=None, help="Use this relative path component as the variant.")
    parser.add_argument("--obfuscation", default=None, help="Static obfuscation label (default: inferred).")
    parser.add_argument("--obfuscation-depth", type=int, default=None, help="Use this relative path component as the obfuscation label.")
    parser.add_argument("--copy-variant-to-obfuscation", action="store_true", help="Set obfuscation equal to the computed variant.")
    parser.add_argument(
        "--extensions",
        nargs="*",
        default=DEFAULT_EXTENSIONS,
        help="File extensions to include (default: .c .js .py).",
    )
    parser.add_argument("--encoding", default="utf-8", help="File encoding to use when counting lines.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_manifest(
        root=Path(args.root),
        output=Path(args.output),
        label=args.label,
        split=args.split,
        variant=args.variant,
        variant_depth=args.variant_depth,
        obfuscation=args.obfuscation,
        obfuscation_depth=args.obfuscation_depth,
        copy_variant_to_obfuscation=args.copy_variant_to_obfuscation,
        extensions=args.extensions if args.extensions else None,
        encoding=args.encoding,
    )


if __name__ == "__main__":
    main()
