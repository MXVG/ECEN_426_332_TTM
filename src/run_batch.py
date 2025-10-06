from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Protocol

from .schema import BatchRunConfig, ManifestRecord
from .utils_io import read_jsonl
from .model_clients.openai_client import OpenAIClient

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


class SupportsGenerate(Protocol):
    def generate(self, prompt: str, **kwargs: object) -> dict:
        ...


def load_manifest(path: str) -> Iterable[ManifestRecord]:
    file_path = Path(path)
    for row in read_jsonl(file_path):
        yield ManifestRecord(
            identifier=row.get("id", ""),
            path=Path(row["path"]),
            split=row.get("split", "train"),
            tags=row.get("tags", []),
            metadata=row.get("metadata", {}),
        )


def run(records: Iterable[ManifestRecord], client: SupportsGenerate, config: BatchRunConfig) -> None:
    for record in records:
        source_text = record.path.read_text(encoding="utf-8")
        response = client.generate(source_text, config=config)
        formatter = getattr(client, "format_response", None)
        formatted = formatter(response) if callable(formatter) else None
        _handle_response(record, response, formatted, config)


def _handle_response(record: ManifestRecord, response: dict, formatted: str | None, config: BatchRunConfig) -> None:
    # Persist response payloads for later inspection.
    _store_response(record, response, formatted, config)
    print(record.identifier)
    print(f"Processed {record.identifier} with response keys: {list(response.keys())}")



def _store_response(record: ManifestRecord, response: dict, formatted: str | None, config: BatchRunConfig) -> None:
    result_dir = RESULTS_DIR / config.model_name
    result_dir.mkdir(parents=True, exist_ok=True)

    safe_identifier = record.identifier.replace('/', '__')
    output_path = result_dir / f"{safe_identifier}.json"

    payload = {
        'identifier': record.identifier,
        'source_path': record.path.as_posix(),
        'model': config.model_name,
        'response': response,
        'formatted_text': formatted,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    rel_path = output_path.relative_to(Path.cwd()) if output_path.is_absolute() else output_path
    print(f"Saved response to {rel_path}")




def main(manifest_path: str, client: SupportsGenerate, config: BatchRunConfig) -> None:
    records = load_manifest(manifest_path)
    run(records, client, config)


if __name__ == "__main__":
    import argparse
    from itertools import islice

    parser = argparse.ArgumentParser(description="Run manifest entries.")
    parser.add_argument("manifest", help="Path to the manifest JSONL file.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N records.")
    parser.add_argument("--api-key", default=None, help="API key (defaults to OPENAI_API_KEY env var).")
    parser.add_argument("--model", default="gpt-4-turbo", help="model name to invoke.")
    parser.add_argument("--endpoint", default=None, help="Override the API endpoint URL.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for the request.")
    parser.add_argument("--max-input-tokens", type=int, default=4096, help="Hint for maximum prompt tokens.")
    parser.add_argument("--max-output-tokens", type=int, default=512, help="Maximum completion tokens to request.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling probability mass.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size metadata for the run.")
    args = parser.parse_args()

    client_kwargs = {"api_key": args.api_key, "model": args.model}
    if args.endpoint:
        client_kwargs["endpoint"] = args.endpoint
    client = OpenAIClient(**{k: v for k, v in client_kwargs.items() if v is not None})

    config = BatchRunConfig(
        model_name=args.model,
        temperature=args.temperature,
        max_input_tokens=args.max_input_tokens,
        max_output_tokens=args.max_output_tokens,
        top_p=args.top_p,
        batch_size=args.batch_size,
    )

    records: Iterable[ManifestRecord] = load_manifest(args.manifest)
    if args.limit is not None:
        records = list(islice(records, args.limit))

    run(records, client, config)
