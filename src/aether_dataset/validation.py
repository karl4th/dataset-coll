from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

from .io import read_jsonl, sha256_file
from .schema import CACHE_SCHEMA


@dataclass
class ValidationSummary:
    shards: int = 0
    examples: int = 0
    audio_seconds: float = 0.0
    semantic_tokens: int = 0


def validate_cache(output: Path, *, verify_checksums: bool = True) -> ValidationSummary:
    manifest = read_jsonl(output / "shards.jsonl")
    if not manifest:
        raise ValueError(f"no shard manifest found in {output}")
    summary = ValidationSummary()
    seen_paths: set[str] = set()
    for record in manifest:
        if record.get("status") != "complete":
            continue
        relative = str(record["path"])
        if relative in seen_paths:
            raise ValueError(f"duplicate completed shard: {relative}")
        seen_paths.add(relative)
        path = output / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        if verify_checksums and sha256_file(path) != record["sha256"]:
            raise ValueError(f"checksum mismatch: {path}")
        table = pq.read_table(path)
        if table.schema != CACHE_SCHEMA:
            raise ValueError(f"schema mismatch: {path}")
        if table.num_rows != int(record["examples"]):
            raise ValueError(f"row count mismatch: {path}")
        lengths = table.column("semantic_length")
        actual_lengths = pc.list_value_length(table.column("semantic_codes"))
        if not pc.all(pc.equal(lengths, actual_lengths)).as_py():
            raise ValueError(f"semantic length mismatch: {path}")
        summary.shards += 1
        summary.examples += table.num_rows
        summary.audio_seconds += float(pc.sum(table.column("audio_seconds")).as_py())
        summary.semantic_tokens += int(pc.sum(lengths).as_py())
    return summary
