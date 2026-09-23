from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .schema import CACHE_SCHEMA, rows_to_table


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {error}") from error
    return records


@dataclass(frozen=True)
class ShardRecord:
    split: str
    shard_index: int
    path: str
    status: str
    examples: int
    audio_seconds: float
    semantic_tokens: int
    sha256: str
    source_shards: list[str]
    last_source_shard: str
    last_source_row_index: int
    completed_at_utc: str


class ShardWriter:
    def __init__(self, output: Path, split: str, shard_index: int) -> None:
        self.output = output
        self.split = split
        self.shard_index = shard_index

    @property
    def relative_path(self) -> Path:
        return Path(self.split) / f"{self.split}-{self.shard_index:05d}.parquet"

    def write_complete(self, rows: list[dict[str, Any]]) -> ShardRecord:
        if not rows:
            raise ValueError("refusing to write an empty shard")
        final_path = self.output / self.relative_path
        final_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = final_path.with_suffix(".parquet.tmp")
        if final_path.exists():
            raise FileExistsError(final_path)
        table = rows_to_table(rows)
        pq.write_table(table, temporary_path, compression="zstd", use_dictionary=True)
        reopened = pq.read_table(temporary_path)
        if reopened.schema != CACHE_SCHEMA:
            raise ValueError(f"schema mismatch after writing {temporary_path}")
        if reopened.num_rows != len(rows):
            raise ValueError(f"row count mismatch after writing {temporary_path}")
        checksum = sha256_file(temporary_path)
        os.replace(temporary_path, final_path)
        record = ShardRecord(
            split=self.split,
            shard_index=self.shard_index,
            path=self.relative_path.as_posix(),
            status="complete",
            examples=len(rows),
            audio_seconds=sum(float(row["audio_seconds"]) for row in rows),
            semantic_tokens=sum(int(row["semantic_length"]) for row in rows),
            sha256=checksum,
            source_shards=sorted({str(row["source_shard"]) for row in rows}),
            last_source_shard=str(rows[-1]["source_shard"]),
            last_source_row_index=int(rows[-1]["source_row_index"]),
            completed_at_utc=utc_now(),
        )
        append_jsonl(self.output / "shards.jsonl", asdict(record))
        return record


def completed_shards(output: Path) -> dict[tuple[str, int], dict[str, Any]]:
    completed: dict[tuple[str, int], dict[str, Any]] = {}
    for record in read_jsonl(output / "shards.jsonl"):
        if record.get("status") == "complete":
            completed[(str(record["split"]), int(record["shard_index"]))] = record
    return completed
