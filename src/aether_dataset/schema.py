from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pyarrow as pa

CACHE_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("source_config", pa.string(), nullable=False),
        pa.field("source_split", pa.string(), nullable=False),
        pa.field("source_shard", pa.string(), nullable=False),
        pa.field("source_row_index", pa.int64(), nullable=False),
        pa.field("speaker_id", pa.string(), nullable=False),
        pa.field("chapter_id", pa.string(), nullable=False),
        pa.field("source_audio_name", pa.string(), nullable=False),
        pa.field("source_sample_rate", pa.int32(), nullable=False),
        pa.field("source_num_samples", pa.int64(), nullable=False),
        pa.field("audio_seconds", pa.float32(), nullable=False),
        pa.field("original_text", pa.string(), nullable=False),
        pa.field("normalized_text", pa.string(), nullable=False),
        pa.field("semantic_codes", pa.list_(pa.uint16()), nullable=False),
        pa.field("semantic_length", pa.int32(), nullable=False),
    ]
)


class RowValidationError(ValueError):
    pass


def validate_row(row: Mapping[str, Any], *, codebook_size: int) -> None:
    missing = [field.name for field in CACHE_SCHEMA if field.name not in row]
    if missing:
        raise RowValidationError(f"missing fields: {', '.join(missing)}")
    if not row["sample_id"]:
        raise RowValidationError("sample_id is empty")
    if int(row["source_sample_rate"]) <= 0 or int(row["source_num_samples"]) <= 0:
        raise RowValidationError("invalid source audio dimensions")
    if float(row["audio_seconds"]) <= 0:
        raise RowValidationError("audio_seconds must be positive")
    if not row["original_text"] or not row["normalized_text"]:
        raise RowValidationError("text must not be empty")
    codes = row["semantic_codes"]
    if int(row["semantic_length"]) != len(codes) or not codes:
        raise RowValidationError("semantic_length does not match semantic_codes")
    if min(codes) < 0 or max(codes) >= codebook_size:
        raise RowValidationError("semantic code is outside the codebook")


def rows_to_table(rows: list[Mapping[str, Any]]) -> pa.Table:
    return pa.Table.from_pylist(list(rows), schema=CACHE_SCHEMA)
