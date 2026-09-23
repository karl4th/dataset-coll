import pyarrow as pa

from aether_dataset.schema import CACHE_SCHEMA, rows_to_table, validate_row


def example_row() -> dict[str, object]:
    return {
        "sample_id": "sample",
        "source_id": "source",
        "source_config": "medium",
        "source_split": "train",
        "source_shard": "medium/train-00000.parquet",
        "source_row_index": 0,
        "speaker_id": "speaker",
        "chapter_id": "chapter",
        "source_audio_name": "audio.flac",
        "source_sample_rate": 16000,
        "source_num_samples": 16000,
        "audio_seconds": 1.0,
        "original_text": "Hello",
        "normalized_text": "hello",
        "semantic_codes": [0, 1024, 2047],
        "semantic_length": 3,
    }


def test_arrow_schema_uses_compact_semantic_types() -> None:
    assert CACHE_SCHEMA.field("semantic_codes").type == pa.list_(pa.uint16())
    assert CACHE_SCHEMA.field("semantic_length").type == pa.int32()


def test_row_roundtrip() -> None:
    row = example_row()
    validate_row(row, codebook_size=2048)
    table = rows_to_table([row])
    assert table.schema == CACHE_SCHEMA
    assert table.to_pylist()[0]["semantic_codes"] == [0, 1024, 2047]
