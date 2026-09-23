from dataclasses import asdict

from aether_dataset.io import ShardWriter, read_jsonl, sha256_file
from aether_dataset.validation import validate_cache

from .test_schema import example_row


def test_atomic_shard_and_manifest(tmp_path) -> None:
    record = ShardWriter(tmp_path, "train", 0).write_complete([example_row()])
    output = tmp_path / record.path
    assert output.is_file()
    assert not output.with_suffix(".parquet.tmp").exists()
    assert sha256_file(output) == record.sha256
    assert read_jsonl(tmp_path / "shards.jsonl") == [asdict(record)]
    summary = validate_cache(tmp_path)
    assert summary.shards == 1
    assert summary.examples == 1
