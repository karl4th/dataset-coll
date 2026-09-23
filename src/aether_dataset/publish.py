from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

from .io import append_jsonl, read_jsonl, utc_now
from .validation import validate_cache

DEFAULT_REPO_ID = "manifestro/stage1_aether"


@dataclass(frozen=True)
class PublishSummary:
    repo_id: str
    uploaded_shards: int
    skipped_shards: int
    examples: int
    audio_seconds: float
    semantic_tokens: int


def publish_private_cache(
    output: Path,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    token: str | None = None,
) -> PublishSummary:
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for private Hugging Face publication")
    validation = validate_cache(output, verify_checksums=True)
    _require_complete_splits(output)

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True, token=token)
    info = api.repo_info(repo_id, repo_type="dataset", token=token)
    if not bool(getattr(info, "private", False)):
        raise RuntimeError(f"refusing to publish because dataset repo is not private: {repo_id}")

    uploaded = 0
    skipped = 0
    for record in _complete_shard_records(output):
        receipt_path = _receipt_path(str(record["path"]))
        if _remote_receipt_matches(api, repo_id, receipt_path, record, token):
            skipped += 1
            continue
        local_shard = output / str(record["path"])
        receipt = {
            "schema_version": "1.0",
            "repo_id": repo_id,
            "path": record["path"],
            "sha256": record["sha256"],
            "examples": record["examples"],
            "audio_seconds": record["audio_seconds"],
            "semantic_tokens": record["semantic_tokens"],
            "published_at_utc": utc_now(),
        }
        api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message=f"Upload verified cache shard {record['path']}",
            operations=[
                CommitOperationAdd(path_in_repo=str(record["path"]), path_or_fileobj=local_shard),
                CommitOperationAdd(
                    path_in_repo=receipt_path,
                    path_or_fileobj=json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8"),
                ),
            ],
        )
        append_jsonl(output / "published.jsonl", receipt)
        uploaded += 1

    metadata_files = {
        "dataset_info.json": output / "dataset_info.json",
        "shards.jsonl": output / "shards.jsonl",
        "splits.jsonl": output / "splits.jsonl",
        "processing_stats.json": output / "processing_stats.json",
        "failures.jsonl": output / "failures.jsonl",
        "DATASET_SCHEMA.md": Path(__file__).resolve().parents[2] / "DATASET_SCHEMA.md",
        "README.md": Path(__file__).resolve().parents[2] / "HF_DATASET_CARD.md",
    }
    operations = [
        CommitOperationAdd(path_in_repo=remote, path_or_fileobj=local)
        for remote, local in metadata_files.items()
        if local.is_file()
    ]
    if operations:
        api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message="Publish validated cache metadata",
            operations=operations,
        )
    _verify_all_remote_receipts(api, repo_id, output, token)
    return PublishSummary(
        repo_id=repo_id,
        uploaded_shards=uploaded,
        skipped_shards=skipped,
        examples=validation.examples,
        audio_seconds=validation.audio_seconds,
        semantic_tokens=validation.semantic_tokens,
    )


def _require_complete_splits(output: Path) -> None:
    info_path = output / "dataset_info.json"
    if not info_path.is_file():
        raise RuntimeError("dataset_info.json is missing")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    expected = set(info.get("expected_audio_hours_by_split", {}))
    completed = {
        str(item["split"])
        for item in read_jsonl(output / "splits.jsonl")
        if item.get("status") == "complete"
    }
    missing = sorted(expected - completed)
    if missing:
        raise RuntimeError(f"refusing to publish incomplete splits: {', '.join(missing)}")


def _complete_shard_records(output: Path) -> list[dict[str, Any]]:
    return [
        record
        for record in read_jsonl(output / "shards.jsonl")
        if record.get("status") == "complete"
    ]


def _receipt_path(shard_path: str) -> str:
    return f"_receipts/{shard_path.replace('/', '--')}.json"


def _remote_receipt_matches(
    api: HfApi,
    repo_id: str,
    receipt_path: str,
    record: dict[str, Any],
    token: str,
) -> bool:
    if not api.file_exists(repo_id, receipt_path, repo_type="dataset", token=token):
        return False
    local = hf_hub_download(repo_id, receipt_path, repo_type="dataset", token=token)
    receipt = json.loads(Path(local).read_text(encoding="utf-8"))
    if receipt.get("sha256") != record["sha256"]:
        raise RuntimeError(f"remote receipt checksum conflict: {receipt_path}")
    return True


def _verify_all_remote_receipts(api: HfApi, repo_id: str, output: Path, token: str) -> None:
    missing = []
    for record in _complete_shard_records(output):
        receipt = _receipt_path(str(record["path"]))
        if not _remote_receipt_matches(api, repo_id, receipt, record, token):
            missing.append(receipt)
    if missing:
        raise RuntimeError(f"remote publication is incomplete: {', '.join(missing)}")
