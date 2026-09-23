from __future__ import annotations

import hashlib
import json
import queue
import signal
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .benchmark import _is_cuda_oom, require_matching_benchmark
from .config import AppConfig
from .io import ShardWriter, append_jsonl, completed_shards, read_jsonl, utc_now
from .mimi import MimiEncoder, PreparedAudio
from .normalization import normalize
from .progress import ProgressState, RunReporter
from .schema import validate_row
from .source import SourceExample, iter_source_examples
from .validation import validate_cache


class StopRequested(Exception):
    pass


@dataclass(frozen=True)
class PreparedBatch:
    items: list[tuple[SourceExample, PreparedAudio, str]]
    last_seen_cursor: tuple[str, int] | None
    failures: int = 0


@dataclass(frozen=True)
class ProducerError:
    error: Exception


def run(config: AppConfig, *, resume: bool) -> None:
    config.require_ready()
    output = config.output_path
    output.mkdir(parents=True, exist_ok=True)
    existing = completed_shards(output)
    if existing and not resume:
        raise RuntimeError(f"{output} already has completed shards; use --resume")
    _write_dataset_info(config)
    benchmark_result = require_matching_benchmark(config)
    mimi_cfg = config.raw["mimi"]
    runtime = config.raw["runtime"]
    encoder = MimiEncoder(
        hf_repo=mimi_cfg["hf_repo"],
        revision=mimi_cfg["revision"],
        device=runtime["device"],
        semantic_codebook_index=int(mimi_cfg["semantic_codebook_index"]),
    )
    if encoder.sample_rate != int(mimi_cfg["sample_rate"]):
        raise RuntimeError("configured Mimi sample rate differs from loaded model")
    stop = {"requested": False}

    def request_stop(signum: int, _frame: object) -> None:
        stop["requested"] = True

    old_int = signal.signal(signal.SIGINT, request_stop)
    old_term = signal.signal(signal.SIGTERM, request_stop)
    try:
        for target_split, split_config in config.splits.items():
            completed_split_names = {
                str(item["split"])
                for item in read_jsonl(output / "splits.jsonl")
                if item.get("status") == "complete"
            }
            if target_split in completed_split_names:
                continue
            split_records = [
                record for (name, _), record in existing.items() if name == target_split
            ]
            next_index = max((int(item["shard_index"]) for item in split_records), default=-1) + 1
            cursor = None
            if split_records:
                last = max(split_records, key=lambda item: int(item["shard_index"]))
                cursor = (str(last["last_source_shard"]), int(last["last_source_row_index"]))
            completed_audio_seconds = sum(float(item["audio_seconds"]) for item in split_records)
            completed_examples = sum(int(item["examples"]) for item in split_records)
            completed_tokens = sum(int(item["semantic_tokens"]) for item in split_records)
            reporter = RunReporter(
                output,
                total_audio_seconds=split_config.expected_audio_hours * 3600,
                initial_audio_seconds=completed_audio_seconds,
            )
            state = ProgressState(
                split=target_split,
                shard_index=next_index,
                examples=completed_examples,
                audio_seconds=completed_audio_seconds,
                semantic_tokens=completed_tokens,
            )
            rows: list[dict[str, Any]] = []
            batch_budget = float(benchmark_result["recommended_batch_audio_seconds"])
            examples = iter_source_examples(
                config.raw["dataset"]["id"],
                config.raw["dataset"]["revision"],
                split_config.config,
                split_config.split,
                start_after=cursor,
            )
            last_seen_cursor = cursor
            prefetch: queue.Queue[PreparedBatch | ProducerError | None] = queue.Queue(
                maxsize=int(runtime["prefetch_batches"])
            )
            producer = threading.Thread(
                target=_produce_batches,
                args=(
                    examples,
                    encoder,
                    config,
                    output,
                    batch_budget,
                    stop,
                    prefetch,
                    cursor,
                ),
                name=f"prefetch-{target_split}",
                daemon=False,
            )
            producer.start()
            try:
                while True:
                    item = prefetch.get()
                    if item is None:
                        break
                    if isinstance(item, ProducerError):
                        raise item.error
                    last_seen_cursor = item.last_seen_cursor
                    state.errors += item.failures
                    if not item.items:
                        continue
                    batch_rows, batch_budget = _encode_with_oom_backoff(
                        encoder, item.items, config, batch_budget
                    )
                    rows.extend(batch_rows)
                    _update_state(state, batch_rows)
                    reporter.report(
                        state,
                        interval=float(runtime["metrics_interval_seconds"]),
                        prefetch_queue_depth=prefetch.qsize(),
                    )
                    accumulated_seconds = sum(float(row["audio_seconds"]) for row in rows)
                    if accumulated_seconds >= config.target_shard_seconds:
                        ShardWriter(output, target_split, state.shard_index).write_complete(rows)
                        state.shard_index += 1
                        rows = []
            except BaseException:
                stop["requested"] = True
                while producer.is_alive():
                    try:
                        prefetch.get(timeout=0.5)
                    except queue.Empty:
                        continue
                producer.join()
                raise
            producer.join()
            if rows:
                ShardWriter(output, target_split, state.shard_index).write_complete(rows)
            reporter.report(state, force=True)
            if stop["requested"]:
                raise StopRequested("stop requested; completed data is safe for --resume")
            append_jsonl(
                output / "splits.jsonl",
                {
                    "split": target_split,
                    "status": "complete",
                    "last_source_shard": last_seen_cursor[0] if last_seen_cursor else None,
                    "last_source_row_index": last_seen_cursor[1] if last_seen_cursor else None,
                    "completed_at_utc": utc_now(),
                },
            )
        validate_cache(output)
        _write_processing_stats(output)
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)


def _produce_batches(
    examples: Any,
    encoder: MimiEncoder,
    config: AppConfig,
    output: Path,
    batch_budget: float,
    stop: dict[str, bool],
    destination: queue.Queue[PreparedBatch | ProducerError | None],
    initial_cursor: tuple[str, int] | None,
) -> None:
    pending: list[tuple[SourceExample, PreparedAudio, str]] = []
    maximum_seconds = 0.0
    failures = 0
    last_seen_cursor = initial_cursor
    pending_last_cursor = initial_cursor

    def put(value: PreparedBatch | ProducerError, *, honor_stop: bool = True) -> bool:
        while not stop["requested"]:
            try:
                destination.put(value, timeout=0.5)
                return True
            except queue.Full:
                continue
        if not honor_stop:
            while True:
                try:
                    destination.put(value, timeout=0.5)
                    return True
                except queue.Full:
                    continue
        return False

    try:
        for example in examples:
            if stop["requested"]:
                break
            current_cursor = (example.source_shard, example.source_row_index)
            try:
                normalized = normalize(example.original_text, config.raw["text"]["normalizer"])
                prepared = encoder.prepare(example.audio_array, example.sample_rate)
                if not normalized:
                    raise ValueError("normalized text is empty")
                candidate_maximum = max(maximum_seconds, prepared.audio_seconds)
                candidate_padded_seconds = candidate_maximum * (len(pending) + 1)
                if pending and candidate_padded_seconds > batch_budget:
                    if not put(PreparedBatch(pending, pending_last_cursor, failures)):
                        return
                    pending = []
                    maximum_seconds = 0.0
                    failures = 0
                pending.append((example, prepared, normalized))
                maximum_seconds = max(maximum_seconds, prepared.audio_seconds)
                pending_last_cursor = current_cursor
                last_seen_cursor = current_cursor
            except Exception as error:
                _record_failure(output, example, error)
                failures += 1
                last_seen_cursor = current_cursor
                continue
            if maximum_seconds * len(pending) < batch_budget:
                continue
            if not put(PreparedBatch(pending, last_seen_cursor, failures)):
                return
            pending = []
            maximum_seconds = 0.0
            failures = 0
        if (pending or failures) and not put(PreparedBatch(pending, last_seen_cursor, failures)):
            return
    except Exception as error:
        put(ProducerError(error))
    finally:
        while True:
            try:
                destination.put(None, timeout=0.5)
                break
            except queue.Full:
                continue


def _encode_with_oom_backoff(
    encoder: MimiEncoder,
    pending: list[tuple[SourceExample, PreparedAudio, str]],
    config: AppConfig,
    budget: float,
) -> tuple[list[dict[str, Any]], float]:
    try:
        codes = encoder.encode_batch([item[1] for item in pending])
    except Exception as error:
        torch = encoder.torch
        if not _is_cuda_oom(encoder, error) or len(pending) == 1:
            raise
        torch.cuda.empty_cache()
        midpoint = len(pending) // 2
        left, left_budget = _encode_with_oom_backoff(
            encoder, pending[:midpoint], config, budget / 2
        )
        right, right_budget = _encode_with_oom_backoff(
            encoder, pending[midpoint:], config, budget / 2
        )
        return left + right, min(left_budget, right_budget)
    rows = [
        _make_row(example, prepared, normalized, semantic, config)
        for (example, prepared, normalized), semantic in zip(pending, codes, strict=True)
    ]
    return rows, budget


def _make_row(
    example: SourceExample,
    prepared: PreparedAudio,
    normalized: str,
    semantic: list[int],
    config: AppConfig,
) -> dict[str, Any]:
    identity = (
        f"{example.source_config}:{example.source_split}:"
        f"{example.source_shard}:{example.source_row_index}"
    )
    row = {
        "sample_id": hashlib.sha256(identity.encode()).hexdigest()[:32],
        "source_id": example.source_id,
        "source_config": example.source_config,
        "source_split": example.source_split,
        "source_shard": example.source_shard,
        "source_row_index": example.source_row_index,
        "speaker_id": example.speaker_id,
        "chapter_id": example.chapter_id,
        "source_audio_name": example.source_audio_name,
        "source_sample_rate": prepared.source_sample_rate,
        "source_num_samples": prepared.source_num_samples,
        "audio_seconds": prepared.audio_seconds,
        "original_text": example.original_text,
        "normalized_text": normalized,
        "semantic_codes": semantic,
        "semantic_length": len(semantic),
    }
    validate_row(row, codebook_size=int(config.raw["mimi"]["codebook_size"]))
    return row


def _record_failure(output: Path, example: SourceExample, error: Exception) -> None:
    append_jsonl(
        output / "failures.jsonl",
        {
            "timestamp_utc": utc_now(),
            "source_id": example.source_id,
            "source_config": example.source_config,
            "source_split": example.source_split,
            "source_shard": example.source_shard,
            "source_row_index": example.source_row_index,
            "source_audio_name": example.source_audio_name,
            "error_type": type(error).__name__,
            "error_message": str(error),
        },
    )


def _update_state(state: ProgressState, rows: list[dict[str, Any]]) -> None:
    state.examples += len(rows)
    state.audio_seconds += sum(float(row["audio_seconds"]) for row in rows)
    state.semantic_tokens += sum(int(row["semantic_length"]) for row in rows)


def _write_dataset_info(config: AppConfig) -> None:
    path = config.output_path / "dataset_info.json"
    benchmark_result = require_matching_benchmark(config)
    payload = {
        "schema_version": "1.0.0",
        "dataset_id": config.raw["dataset"]["id"],
        "dataset_configs": [item.config for item in config.splits.values()],
        "dataset_revision": config.raw["dataset"]["revision"],
        "mimi_model_id": config.raw["mimi"]["hf_repo"],
        "mimi_model_revision": config.raw["mimi"]["revision"],
        "mimi_model_sha256": benchmark_result["mimi_model_sha256"],
        "mimi_implementation": benchmark_result["mimi_implementation"],
        "mimi_sample_rate": config.raw["mimi"]["sample_rate"],
        "mimi_frame_rate": config.raw["mimi"]["frame_rate"],
        "semantic_codebook_index": config.raw["mimi"]["semantic_codebook_index"],
        "semantic_codebook_size": config.raw["mimi"]["codebook_size"],
        "semantic_code_dtype": "uint16",
        "audio_channel_policy": "average channels to mono",
        "resampler": "torchaudio.functional.resample",
        "target_audio_hours_per_shard": config.raw["output"]["target_audio_hours_per_shard"],
        "expected_audio_hours_by_split": {
            name: item.expected_audio_hours for name, item in config.splits.items()
        },
        "text_normalizer_version": config.raw["text"]["normalizer"],
        "processing_code_version": "aether-dataset==0.1.0",
        "created_at_utc": utc_now(),
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        stable_keys = set(payload) - {"created_at_utc"}
        mismatches = [key for key in stable_keys if existing.get(key) != payload[key]]
        if mismatches:
            raise RuntimeError(
                f"dataset_info mismatch in existing output for: {', '.join(mismatches)}"
            )
        return
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(encoded, encoding="utf-8")


def _write_processing_stats(output: Path) -> None:
    records = list(completed_shards(output).values())
    failures_path = output / "failures.jsonl"
    failure_count = 0
    if failures_path.exists():
        with failures_path.open("r", encoding="utf-8") as handle:
            failure_count = sum(1 for line in handle if line.strip())
    payload = {
        "created_at_utc": utc_now(),
        "completed_shards": len(records),
        "successful_examples": sum(int(item["examples"]) for item in records),
        "successful_audio_seconds": sum(float(item["audio_seconds"]) for item in records),
        "semantic_tokens": sum(int(item["semantic_tokens"]) for item in records),
        "failure_records": failure_count,
    }
    path = output / "processing_stats.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
