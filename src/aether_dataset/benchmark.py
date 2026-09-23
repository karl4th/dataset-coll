from __future__ import annotations

import json
import time
from importlib.metadata import version
from itertools import cycle, islice
from typing import Any

from .config import AppConfig
from .io import sha256_file, utc_now
from .mimi import MimiEncoder, PreparedAudio
from .source import iter_source_examples

BATCHING_STRATEGY = "max_padded_audio_seconds_v2"


def benchmark(config: AppConfig, *, correctness_examples: int = 6) -> dict[str, Any]:
    config.require_ready()
    runtime = config.raw["runtime"]
    mimi_cfg = config.raw["mimi"]
    train = config.splits["train"]
    encoder = MimiEncoder(
        hf_repo=mimi_cfg["hf_repo"],
        revision=mimi_cfg["revision"],
        device=runtime["device"],
        semantic_codebook_index=int(mimi_cfg["semantic_codebook_index"]),
    )
    source = iter_source_examples(
        config.raw["dataset"]["id"],
        config.raw["dataset"]["revision"],
        train.config,
        train.split,
    )
    prepared: list[PreparedAudio] = []
    source_ids: list[str] = []
    for example in islice(source, correctness_examples):
        prepared.append(encoder.prepare(example.audio_array, example.sample_rate))
        source_ids.append(example.source_id)
    if len(prepared) < 2:
        raise RuntimeError("benchmark needs at least two source examples")

    individual = [encoder.encode_one(item) for item in prepared]
    batched = encoder.encode_batch(prepared)
    mismatches = [
        source_id
        for source_id, expected, actual in zip(source_ids, individual, batched, strict=True)
        if expected != actual
    ]
    if mismatches:
        raise RuntimeError(
            "batch/single Mimi mismatch; full run is blocked for samples: " + ", ".join(mismatches)
        )

    trials: list[dict[str, Any]] = []
    budget = float(runtime["min_batch_audio_seconds"])
    maximum = float(runtime["max_batch_audio_seconds"])
    target_fraction = float(runtime["target_vram_fraction"])
    recommended = budget
    last_safe_budget: float | None = None
    while budget <= maximum:
        trial_batch = _fill_budget(prepared, budget)
        try:
            if encoder.device.type == "cuda":
                encoder.torch.cuda.empty_cache()
                encoder.torch.cuda.reset_peak_memory_stats(encoder.device)
            _synchronize(encoder)
            started = time.perf_counter()
            encoder.encode_batch(trial_batch)
            _synchronize(encoder)
            elapsed = time.perf_counter() - started
            audio_seconds = sum(item.audio_seconds for item in trial_batch)
            memory_fraction = _peak_memory_fraction(encoder)
            trials.append(
                {
                    "budget_audio_seconds": budget,
                    "examples": len(trial_batch),
                    "actual_audio_seconds": audio_seconds,
                    "elapsed_seconds": elapsed,
                    "throughput_x_realtime": audio_seconds / elapsed,
                    "gpu_memory_fraction": memory_fraction,
                    "status": "ok",
                }
            )
            if memory_fraction is None or memory_fraction <= target_fraction:
                last_safe_budget = budget
                recommended = budget
            if memory_fraction is not None and memory_fraction >= target_fraction:
                break
            budget *= 2
        except Exception as error:
            if not _is_cuda_oom(encoder, error):
                raise
            encoder.torch.cuda.empty_cache()
            trials.append({"budget_audio_seconds": budget, "status": "oom"})
            break
    if last_safe_budget is None:
        raise RuntimeError("no batch fits the configured peak VRAM target")

    payload = {
        "created_at_utc": utc_now(),
        "dataset_revision": config.raw["dataset"]["revision"],
        "mimi_model_id": mimi_cfg["hf_repo"],
        "mimi_model_revision": mimi_cfg["revision"],
        "mimi_model_sha256": sha256_file(encoder.weights_path),
        "mimi_implementation": f"moshi=={version('moshi')}",
        "device": runtime["device"],
        "batch_single_equivalent": True,
        "batching_strategy": BATCHING_STRATEGY,
        "correctness_examples": len(prepared),
        "recommended_batch_audio_seconds": recommended,
        "trials": trials,
    }
    output = config.output_path
    output.mkdir(parents=True, exist_ok=True)
    path = output / "benchmark.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def require_matching_benchmark(config: AppConfig) -> dict[str, Any]:
    path = config.output_path / "benchmark.json"
    if not path.is_file():
        raise RuntimeError("benchmark.json is missing; run the benchmark command first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "dataset_revision": config.raw["dataset"]["revision"],
        "mimi_model_id": config.raw["mimi"]["hf_repo"],
        "mimi_model_revision": config.raw["mimi"]["revision"],
        "device": config.raw["runtime"]["device"],
        "batch_single_equivalent": True,
        "batching_strategy": BATCHING_STRATEGY,
    }
    mismatches = [key for key, value in expected.items() if payload.get(key) != value]
    if mismatches:
        raise RuntimeError(f"benchmark does not match config: {', '.join(mismatches)}")
    return payload


def _fill_budget(source: list[PreparedAudio], budget: float) -> list[PreparedAudio]:
    selected: list[PreparedAudio] = []
    maximum_seconds = 0.0
    for item in cycle(source):
        candidate_maximum = max(maximum_seconds, item.audio_seconds)
        if selected and candidate_maximum * (len(selected) + 1) > budget:
            return selected
        selected.append(item)
        maximum_seconds = candidate_maximum
        if maximum_seconds * len(selected) >= budget:
            return selected
    raise AssertionError("unreachable")


def _synchronize(encoder: MimiEncoder) -> None:
    if encoder.device.type == "cuda":
        encoder.torch.cuda.synchronize(encoder.device)


def _peak_memory_fraction(encoder: MimiEncoder) -> float | None:
    if encoder.device.type != "cuda":
        return None
    total = encoder.torch.cuda.get_device_properties(encoder.device).total_memory
    peak = encoder.torch.cuda.max_memory_allocated(encoder.device)
    return peak / total


def _is_cuda_oom(encoder: MimiEncoder, error: Exception) -> bool:
    message = str(error).lower()
    return isinstance(error, encoder.torch.cuda.OutOfMemoryError) or (
        "cuda" in message and "out of memory" in message
    )
