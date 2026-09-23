from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from .io import append_jsonl, utc_now
from .resources import ResourceMonitor


@dataclass
class ProgressState:
    split: str
    shard_index: int
    examples: int = 0
    audio_seconds: float = 0.0
    semantic_tokens: int = 0
    errors: int = 0


class RunReporter:
    def __init__(
        self,
        output: Path,
        total_audio_seconds: float | None = None,
        initial_audio_seconds: float = 0.0,
    ) -> None:
        self.output = output
        self.total_audio_seconds = total_audio_seconds
        self.initial_audio_seconds = initial_audio_seconds
        self.started = time.monotonic()
        self.last_report = 0.0
        self.console = Console()
        self.resources = ResourceMonitor(output)

    def report(
        self,
        state: ProgressState,
        *,
        force: bool = False,
        interval: float = 5.0,
        prefetch_queue_depth: int | None = None,
    ) -> None:
        now = time.monotonic()
        if not force and now - self.last_report < interval:
            return
        self.last_report = now
        elapsed = max(now - self.started, 1e-6)
        session_audio_seconds = max(state.audio_seconds - self.initial_audio_seconds, 0.0)
        speed = session_audio_seconds / elapsed
        progress = None
        eta = None
        if self.total_audio_seconds:
            progress = min(state.audio_seconds / self.total_audio_seconds, 1.0)
            if speed > 0:
                eta = max(self.total_audio_seconds - state.audio_seconds, 0.0) / speed
        resources = self.resources.snapshot()
        payload = {
            "timestamp_utc": utc_now(),
            "split": state.split,
            "shard_index": state.shard_index,
            "examples": state.examples,
            "audio_seconds": state.audio_seconds,
            "semantic_tokens": state.semantic_tokens,
            "errors": state.errors,
            "elapsed_seconds": elapsed,
            "audio_seconds_per_second": speed,
            "realtime_factor": (1.0 / speed) if speed > 0 else None,
            "progress_fraction": progress,
            "eta_seconds": eta,
            "prefetch_queue_depth": prefetch_queue_depth,
            **resources.as_dict(),
        }
        append_jsonl(self.output / "run_metrics.jsonl", payload)
        percentage = f"{progress * 100:6.2f}%" if progress is not None else "   n/a"
        eta_text = _duration(eta) if eta is not None else "n/a"
        gpu = (
            f"GPU {resources.gpu_utilization_percent:.0f}% "
            f"VRAM {_gib(resources.gpu_memory_used_bytes)}/{_gib(resources.gpu_memory_total_bytes)}"
            if resources.gpu_utilization_percent is not None
            else "GPU metrics n/a"
        )
        self.console.print(
            f"[{state.split} shard {state.shard_index:05d}] {percentage} | "
            f"audio {_duration(state.audio_seconds)} | {speed:.1f}x realtime | "
            f"ETA {eta_text} | {gpu} | CPU {resources.cpu_percent:.0f}% | "
            f"queue {prefetch_queue_depth if prefetch_queue_depth is not None else 'n/a'} | "
            f"errors {state.errors}"
        )


def _duration(value: float | None) -> str:
    if value is None:
        return "n/a"
    seconds = max(int(value), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _gib(value: int | None) -> str:
    return "n/a" if value is None else f"{value / 2**30:.1f}GiB"
