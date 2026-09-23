from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .benchmark import benchmark
from .config import ConfigError, load_config
from .io import read_jsonl
from .publish import DEFAULT_REPO_ID, publish_private_cache
from .runner import StopRequested, run
from .validation import validate_cache


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aether-dataset")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="build the Mimi semantic cache")
    run_parser.add_argument("--config", required=True, type=Path)
    run_parser.add_argument("--resume", action="store_true")

    benchmark = subparsers.add_parser("benchmark", help="benchmark batch correctness and speed")
    benchmark.add_argument("--config", required=True, type=Path)

    status = subparsers.add_parser("status", help="show saved cache progress")
    status.add_argument("--output", required=True, type=Path)

    validate = subparsers.add_parser("validate", help="validate completed cache shards")
    validate.add_argument("--output", required=True, type=Path)
    validate.add_argument("--skip-checksums", action="store_true")

    publish = subparsers.add_parser("publish", help="publish a validated private HF dataset")
    publish.add_argument("--output", required=True, type=Path)
    publish.add_argument("--repo", default=DEFAULT_REPO_ID)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console()
    try:
        if args.command == "run":
            run(load_config(args.config), resume=args.resume)
        elif args.command == "benchmark":
            result = benchmark(load_config(args.config))
            console.print_json(json.dumps(result))
        elif args.command == "status":
            _show_status(args.output, console)
        elif args.command == "validate":
            summary = validate_cache(args.output, verify_checksums=not args.skip_checksums)
            console.print_json(json.dumps(asdict(summary)))
        elif args.command == "publish":
            summary = publish_private_cache(args.output, repo_id=args.repo)
            console.print_json(json.dumps(asdict(summary)))
        return 0
    except StopRequested as error:
        console.print(f"[yellow]{error}[/yellow]")
        return 130
    except (ConfigError, FileNotFoundError, RuntimeError, ValueError) as error:
        Console(stderr=True).print(f"[red]error:[/red] {error}")
        return 2


def _show_status(output: Path, console: Console) -> None:
    records = [item for item in read_jsonl(output / "shards.jsonl") if item["status"] == "complete"]
    table = Table(title=f"Mimi cache: {output}")
    for heading in (
        "split",
        "shards",
        "examples",
        "audio hours",
        "progress",
        "semantic tokens",
    ):
        table.add_column(heading)
    expected: dict[str, float] = {}
    info_path = output / "dataset_info.json"
    if info_path.is_file():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        expected = info.get("expected_audio_hours_by_split", {})
    splits = sorted({str(item["split"]) for item in records})
    for split in splits:
        selected = [item for item in records if item["split"] == split]
        audio_hours = sum(float(item["audio_seconds"]) for item in selected) / 3600
        progress = (
            f"{min(audio_hours / float(expected[split]), 1.0) * 100:.2f}%"
            if split in expected
            else "n/a"
        )
        table.add_row(
            split,
            str(len(selected)),
            f"{sum(int(item['examples']) for item in selected):,}",
            f"{audio_hours:,.2f}",
            progress,
            f"{sum(int(item['semantic_tokens']) for item in selected):,}",
        )
    if not records:
        console.print("No completed shards.")
    else:
        console.print(table)
    metrics = read_jsonl(output / "run_metrics.jsonl")
    if metrics:
        console.print("Latest runtime metrics:")
        console.print_json(json.dumps(metrics[-1]))


if __name__ == "__main__":
    raise SystemExit(main())
