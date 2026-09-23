# aether-dataset

Restartable builder for the LoquaciousSet Mimi semantic cache described in
[`DATASET_SCHEMA.md`](DATASET_SCHEMA.md).

## Development

```bash
uv sync
uv run pytest
uv run aether-dataset --help
```

## Commands

Сначала задайте абсолютный output path в `configs/loquacious-medium.yaml`. Ревизии
LoquaciousSet и Mimi уже закреплены immutable commit SHA.

```bash
# Обязательный preflight на целевой GPU. Он проверяет batch/single equivalence
# и подбирает устойчивый batch budget.
uv run aether-dataset benchmark --config configs/loquacious-medium.yaml

# Полный запуск разрешён только при наличии совместимого benchmark.json.
uv run aether-dataset run --config configs/loquacious-medium.yaml
uv run aether-dataset run --config configs/loquacious-medium.yaml --resume
uv run aether-dataset status --output /data/loquacious-medium-mimi
uv run aether-dataset validate --output /data/loquacious-medium-mimi
```

`run` and `benchmark` intentionally refuse to start while required configuration
placeholders remain unresolved.

Полный прогон не следует запускать на локальном Mac: benchmark и extraction должны
выполняться в одном типе CUDA-окружения на целевой GPU.
