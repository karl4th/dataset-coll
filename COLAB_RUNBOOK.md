# Google Colab runbook

Используйте [`notebooks/stage1_aether_cache_colab.ipynb`](notebooks/stage1_aether_cache_colab.ipynb).

## До запуска

1. Код клонируется командой `git clone` из
   `https://github.com/karl4th/dataset-coll.git`. Перед полным прогоном желательно
   выполнить `git checkout --detach <COMMIT_SHA>`.
2. В Colab выберите GPU runtime.
3. Добавьте секрет `HF_TOKEN` через панель Colab Secrets. Токен должен иметь write
   access к организации `manifestro`.
4. Не вставляйте HF-токен в notebook, YAML, вывод ячеек или Google Drive.

Notebook использует:

- `git clone` в `/content/aether-dataset` для кода и окружения;
- `MyDrive/manifestro/stage1_aether_cache` для restartable output;
- приватный dataset repo `manifestro/stage1_aether` для финальной публикации.

GPU Colab не гарантируется даже в платных планах. Всегда сначала запускайте ячейки
preflight и benchmark. Если runtime оборвался, снова выполните setup, а затем `run
--resume`; завершённые шарды повторно не кодируются.

Публикация намеренно вынесена в последнюю отдельную ячейку. Она начнётся только
после полной checksum-валидации всех локальных шардов и всех трёх split.

Google Drive не используется для переноса кода. Он используется только для
restartable cache, `benchmark.json`, manifests и итоговых Parquet-шардов.
