# Google Colab runbook

Используйте [`notebooks/stage1_aether_cache_colab.ipynb`](notebooks/stage1_aether_cache_colab.ipynb).

## До запуска

1. Код клонируется из `https://github.com/karl4th/dataset-coll.git`. Перед полным
   прогоном замените `CODE_REVISION = "main"` в первой ячейке notebook на SHA
   опубликованного коммита, содержащего текущую версию pipeline.
2. В Colab выберите GPU runtime.
3. Добавьте секрет `HF_TOKEN` через панель Colab Secrets. Токен должен иметь write
   access к организации `manifestro`.
4. Если Git-репозиторий приватный, добавьте `GITHUB_TOKEN` в Colab Secrets с
   минимальным read-only доступом к этому репозиторию.
5. Не вставляйте токены в Git URL, notebook, YAML, вывод ячеек или Google Drive.

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
