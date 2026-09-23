---
language:
- en
license:
- cc-by-3.0
- cc-by-4.0
task_categories:
- automatic-speech-recognition
tags:
- audio
- mimi
- semantic-tokens
- manifestro
---

# Manifestro Stage 1 Aether — LoquaciousSet Mimi Semantic Cache

Private, derived cache for Manifestro research. It contains pinned LoquaciousSet
audio metadata, original and normalized English transcripts, and the semantic
codebook stream produced by a pinned Kyutai Mimi checkpoint.

The cache does not redistribute decoded source audio. Source licenses and attribution
requirements remain applicable. Exact source/model revisions, preprocessing settings,
schema, checksums, failures, and processing statistics are included in the repository.

## Splits

- `train`: LoquaciousSet `medium/train`
- `validation`: official LoquaciousSet `dev`
- `test`: official LoquaciousSet `test`, reserved for final evaluation

See `DATASET_SCHEMA.md` and `dataset_info.json` before use.
