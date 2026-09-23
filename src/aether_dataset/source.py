from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePosixPath
from typing import Any


@dataclass(frozen=True)
class SourceExample:
    source_id: str
    source_config: str
    source_split: str
    source_shard: str
    source_row_index: int
    speaker_id: str
    chapter_id: str
    source_audio_name: str
    original_text: str
    audio_array: Any
    sample_rate: int


def list_parquet_shards(dataset_id: str, revision: str, config: str, split: str) -> list[str]:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(dataset_id, repo_type="dataset", revision=revision)
    prefix = f"{config}/"
    candidates = [
        name
        for name in files
        if name.startswith(prefix)
        and name.endswith(".parquet")
        and PurePosixPath(name).name.startswith(f"{split}-")
    ]
    if not candidates:
        raise FileNotFoundError(
            f"no parquet shards for {dataset_id}@{revision} config={config} split={split}"
        )
    return sorted(candidates)


def iter_source_examples(
    dataset_id: str,
    revision: str,
    config: str,
    split: str,
    *,
    start_after: tuple[str, int] | None = None,
) -> Iterator[SourceExample]:
    from datasets import Audio, load_dataset

    passed_cursor = start_after is None
    for shard in list_parquet_shards(dataset_id, revision, config, split):
        if not passed_cursor and start_after is not None and shard < start_after[0]:
            continue
        uri = f"hf://datasets/{dataset_id}@{revision}/{shard}"
        dataset = load_dataset(
            "parquet",
            data_files={"source": uri},
            split="source",
            streaming=True,
        )
        features = getattr(dataset, "features", None)
        if features is not None and "wav" in features:
            dataset = dataset.cast_column("wav", Audio(decode=False))
        for row_index, row in enumerate(dataset):
            if not passed_cursor:
                assert start_after is not None
                if shard == start_after[0] and row_index <= start_after[1]:
                    continue
                passed_cursor = True
            audio = row.get("wav") or row.get("audio")
            array, sample_rate, decoded_path = _decoded_audio(audio, shard, row_index)
            source_id = str(row.get("ID") or row.get("id") or f"{shard}:{row_index}")
            audio_path = decoded_path or row.get("file") or source_id
            yield SourceExample(
                source_id=source_id,
                source_config=config,
                source_split=split,
                source_shard=shard,
                source_row_index=row_index,
                speaker_id=str(row.get("spk_id") or row.get("speaker_id") or ""),
                chapter_id=str(row.get("chapter_id") or ""),
                source_audio_name=PurePosixPath(str(audio_path)).name,
                original_text=str(row.get("text") or row.get("transcript") or ""),
                audio_array=array,
                sample_rate=sample_rate,
            )


def _decoded_audio(audio: Any, shard: str, row_index: int) -> tuple[Any, int, str | None]:
    if isinstance(audio, dict) and "array" in audio:
        return audio["array"], int(audio["sampling_rate"]), audio.get("path")
    if isinstance(audio, dict) and audio.get("bytes") is not None:
        import soundfile as sf

        array, sample_rate = sf.read(BytesIO(audio["bytes"]), dtype="float32", always_2d=True)
        return array.T, int(sample_rate), audio.get("path")
    raise ValueError(f"missing decoded audio in {shard}:{row_index}")
