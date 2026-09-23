import queue
from pathlib import Path

from aether_dataset.config import AppConfig
from aether_dataset.mimi import PreparedAudio
from aether_dataset.runner import PreparedBatch, _produce_batches
from aether_dataset.source import SourceExample


class FakeEncoder:
    def prepare(self, _array, sample_rate: int) -> PreparedAudio:
        return PreparedAudio(
            samples=object(),
            source_num_samples=sample_rate,
            source_sample_rate=sample_rate,
            audio_seconds=1.0,
            model_num_samples=24000,
        )


def make_example(index: int) -> SourceExample:
    return SourceExample(
        source_id=str(index),
        source_config="medium",
        source_split="train",
        source_shard="medium/train-00000.parquet",
        source_row_index=index,
        speaker_id="speaker",
        chapter_id="",
        source_audio_name=f"{index}.flac",
        original_text="Hello",
        audio_array=object(),
        sample_rate=16000,
    )


def test_prefetch_batches_preserve_source_order(tmp_path) -> None:
    config = AppConfig(
        raw={"text": {"normalizer": "english_v1"}},
        path=Path("config.yaml"),
    )
    destination: queue.Queue = queue.Queue()
    _produce_batches(
        [make_example(0), make_example(1), make_example(2)],
        FakeEncoder(),  # type: ignore[arg-type]
        config,
        tmp_path,
        2.0,
        {"requested": False},
        destination,
        None,
    )
    first = destination.get_nowait()
    second = destination.get_nowait()
    sentinel = destination.get_nowait()
    assert isinstance(first, PreparedBatch)
    assert isinstance(second, PreparedBatch)
    assert [item[0].source_row_index for item in first.items] == [0, 1]
    assert [item[0].source_row_index for item in second.items] == [2]
    assert sentinel is None
