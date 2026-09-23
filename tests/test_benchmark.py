from aether_dataset.benchmark import _fill_budget
from aether_dataset.mimi import PreparedAudio


def audio(seconds: float) -> PreparedAudio:
    return PreparedAudio(
        samples=object(),
        source_num_samples=int(seconds * 16000),
        source_sample_rate=16000,
        audio_seconds=seconds,
        model_num_samples=int(seconds * 24000),
    )


def test_fill_budget_limits_padded_audio_not_duration_sum() -> None:
    selected = _fill_budget([audio(1), audio(10), audio(1)], budget=25)
    assert [item.audio_seconds for item in selected] == [1, 10]
    assert max(item.audio_seconds for item in selected) * len(selected) <= 25
