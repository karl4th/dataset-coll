from aether_dataset.normalization import normalize_english_v1


def test_normalizer_is_conservative_and_deterministic() -> None:
    assert normalize_english_v1("  It’s\tA  TEST. ") == "it's a test."
