import pytest

from src.data_processing.loader import SAMPLE_PATH, load_ai4i


@pytest.mark.skipif(not SAMPLE_PATH.exists(), reason="demo sample not present")
def test_committed_sample_matches_verified_facts():
    df, report = load_ai4i(SAMPLE_PATH)
    assert report.passed
    assert len(df) == 10_000
    assert report.info["n_failures"] == 339
    assert report.info["failures_with_no_mode_flag"] == 9
    assert report.info["mode_flag_without_failure"] == 18